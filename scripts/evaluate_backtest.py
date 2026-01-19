import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import clone

# --- UPDATED IMPORTS (Matching the Bot) ---
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# --- PATH SETUP ---
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.settings import DB_PATH
from config.all_symbols import SYMBOLS
from config.logging_config import setup_logging
from data.tick_storage import TickStorage
from data.tick_to_bars import ImbalanceBarGenerator
from strategies.risklabai_bot import RiskLabAIModel
from utils.financial_ml import PurgedCrossValidator

logger = setup_logging(script_name="backtest")

TEST_SPLIT = 0.20
# 0.0002 = 2 basis points (approx spread + fee on liquid ETFs like QQQ)
TRANSACTION_COST = 0.0002 

def parse_args():
    parser = argparse.ArgumentParser(description="Backtest RiskLabAI Strategy")
    parser.add_argument("--symbol", help="Symbol to test (e.g., QQQ)")
    parser.add_argument("--all", action="store_true", help="Test all symbols")
    return parser.parse_args()

def calculate_trade_metrics(positions, bar_returns):
    """
    Calculates metrics including Transaction Costs.
    """
    # 1. Gross Returns
    gross_returns = positions * bar_returns
    
    # 2. Calculate Costs
    # We pay a fee every time the position CHANGES (Buy or Sell)
    # prepend=0 assumes we start flat.
    position_changes = np.abs(np.diff(positions, prepend=0))
    costs = position_changes * TRANSACTION_COST
    
    # 3. Net Returns
    net_strategy_returns = gross_returns - costs
    cum_return = (1 + net_strategy_returns).prod() - 1
    
    # 4. Trade Stats
    # A "trade" is arguably an entry, so we count when position goes 0 -> 1/ -1
    # or just count total turnover. Let's count non-zero bars for exposure.
    trade_indices = np.nonzero(positions)[0]
    total_trades = np.sum(position_changes > 0) # Count execution events
    
    if total_trades == 0:
        return 0.0, 0, 0.0, 0.0
    
    # 5. Win Rate (on bars where we held a position)
    # Note: This is per-bar win rate, hard to do per-trade in vectorization
    # We will approximate by looking at positive return bars vs total active bars
    winning_bars = np.sum(net_strategy_returns > 0)
    total_active_bars = len(trade_indices)
    real_win_rate = winning_bars / total_active_bars if total_active_bars > 0 else 0
    
    # 6. Sharpe
    std = np.std(net_strategy_returns)
    sharpe = (np.mean(net_strategy_returns) / std) * np.sqrt(252 * 78) if std > 1e-9 else 0.0
    
    return cum_return, total_trades, real_win_rate, sharpe

def run_simulation(symbol):
    logger.info(f"--- Simulating {symbol} ---")
    
    # 1. Load Data
    storage = TickStorage(DB_PATH)
    ticks = storage.load_ticks(symbol)
    storage.close()
    
    if ticks.empty or len(ticks) < 1000:
        logger.warning(f"[{symbol}] Not enough data.")
        return None

    # 2. Generate Bars
    bars = ImbalanceBarGenerator.process_ticks(ticks, threshold=10000)
    if bars.empty: return None
    
    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    
    # 3. Generate Features
    model = RiskLabAIModel()
    features = model.generate_features(bars)
    
    if features.empty: return None
        
    # 4. Generate Labels
    close_prices = bars.loc[features.index, 'close']
    events = model.cusum.get_events(close_prices)
    labels = model.labeler.label(close_prices, pd.DataFrame(index=events))
    
    # Align Data
    common_idx = labels.index.intersection(features.index)
    X = features.loc[common_idx, model.feature_names]
    
    # --- FIX: Encode Labels for XGBoost (Same as Bot) ---
    le = LabelEncoder()
    y_raw = labels.loc[common_idx, 'bin']
    y_encoded = le.fit_transform(y_raw)
    y = pd.Series(y_encoded, index=common_idx)
    
    t1 = labels.loc[common_idx, 't1']
    
    # Normalize Timezones
    if X.index.tz is not None: X.index = X.index.tz_localize(None)
    if y.index.tz is not None: y.index = y.index.tz_localize(None)
    if t1.index.tz is not None: t1.index = t1.index.tz_localize(None)
    if pd.api.types.is_datetime64_any_dtype(t1) and hasattr(t1, 'dt') and getattr(t1.dt, 'tz', None) is not None:
        t1 = t1.dt.tz_localize(None)
    
    # 5. Split Train/Test
    split_idx = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    t1_train = t1.iloc[:split_idx]
    
    if len(X_test) < 10: return None

    logger.info(f"[{symbol}] Train: {len(X_train)} | Test: {len(X_test)}")

    # 6. Train Models (XGBoost + LogisticRegression)
    primary_base = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', XGBClassifier(
            n_estimators=100, 
            max_depth=3, 
            learning_rate=0.1, 
            eval_metric='mlogloss', 
            n_jobs=1
        ))
    ])
    
    # Generate Out-of-Sample Predictions
    meta_features = np.zeros(len(y_train))
    valid_indices_mask = np.zeros(len(y_train), dtype=bool)
    cv = PurgedCrossValidator(n_splits=5, embargo_pct=0.01)
    
    logger.info(f"[{symbol}] Training Meta-Model (Purged CV)...")
    try:
        cv_gen = cv.get_cv(t1_train)
        for train_idx, val_idx in cv_gen.split(X_train, y_train):
            temp_model = clone(primary_base)
            temp_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            val_preds = temp_model.predict(X_train.iloc[val_idx])
            meta_features[val_idx] = val_preds
            valid_indices_mask[val_idx] = True
    except Exception as e:
        logger.warning(f"Purged CV Error: {e}")
        return None

    X_meta_train = X_train.iloc[valid_indices_mask]
    y_meta_train = (meta_features[valid_indices_mask] == y_train.iloc[valid_indices_mask]).astype(int)
    
    # Final Training
    primary_final = clone(primary_base)
    primary_final.fit(X_train, y_train)
    
    # Meta Model: Logistic Regression
    meta_final = LogisticRegression(class_weight='balanced', solver='liblinear')
    meta_final.fit(X_meta_train, y_meta_train)
    
    # 7. Predictions
    p_preds_encoded = primary_final.predict(X_test)
    
    # Decode signals back to -1, 0, 1
    p_preds_test = le.inverse_transform(p_preds_encoded)
    
    # Meta Confidence
    if hasattr(meta_final, "predict_proba"):
        probs = meta_final.predict_proba(X_test)
        meta_probs = probs[:, 1] if probs.shape[1] > 1 else np.zeros(len(X_test))
    else:
        meta_probs = meta_final.predict(X_test)
    
    # 8. THRESHOLD OPTIMIZATION LOOP
    logger.info(f"[{symbol}] Optimizing Threshold (With Fees)...")
    
    best_res = None
    best_sharpe = -999
    
    print(f"\n{'Threshold':<10} | {'Trades':<8} | {'Win Rate':<10} | {'Net Return':<10}")
    print("-" * 55)
    
    bar_returns = bars.loc[X_test.index, 'close'].pct_change().shift(-1).fillna(0)
    
    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        positions = np.zeros(len(X_test))
        for i in range(len(X_test)):
            signal = p_preds_test[i]
            conf = meta_probs[i]
            if signal != 0 and conf > threshold:
                positions[i] = signal 
        
        ret, trades, win_rate, sharpe = calculate_trade_metrics(positions, bar_returns)
        
        print(f"{threshold:<10} | {trades:<8} | {win_rate:<10.1%} | {ret:<10.2%}")
        
        if sharpe > best_sharpe and trades > 10:
            best_sharpe = sharpe
            best_res = {
                "Symbol": symbol,
                "Best_Thresh": threshold,
                "Net Return": f"{ret:.2%}",
                "Trades": trades,
                "Win Rate": f"{win_rate:.1%}",
                "Raw_Net": ret
            }

    if best_res:
        logger.info(f"[{symbol}] BEST: Thresh={best_res['Best_Thresh']} | Return={best_res['Net Return']}")
        return best_res
    else:
        logger.warning(f"[{symbol}] No profitable threshold found.")
        return None

def main():
    args = parse_args()
    targets = []
    if args.symbol: targets = [args.symbol.upper()]
    elif args.all: targets = SYMBOLS
    else: 
        logger.error("Please specify --symbol <TICKER> or --all")
        return

    results = []
    for sym in targets:
        try:
            res = run_simulation(sym)
            if res: results.append(res)
        except Exception as e:
            logger.error(f"Error testing {sym}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*60)
        print("OPTIMIZED BACKTEST RESULTS (Net of Fees)")
        print("="*60)
        print(df[["Symbol", "Best_Thresh", "Net Return", "Win Rate", "Trades"]].to_string(index=False))
        print("="*60)

if __name__ == "__main__":
    main()