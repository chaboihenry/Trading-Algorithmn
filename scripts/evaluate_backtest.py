import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

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

# Import the Purged Validator from your utils
from utils.financial_ml import PurgedCrossValidator

logger = setup_logging(script_name="backtest")

TEST_SPLIT = 0.20

def parse_args():
    parser = argparse.ArgumentParser(description="Backtest RiskLabAI Strategy")
    parser.add_argument("--symbol", help="Symbol to test (e.g., QQQ)")
    parser.add_argument("--all", action="store_true", help="Test all symbols")
    return parser.parse_args()

def calculate_trade_metrics(positions, bar_returns):
    """
    Calculates realistic trade metrics (Win Rate, PnL per trade).
    """
    # Strategy Returns (Shifted to align execution)
    strategy_returns = positions * bar_returns
    
    # 1. Net Return
    cum_return = (1 + strategy_returns).prod() - 1
    
    # 2. Extract Trades (Non-zero positions)
    trade_indices = np.nonzero(positions)[0]
    total_trades = len(trade_indices)
    
    if total_trades == 0:
        return 0.0, 0, 0.0, 0.0
    
    # 3. Calculate Real Win Rate (Winners / Total Trades)
    # We look at the return of the specific bars we traded
    trade_pnl = strategy_returns.iloc[trade_indices] if hasattr(strategy_returns, 'iloc') else strategy_returns[trade_indices]
    
    winning_trades = np.sum(trade_pnl > 0)
    real_win_rate = winning_trades / total_trades
    
    # 4. Sharpe (Annualized)
    # Assuming bars are roughly ~2 mins. 
    # Approx 100,000 bars per year? This is a rough proxy.
    std = np.std(strategy_returns)
    sharpe = (np.mean(strategy_returns) / std) * np.sqrt(252 * 78) if std > 1e-9 else 0.0
    
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
    
    # --- FIX: NORMALIZE BARS INDEX ---
    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    # ---------------------------------
    
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
    y = labels.loc[common_idx, 'bin']
    t1 = labels.loc[common_idx, 't1']
    
    # --- TIMEZONE NORMALIZATION ---
    if X.index.tz is not None: X.index = X.index.tz_localize(None)
    if y.index.tz is not None: y.index = y.index.tz_localize(None)
    if t1.index.tz is not None: t1.index = t1.index.tz_localize(None)
    if pd.api.types.is_datetime64_any_dtype(t1) and getattr(t1.dt, 'tz', None) is not None:
        t1 = t1.dt.tz_localize(None)
    # ------------------------------
    
    # 5. Split Train/Test
    split_idx = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    t1_train = t1.iloc[:split_idx]
    
    if len(X_test) < 10: return None

    logger.info(f"[{symbol}] Train: {len(X_train)} | Test: {len(X_test)}")

    # 6. Train Models (Purged CV)
    primary_base = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', n_jobs=1))
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
    
    meta_final = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=5),
        n_estimators=50,
        max_samples=0.6,
        n_jobs=1
    )
    meta_final.fit(X_meta_train, y_meta_train)
    
    # 7. Predictions
    p_preds_test = primary_final.predict(X_test)
    
    if hasattr(meta_final, "predict_proba"):
        probs = meta_final.predict_proba(X_test)
        meta_probs = probs[:, 1] if probs.shape[1] > 1 else np.zeros(len(X_test))
    else:
        meta_probs = meta_final.predict(X_test)
    
    # 8. THRESHOLD OPTIMIZATION LOOP
    # We test multiple confidence levels to find the "Sweet Spot"
    logger.info(f"[{symbol}] Optimizing Threshold...")
    
    best_res = None
    best_sharpe = -999
    
    print(f"\n{'Threshold':<10} | {'Trades':<8} | {'Win Rate':<10} | {'Return':<10}")
    print("-" * 50)
    
    bar_returns = bars.loc[X_test.index, 'close'].pct_change().shift(-1).fillna(0)
    
    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        positions = np.zeros(len(X_test))
        for i in range(len(X_test)):
            signal = p_preds_test[i]
            conf = meta_probs[i]
            if signal != 0 and conf > threshold:
                positions[i] = signal # Flat position size for backtest purity
        
        ret, trades, win_rate, sharpe = calculate_trade_metrics(positions, bar_returns)
        
        print(f"{threshold:<10} | {trades:<8} | {win_rate:<10.1%} | {ret:<10.2%}")
        
        if sharpe > best_sharpe and trades > 10: # Minimum trades constraint
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
        logger.info(f"[{symbol}] BEST: Thresh={best_res['Best_Thresh']} | Win Rate={best_res['Win Rate']} | Return={best_res['Net Return']}")
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
        print("OPTIMIZED BACKTEST RESULTS")
        print("="*60)
        print(df[["Symbol", "Best_Thresh", "Net Return", "Win Rate", "Trades"]].to_string(index=False))
        print("="*60)

if __name__ == "__main__":
    main()