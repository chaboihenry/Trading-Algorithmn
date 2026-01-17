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
    
    # 3. Generate Features
    model = RiskLabAIModel()
    features = model.generate_features(bars)
    
    if features.empty:
        logger.warning(f"[{symbol}] No features generated.")
        return None
        
    # 4. Generate Labels & Get Barrier Times (t1)
    close_prices = bars.loc[features.index, 'close']
    events = model.cusum.get_events(close_prices)
    labels = model.labeler.label(close_prices, pd.DataFrame(index=events))
    
    # Align Data
    common_idx = labels.index.intersection(features.index)
    X = features.loc[common_idx, model.feature_names]
    y = labels.loc[common_idx, 'bin']
    t1 = labels.loc[common_idx, 't1'] # Critical for Purged CV
    
    # 5. Split Train (Past) vs Test (Future)
    split_idx = int(len(X) * (1 - TEST_SPLIT))
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    t1_train = t1.iloc[:split_idx]
    
    if len(X_test) < 10:
        logger.warning(f"[{symbol}] Test set too small.")
        return None

    logger.info(f"[{symbol}] Train: {len(X_train)} events | Test: {len(X_test)} events")

    # 6. Train Models (THE RISKLABAI WAY)
    
    # A. Define Base Primary Model
    primary_base = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', n_jobs=1))
    ])
    
    # B. Generate Out-of-Sample Predictions for Meta-Model Training
    # We use Purged K-Fold to generate predictions on the training set
    # without looking ahead or using overlapping trades.
    
    # Initialize array to store the unbiased predictions
    # We fill with 0 initially; any skipped points (due to purging) will be ignored later.
    meta_features = np.zeros(len(y_train))
    valid_indices_mask = np.zeros(len(y_train), dtype=bool)
    
    # Use the Purged Cross Validator from your Utils
    cv = PurgedCrossValidator(n_splits=5, embargo_pct=0.01)
    
    logger.info(f"[{symbol}] Generating Meta-Features via Purged CV...")
    
    # Custom CV Loop
    # We pass the t1 series (end times) which PurgedKFold uses to prevent leakage
    try:
        cv_gen = cv.get_cv(t1_train)
        
        for train_idx, val_idx in cv_gen.split(X_train, y_train):
            # 1. Clone a fresh primary model
            temp_model = clone(primary_base)
            
            # 2. Fit on the Fold's Training Set
            temp_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            
            # 3. Predict on the Fold's Validation Set (The "Out of Sample" part)
            val_preds = temp_model.predict(X_train.iloc[val_idx])
            
            # 4. Store predictions
            # Map validation indices back to our meta_features array position
            meta_features[val_idx] = val_preds
            valid_indices_mask[val_idx] = True
            
    except Exception as e:
        logger.error(f"Purged CV Failed: {e}. Falling back to standard CV.")
        from sklearn.model_selection import cross_val_predict
        meta_features = cross_val_predict(primary_base, X_train, y_train, cv=5)
        valid_indices_mask[:] = True

    # C. Prepare Meta-Model Training Data
    # Only use points where we successfully generated a prediction (purging might drop some)
    X_meta_train = X_train.iloc[valid_indices_mask]
    y_meta_train = (meta_features[valid_indices_mask] == y_train.iloc[valid_indices_mask]).astype(int)
    
    # D. Train the Actual Models for Future Use
    
    # 1. Fit Final Primary Model (on ALL training data)
    primary_final = clone(primary_base)
    primary_final.fit(X_train, y_train)
    
    # 2. Fit Final Meta Model (on the Unbiased predictions we just generated)
    if len(np.unique(y_meta_train)) < 2:
         logger.warning(f"[{symbol}] Warning: Meta-Model targets are mono-class. Check Primary Model variance.")
         
    meta_final = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=5),
        n_estimators=50,
        max_samples=0.6,
        n_jobs=1
    )
    meta_final.fit(X_meta_train, y_meta_train)
    
    # 7. Simulate Trading (On Future Data)
    p_preds_test = primary_final.predict(X_test)
    
    # Get Meta Confidence
    if hasattr(meta_final, "predict_proba"):
        try:
            probs = meta_final.predict_proba(X_test)
            if probs.shape[1] > 1:
                meta_probs = probs[:, 1]
            else:
                single_class = meta_final.classes_[0]
                meta_probs = np.ones(len(X_test)) if single_class == 1 else np.zeros(len(X_test))
        except Exception as e:
            logger.error(f"Probability error: {e}")
            meta_probs = np.zeros(len(X_test))
    else:
        meta_probs = meta_final.predict(X_test)
    
    # Apply Strategy Logic
    meta_threshold = 0.6
    positions = np.zeros(len(X_test))
    
    for i in range(len(X_test)):
        signal = p_preds_test[i]
        conf = meta_probs[i]
        
        if signal != 0 and conf > meta_threshold:
            positions[i] = signal * conf 
        else:
            positions[i] = 0

    # 8. Calculate Returns
    bar_returns = bars.loc[X_test.index, 'close'].pct_change().shift(-1).fillna(0)
    strategy_returns = positions * bar_returns
    
    cum_return = (1 + strategy_returns).prod() - 1
    
    result = {
        "Symbol": symbol,
        "Net Return": f"{cum_return:.2%}",
        "Trades": np.count_nonzero(positions),
        "Win Rate": f"{np.mean(strategy_returns > 0):.1%}",
        "Raw_Net": cum_return
    }
    
    logger.info(f"[{symbol}] Result: {result['Net Return']} ({result['Trades']} trades)")
    return result

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
            
    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*50)
        print("BACKTEST RESULTS (Out-of-Sample)")
        print("="*50)
        print(df[["Symbol", "Net Return", "Win Rate", "Trades"]].to_string(index=False))
        print("="*50)

if __name__ == "__main__":
    main()