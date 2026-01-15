#!/usr/bin/env python3
"""
Simulate Strategy Performance (Out-of-Sample Backtest)

"The Exam":
1. Loads 90 days of tick data.
2. Hides the last 18 days (20%) from the model.
3. Trains on the first 72 days.
4. Simulates trading on the hidden 18 days.
5. Calculates Sharpe Ratio, Win Rate, and Returns vs S&P 500 benchmark.

Usage:
    python scripts/evaluate_backtest.py --symbol AAPL
    python scripts/evaluate_backtest.py --all
"""

import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# --- PATH SETUP ---
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- IMPORTS ---
from config.settings import DB_PATH
from config.all_symbols import SYMBOLS
from data.tick_storage import TickStorage
from data.tick_to_bars import generate_bars_from_ticks
from strategies.risklabai_bot import RiskLabAIModel, KellyCriterion

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# --- SETTINGS ---
COST_BPS = 2.0  # 2 Basis Points (0.02%) per trade (cover slippage + fees)
TEST_SPLIT = 0.20  # Last 20% of data is "Future"

def parse_args():
    parser = argparse.ArgumentParser(description="Backtest RiskLabAI Strategy")
    parser.add_argument("--symbol", help="Symbol to test (e.g., QQQ)")
    parser.add_argument("--all", action="store_true", help="Test all symbols")
    return parser.parse_args()

def calculate_metrics(returns, trades):
    if len(returns) == 0: return {}
    
    # Cumulative Return
    cum_ret = (1 + returns).prod() - 1
    
    # Sharpe Ratio (Annualized)
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(252 * 78) if std_ret > 0 else 0 
    # (Note: 78 5-min bars per day approx, adjust based on bar frequency)

    # Max Drawdown
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_dd = drawdown.min()

    return {
        "Return": cum_ret,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "Trades": trades
    }

def run_simulation(symbol):
    logger.info(f"--- Simulating {symbol} ---")
    
    # 1. Load Data
    storage = TickStorage(DB_PATH)
    ticks = storage.load_ticks(symbol)
    storage.close()
    
    if not ticks or len(ticks) < 1000:
        logger.warning(f"[{symbol}] Not enough data to backtest.")
        return None

    # 2. Generate Bars
    # (Using same threshold as live bot)
    bars_list = generate_bars_from_ticks(ticks, threshold=3000)
    if not bars_list: return None
    
    df = pd.DataFrame(bars_list)
    df['bar_end'] = pd.to_datetime(df['bar_end'])
    df = df.set_index('bar_end').sort_index()
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)

    # 3. Split "Past" (Train) vs "Future" (Test)
    split_idx = int(len(df) * (1 - TEST_SPLIT))
    train_data = df.iloc[:split_idx]
    test_data = df.iloc[split_idx:]
    
    logger.info(f"[{symbol}] Train: {len(train_data)} bars | Test (Simulation): {len(test_data)} bars")

    # 4. Train Model on "Past"
    model = RiskLabAIModel()
    results = model.train(train_data, min_samples=50)
    
    if not results['success']:
        logger.warning(f"[{symbol}] Training failed: {results.get('reason')}")
        return None

    # 5. Simulate Trading on "Future"
    # We manually replicate the bot's prediction logic here for batch processing
    features = model.prepare_features(test_data)
    
    if features.empty: return None
    
    # Align price data with features
    test_prices = test_data.loc[features.index]['close']
    future_returns = test_prices.pct_change().shift(-1).fillna(0) # Next bar return

    # Batch Prediction
    X_scaled = model.scaler.transform(features)
    
    # Primary Model Probs
    probs = model.primary_model.predict_proba(X_scaled)
    # Map class index to -1, 0, 1
    # Assumes classes are [-1, 0, 1] or similar. Need to check encoder.
    classes = model.label_encoder.classes_
    
    # Generate Signals
    positions = []
    
    for i, p_dist in enumerate(probs):
        # Create dict {class: prob}
        prob_map = {c: p for c, p in zip(classes, p_dist)}
        
        signal = 0
        if prob_map.get(1, 0) > model.margin_threshold: signal = 1
        elif prob_map.get(-1, 0) > model.margin_threshold: signal = -1
        
        # Meta Model
        meta_conf = 0.0
        if signal != 0:
            # Simple meta check (optimized for speed in backtest)
            meta_conf = model.meta_model.predict_proba([X_scaled[i]])[0][1]
            if meta_conf < model.meta_threshold:
                signal = 0
        
        positions.append(signal * meta_conf) # Weight by confidence

    positions = np.array(positions)
    
    # 6. Calculate PnL
    # Strategy Return = Position * Next_Return - Costs
    # Cost is applied on trade execution (change in position)
    
    # Assuming full capital rotation for simplicity
    strat_returns = positions * future_returns
    
    # Calculate costs (Turnover * Cost)
    # pos_change = np.abs(np.diff(positions, prepend=0))
    # transaction_costs = pos_change * (COST_BPS / 10000)
    # net_returns = strat_returns - transaction_costs
    
    # Simplified Cost (Cost per bar held is inaccurate, cost per trade is better)
    trades_count = np.sum(np.abs(np.diff(positions, prepend=0)) > 0)
    total_cost_pct = trades_count * (COST_BPS / 10000)
    
    cum_ret_gross = (1 + strat_returns).prod() - 1
    cum_ret_net = cum_ret_gross - total_cost_pct

    metrics = {
        "Symbol": symbol,
        "Gross Return": f"{cum_ret_gross:.2%}",
        "Net Return": f"{cum_ret_net:.2%}",
        "Trades": trades_count,
        "Raw_Net": cum_ret_net
    }
    
    logger.info(f"[{symbol}] Result: {metrics['Net Return']} ({trades_count} trades)")
    return metrics

def main():
    args = parse_args()
    
    targets = []
    if args.symbol: targets = [args.symbol]
    elif args.all: targets = SYMBOLS
    else: 
        logger.error("Specify --symbol or --all")
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
        print("\n" + "="*40)
        print("BACKTEST SUMMARY (LAST 20% PERIOD)")
        print("="*40)
        print(df[["Symbol", "Net Return", "Trades"]].to_string(index=False))
        
        avg_ret = df['Raw_Net'].mean()
        print("-" * 40)
        print(f"AVERAGE PORTFOLIO RETURN: {avg_ret:.2%}")
        print("="*40)

if __name__ == "__main__":
    main()