#!/usr/bin/env python3
"""
Parallel Parameter Sweep Worker

Each worker processes a subset of parameter combinations.
Usage: python parameter_sweep_parallel.py --worker-id 0 --total-workers 6
"""

import sys
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy
from config.tick_config import TICK_DB_PATH, INITIAL_IMBALANCE_THRESHOLD
from data.tick_storage import TickStorage
from data.tick_to_bars import generate_bars_from_ticks

logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def backtest_configuration(strategy, bars_df, config):
    """Run backtest for a single parameter configuration."""

    # Extract config parameters for this backtest
    profit_target = config['profit_target']
    stop_loss = config['stop_loss']
    max_holding_period = 20  # Fixed max holding period

    # Track performance
    equity = 10000
    equity_curve = [equity]
    trades = []
    position = 0
    entry_price = 0
    entry_bar = 0

    for i in range(100, len(bars_df)):
        bars = bars_df.iloc[:i].copy()
        current_bar = bars_df.iloc[i]

        # Get signal with model's optimized prob threshold (1.5%)
        signal, bet_size = strategy.predict(bars, prob_threshold=0.015)

        # Override meta threshold
        if signal != 0:
            features = strategy.prepare_features(bars)
            if len(features) > 0:
                X = features.iloc[[-1]]
                if strategy.meta_model is not None:
                    trade_prob = strategy.meta_model.predict_proba(X)[0, 1]
                    if trade_prob < config['meta_threshold']:
                        signal = 0
                        bet_size = 0.0

        # Position management
        if position == 0 and signal != 0:
            # Enter position
            position = signal
            entry_price = current_bar['close']
            entry_bar = i

        elif position != 0:
            # Check exit conditions
            pnl_pct = (current_bar['close'] - entry_price) / entry_price * position
            bars_held = i - entry_bar

            exit_trade = False
            exit_reason = None

            # Profit target
            if pnl_pct >= profit_target:
                exit_trade = True
                exit_reason = 'profit_target'

            # Stop loss
            elif pnl_pct <= -stop_loss:
                exit_trade = True
                exit_reason = 'stop_loss'

            # Max holding period
            elif bars_held >= max_holding_period:
                exit_trade = True
                exit_reason = 'max_holding'

            if exit_trade:
                pnl = equity * pnl_pct
                equity += pnl
                equity_curve.append(equity)

                trades.append({
                    'entry_bar': entry_bar,
                    'exit_bar': i,
                    'direction': position,
                    'entry_price': entry_price,
                    'exit_price': current_bar['close'],
                    'pnl_pct': pnl_pct,
                    'pnl': pnl,
                    'exit_reason': exit_reason
                })

                position = 0

    # Calculate metrics
    if len(trades) == 0:
        return None

    trades_df = pd.DataFrame(trades)

    # Calculate returns
    returns = trades_df['pnl_pct'].values

    # Monthly return (annualized)
    total_bars = len(bars_df)
    num_months = total_bars / (252 * 78)  # Assume 78 bars per day, 252 trading days/year
    total_return = (equity - 10000) / 10000
    monthly_return = (1 + total_return) ** (1 / max(num_months, 0.1)) - 1

    # Sharpe ratio
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Win rate
    wins = (trades_df['pnl'] > 0).sum()
    win_rate = wins / len(trades_df)

    # Max drawdown
    equity_curve = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        'meta_threshold': config['meta_threshold'],
        'profit_target': config['profit_target'],
        'stop_loss': config['stop_loss'],
        'monthly_return': monthly_return,
        'sharpe_ratio': sharpe,
        'win_rate': win_rate,
        'num_trades': len(trades_df),
        'max_drawdown': max_drawdown,
        'final_equity': equity
    }


def run_worker(worker_id, total_workers, use_reduced_data=True):
    """Run a worker that processes a subset of configurations."""

    print(f"\n{'='*80}", flush=True)
    print(f"WORKER {worker_id} STARTING", flush=True)
    print(f"{'='*80}\n", flush=True)

    # Load model
    model_path = 'models/risklabai_tick_models_optimized.pkl'
    print(f"Worker {worker_id}: Loading model: {model_path}", flush=True)
    strategy = RiskLabAIStrategy(
        profit_taking=0.5,
        stop_loss=0.5,
        max_holding=20,
        n_cv_splits=5
    )
    strategy.load_models(model_path)
    print(f"Worker {worker_id}: ✓ Model loaded", flush=True)

    # Load data
    print(f"Worker {worker_id}: Loading bar data...", flush=True)
    storage = TickStorage(TICK_DB_PATH)
    ticks = storage.load_ticks('SPY')
    storage.close()

    bars_list = generate_bars_from_ticks(ticks, threshold=INITIAL_IMBALANCE_THRESHOLD)
    bars_df = pd.DataFrame(bars_list)
    bars_df['bar_end'] = pd.to_datetime(bars_df['bar_end'])
    if bars_df['bar_end'].dt.tz is not None:
        bars_df['bar_end'] = bars_df['bar_end'].dt.tz_localize(None)
    bars_df.set_index('bar_end', inplace=True)

    # OPTIMIZATION: Use only last 2000 bars for faster processing
    if use_reduced_data:
        bars_df = bars_df.iloc[-2000:].copy()
        print(f"Worker {worker_id}: Using last 2000 bars (FAST MODE)", flush=True)
    else:
        print(f"Worker {worker_id}: Using all {len(bars_df)} bars", flush=True)

    # Define all parameter combinations
    meta_thresholds = [0.0001, 0.001, 0.01, 0.10, 0.30, 0.50, 0.60, 0.70]

    # Profit/stop configurations (asymmetric ratios for higher returns)
    profit_stop_configs = [
        (0.02, 0.01),   # 1:2 R:R
        (0.03, 0.015),  # 1:2 R:R
        (0.04, 0.02),   # 1:2 R:R
        (0.03, 0.01),   # 1:3 R:R
        (0.04, 0.015),  # ~1:2.7 R:R
        (0.05, 0.02),   # 1:2.5 R:R
    ]

    # Generate all combinations
    all_combinations = []
    for meta_t, (profit, stop) in product(
        meta_thresholds, profit_stop_configs
    ):
        all_combinations.append({
            'meta_threshold': meta_t,
            'profit_target': profit,
            'stop_loss': stop
        })

    total_configs = len(all_combinations)
    print(f"Worker {worker_id}: Total configurations: {total_configs}", flush=True)

    # Split work among workers
    configs_per_worker = total_configs // total_workers
    start_idx = worker_id * configs_per_worker
    if worker_id == total_workers - 1:
        end_idx = total_configs  # Last worker takes remaining configs
    else:
        end_idx = start_idx + configs_per_worker

    my_configs = all_combinations[start_idx:end_idx]

    print(f"Worker {worker_id}: Processing configs {start_idx} to {end_idx-1} ({len(my_configs)} total)", flush=True)
    print(f"Worker {worker_id}: Starting backtests...\n", flush=True)

    # Process configurations
    results = []
    start_time = datetime.now()

    for idx, config in enumerate(my_configs):
        result = backtest_configuration(strategy, bars_df, config)

        if result is not None:
            results.append(result)

        # Progress update every 10 configs
        if (idx + 1) % 10 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            configs_per_sec = (idx + 1) / elapsed
            remaining = len(my_configs) - (idx + 1)
            eta_seconds = remaining / configs_per_sec if configs_per_sec > 0 else 0
            eta_minutes = eta_seconds / 60

            print(f"Worker {worker_id}: Progress {idx+1}/{len(my_configs)} ({(idx+1)/len(my_configs)*100:.1f}%) - "
                  f"ETA: {eta_minutes:.1f} min", flush=True)

    # Save results
    if len(results) > 0:
        results_df = pd.DataFrame(results)
        output_file = f'results/parameter_sweep_worker_{worker_id}.csv'
        results_df.to_csv(output_file, index=False)

        print(f"\nWorker {worker_id}: ✓ Saved {len(results)} results to {output_file}", flush=True)

        # Show best result for this worker
        best = results_df.nlargest(1, 'monthly_return').iloc[0]
        print(f"Worker {worker_id}: Best config - Monthly: {best['monthly_return']:.2%}, "
              f"Sharpe: {best['sharpe_ratio']:.2f}, Meta: {best['meta_threshold']:.4f}", flush=True)
    else:
        print(f"\nWorker {worker_id}: ❌ No valid results", flush=True)

    total_time = (datetime.now() - start_time).total_seconds() / 60
    print(f"Worker {worker_id}: ✓ COMPLETE in {total_time:.1f} minutes", flush=True)
    print(f"{'='*80}\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallel parameter sweep worker')
    parser.add_argument('--worker-id', type=int, required=True, help='Worker ID (0-indexed)')
    parser.add_argument('--total-workers', type=int, required=True, help='Total number of workers')
    parser.add_argument('--full-data', action='store_true', help='Use full dataset (slower)')

    args = parser.parse_args()

    try:
        run_worker(args.worker_id, args.total_workers, use_reduced_data=not args.full_data)
    except Exception as e:
        print(f"\nWorker {args.worker_id}: ❌ ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
