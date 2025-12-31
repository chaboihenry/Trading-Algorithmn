#!/usr/bin/env python3
"""
Combine Parallel Sweep Results

Merges results from all workers and generates final analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def combine_results():
    print("\n" + "="*80)
    print("COMBINING PARALLEL SWEEP RESULTS")
    print("="*80 + "\n")

    # Find all worker result files
    results_dir = Path('results')
    worker_files = sorted(results_dir.glob('parameter_sweep_worker_*.csv'))

    if len(worker_files) == 0:
        print("❌ No worker result files found!")
        return None

    print(f"Found {len(worker_files)} worker result files:")
    for f in worker_files:
        print(f"  - {f.name}")

    # Combine all results
    all_results = []
    for worker_file in worker_files:
        df = pd.read_csv(worker_file)
        all_results.append(df)
        print(f"  ✓ Loaded {len(df)} results from {worker_file.name}")

    combined_df = pd.concat(all_results, ignore_index=True)
    print(f"\n✓ Combined total: {len(combined_df)} configurations\n")

    # Save combined results
    combined_file = 'results/parameter_sweep_results_combined.csv'
    combined_df.to_csv(combined_file, index=False)
    print(f"💾 Saved to: {combined_file}\n")

    # Analyze results
    print("="*80)
    print("ANALYSIS")
    print("="*80 + "\n")

    # Filter valid configurations
    valid_configs = combined_df[
        (combined_df['num_trades'] >= 10) &
        (combined_df['max_drawdown'] > -0.20)
    ].copy()

    print(f"📊 VALID CONFIGURATIONS: {len(valid_configs)} / {len(combined_df)}\n")

    if len(valid_configs) == 0:
        print("❌ No valid configurations found!")
        return combined_df

    # Top performers by monthly return
    print("🏆 TOP 10 BY MONTHLY RETURN:")
    top_monthly = valid_configs.nlargest(10, 'monthly_return')
    print(top_monthly[[
        'meta_threshold', 'profit_target', 'stop_loss',
        'monthly_return', 'sharpe_ratio', 'win_rate',
        'num_trades', 'max_drawdown'
    ]].to_string(index=False))
    print()

    # Top performers by Sharpe ratio
    print("📈 TOP 10 BY SHARPE RATIO:")
    top_sharpe = valid_configs.nlargest(10, 'sharpe_ratio')
    print(top_sharpe[[
        'meta_threshold', 'profit_target', 'stop_loss',
        'sharpe_ratio', 'monthly_return', 'win_rate',
        'num_trades', 'max_drawdown'
    ]].to_string(index=False))
    print()

    # Configurations achieving 2%+ monthly
    print("🎯 CONFIGURATIONS ACHIEVING 2%+ MONTHLY:")
    target_achievers = valid_configs[valid_configs['monthly_return'] >= 0.02]

    if len(target_achievers) > 0:
        print(f"   ✅ Found {len(target_achievers)} configurations meeting target!\n")
        best_target = target_achievers.nlargest(1, 'sharpe_ratio').iloc[0]

        print("   RECOMMENDED CONFIGURATION:")
        print(f"   Meta Threshold: {best_target['meta_threshold']:.4f}")
        print(f"   Profit Target: {best_target['profit_target']:.2%}")
        print(f"   Stop Loss: {best_target['stop_loss']:.2%}")
        print(f"   Risk:Reward: 1:{best_target['profit_target']/best_target['stop_loss']:.1f}")
        print(f"\n   PERFORMANCE:")
        print(f"   Monthly Return: {best_target['monthly_return']:.2%}")
        print(f"   Sharpe Ratio: {best_target['sharpe_ratio']:.2f}")
        print(f"   Win Rate: {best_target['win_rate']:.1%}")
        print(f"   Num Trades: {best_target['num_trades']:.0f}")
        print(f"   Max Drawdown: {best_target['max_drawdown']:.2%}")
    else:
        print(f"   ❌ No configurations achieved 2%+ monthly")
        print(f"\n   Best monthly return: {valid_configs['monthly_return'].max():.2%}")

        best_overall = valid_configs.nlargest(1, 'monthly_return').iloc[0]
        print(f"\n   BEST AVAILABLE CONFIGURATION:")
        print(f"   Meta Threshold: {best_overall['meta_threshold']:.4f}")
        print(f"   Profit Target: {best_overall['profit_target']:.2%}")
        print(f"   Stop Loss: {best_overall['stop_loss']:.2%}")
        print(f"   Monthly Return: {best_overall['monthly_return']:.2%}")

    print("\n" + "="*80)

    return combined_df


if __name__ == "__main__":
    try:
        results = combine_results()
        print("\n✓ Analysis complete!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
