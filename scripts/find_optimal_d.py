#!/usr/bin/env python3
"""
Find Optimal Fractional Differencing Parameter (d) - RiskLabAI Approach

The goal is to find the MINIMUM d that achieves "good enough" stationarity.
This preserves maximum memory while making the series stationary enough for ML.

Key principles:
1. Lower d = more memory preserved = better for prediction
2. Higher d = more stationary = better for ML models
3. We want the sweet spot: minimum d for acceptable stationarity

Practical approach (López de Prado):
- Use 10-15% significance level (less strict than academic 5%)
- Accept p-value < 0.10 as "good enough" for trading
- Typical financial series: d = 0.3-0.6
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.stattools import adfuller

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.tick_storage import TickStorage
from data.tick_to_bars import generate_bars_from_ticks
from config.tick_config import TICK_DB_PATH, INITIAL_IMBALANCE_THRESHOLD


def test_adf(series):
    """
    Augmented Dickey-Fuller test for stationarity.

    Returns:
        p_value: Lower is better (< 0.10 = stationary enough)
    """
    try:
        result = adfuller(series.dropna(), autolag='AIC')
        return result[1]  # p-value
    except:
        return 1.0


def frac_diff_ffd(series, d, threshold=1e-5):
    """
    Fast Fractional Differencing (FFD) - López de Prado method.

    More efficient than standard fractional differencing.
    Stops when weights become negligible (< threshold).
    """
    # Calculate weights
    weights = [1.0]
    k = 1
    while True:
        weight = -weights[-1] * (d - k + 1) / k
        if abs(weight) < threshold:
            break
        weights.append(weight)
        k += 1

    weights = np.array(weights[::-1])
    width = len(weights) - 1

    # Apply weights
    diff = pd.Series(index=series.index, dtype=float)
    for i in range(width, len(series)):
        diff.iloc[i] = np.dot(weights, series.iloc[i-width:i+1])

    return diff[diff.notna()]


def find_min_d_for_stationarity(series, target_pvalue=0.10):
    """
    Find minimum d where p-value < target.

    This is the RiskLabAI approach: minimum differencing for acceptable stationarity.

    Strategy:
    1. Test d from 0.0 to 1.0 in steps of 0.05
    2. Return first d where p-value < target_pvalue
    3. If none found, use conservative default

    Args:
        series: Price series
        target_pvalue: Accept stationarity if p < this (default 0.10 = 10%)

    Returns:
        d: Optimal fractional differencing parameter
    """
    print(f"\n{'='*80}")
    print(f"FINDING MINIMUM d FOR STATIONARITY")
    print(f"{'='*80}")
    print(f"Target: p-value < {target_pvalue:.2f} ({target_pvalue*100:.0f}% significance)\n")

    # Test original series
    p0 = test_adf(series)
    print(f"d=0.00 (original series): p-value = {p0:.4f}")

    if p0 < target_pvalue:
        print(f"✓ Already stationary! Using d=0.0 (preserves all memory)")
        return 0.0

    # Test d values from 0.05 to 1.0
    d_values = np.arange(0.05, 1.05, 0.05)
    results = []

    print(f"\nTesting d values...\n")
    for d in d_values:
        try:
            diff_series = frac_diff_ffd(series, d)
            p_value = test_adf(diff_series)
            results.append((d, p_value))

            # Status indicator
            if p_value < target_pvalue:
                status = "✓ ACCEPT"
            elif p_value < target_pvalue * 1.5:
                status = "~ close"
            else:
                status = "✗ reject"

            print(f"d={d:.2f}: p-value={p_value:.4f}  {status}")

            # Return first acceptable d (minimum that works)
            if p_value < target_pvalue:
                print(f"\n{'='*80}")
                print(f"✓ FOUND MINIMUM d = {d:.2f}")
                print(f"{'='*80}")
                print(f"P-value: {p_value:.4f} (< {target_pvalue:.2f} threshold)")
                print(f"\nThis balances:")
                print(f"  • Stationarity: Good enough for ML")
                print(f"  • Memory: Preserves {(1-d)*100:.0f}% of original correlation structure")
                return d

        except Exception as e:
            print(f"d={d:.2f}: ERROR - {e}")
            continue

    # If we get here, no d achieved target
    # Find best (lowest p-value) result
    if results:
        results.sort(key=lambda x: x[1])  # Sort by p-value
        best_d, best_p = results[0]

        print(f"\n{'='*80}")
        print(f"⚠️  NO d ACHIEVED TARGET (p < {target_pvalue:.2f})")
        print(f"{'='*80}")
        print(f"Best result: d={best_d:.2f}, p-value={best_p:.4f}")

        # If best is close, use it
        if best_p < target_pvalue * 2:
            print(f"\nUsing d={best_d:.2f} (closest to target)")
            return best_d

    # Conservative fallback
    print(f"\n⚠️  Using conservative default: d=0.40")
    print(f"   (Typical for financial series, preserves 60% memory)")
    return 0.40


def main():
    print("=" * 80)
    print("OPTIMAL FRACTIONAL DIFFERENCING FOR RISKLABAI")
    print("=" * 80)
    print("\nGoal: Find MINIMUM d for acceptable stationarity")
    print("Why: Lower d = more memory = better predictions")

    # Load data
    print("\n1. Loading tick imbalance bars...")
    storage = TickStorage(TICK_DB_PATH)
    ticks = storage.load_ticks('SPY')
    storage.close()

    bars = generate_bars_from_ticks(ticks, threshold=INITIAL_IMBALANCE_THRESHOLD)
    df = pd.DataFrame(bars)

    # Set datetime index
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    elif 'bar_end' in df.columns:
        df['bar_end'] = pd.to_datetime(df['bar_end'])
        df = df.set_index('bar_end')

    print(f"   Bars: {len(df):,}")
    print(f"   Period: {df.index.min().date()} to {df.index.max().date()}")

    # Use log prices
    print("\n2. Preparing log-price series...")
    log_prices = np.log(df['close'])
    print(f"   Series length: {len(log_prices)}")

    # Find optimal d
    optimal_d = find_min_d_for_stationarity(log_prices, target_pvalue=0.10)

    # Summary
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print(f"\nOptimal d = {optimal_d:.2f}")
    print(f"\nUpdate config/tick_config.py:")
    print(f"  OPTIMAL_FRACTIONAL_D = {optimal_d:.2f}")
    print("\n" + "=" * 80)

    return optimal_d


if __name__ == "__main__":
    try:
        d = main()
        print(f"\n✓ Complete: d = {d:.2f}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
