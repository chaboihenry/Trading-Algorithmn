#!/usr/bin/env python3
"""
Threshold Optimization Script

Tests multiple combinations of:
- Primary probability threshold (for directional signals)
- Meta probability threshold (for trade filtering)

To find the optimal balance between trade frequency and quality.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy
from config.tick_config import TICK_DB_PATH, INITIAL_IMBALANCE_THRESHOLD
from data.tick_storage import TickStorage
from data.tick_to_bars import generate_bars_from_ticks

logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_threshold_combination(
    strategy,
    bars_df,
    primary_threshold,
    meta_threshold,
    test_bars=500
):
    """Test a specific combination of thresholds."""
    signals = {'long': 0, 'short': 0, 'none': 0}
    bet_sizes = []

    # Temporarily modify the strategy's predict method to use custom meta threshold
    # We'll do this by monkeypatching
    original_predict = strategy.predict

    def custom_predict(bars, prob_threshold=primary_threshold):
        # Call original predict logic but intercept meta model
        if strategy.primary_model is None:
            return 0, 0.0

        features = strategy.prepare_features(bars)
        if len(features) == 0:
            return 0, 0.0

        X = features.iloc[[-1]]

        # Primary model: probability-based direction
        probs = strategy.primary_model.predict_proba(X)[0]

        if probs[2] > prob_threshold:
            direction = 1
        elif probs[0] > prob_threshold:
            direction = -1
        else:
            direction = 0

        # Meta model with CUSTOM threshold
        if strategy.meta_model is not None:
            trade_prob = strategy.meta_model.predict_proba(X)[0, 1]

            if trade_prob < meta_threshold:  # CUSTOM meta threshold
                return 0, 0.0
            else:
                bet_size = min(trade_prob, 1.0)
                return int(direction), float(bet_size)
        else:
            return int(direction), 0.5

    # Test on last N bars
    for i in range(len(bars_df) - test_bars, len(bars_df)):
        historical_bars = bars_df.iloc[:i].copy()

        if len(historical_bars) < 100:
            continue

        signal, bet_size = custom_predict(historical_bars)

        if signal == 1:
            signals['long'] += 1
        elif signal == -1:
            signals['short'] += 1
        else:
            signals['none'] += 1

        bet_sizes.append(bet_size)

    total_signals = signals['long'] + signals['short']
    avg_bet_size = np.mean(bet_sizes) if bet_sizes else 0.0

    return {
        'primary_threshold': primary_threshold,
        'meta_threshold': meta_threshold,
        'total_signals': total_signals,
        'long': signals['long'],
        'short': signals['short'],
        'none': signals['none'],
        'signal_rate': total_signals / len(bet_sizes) if bet_sizes else 0.0,
        'avg_bet_size': avg_bet_size
    }


def optimize_thresholds():
    """Test multiple threshold combinations to find optimal settings."""

    print("\n" + "=" * 80)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 80)

    # Load model
    model_path = 'models/risklabai_tick_models_optimized.pkl'
    print(f"\n1. Loading model: {model_path}")
    strategy = RiskLabAIStrategy(
        profit_taking=0.5,
        stop_loss=0.5,
        max_holding=20,
        n_cv_splits=5
    )
    strategy.load_models(model_path)
    print("   ✓ Model loaded")

    # Load data
    print("\n2. Loading bar data...")
    storage = TickStorage(TICK_DB_PATH)
    ticks = storage.load_ticks('SPY')
    storage.close()

    bars_list = generate_bars_from_ticks(ticks, threshold=INITIAL_IMBALANCE_THRESHOLD)
    bars_df = pd.DataFrame(bars_list)
    bars_df['bar_end'] = pd.to_datetime(bars_df['bar_end'])
    if bars_df['bar_end'].dt.tz is not None:
        bars_df['bar_end'] = bars_df['bar_end'].dt.tz_localize(None)
    bars_df.set_index('bar_end', inplace=True)

    print(f"   ✓ Loaded {len(bars_df)} bars")

    # Define threshold ranges to test
    primary_thresholds = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.10]  # 0.5% to 10%
    meta_thresholds = [0.0001, 0.001, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50]  # 0.01% to 50%

    print(f"\n3. Testing {len(primary_thresholds)} x {len(meta_thresholds)} = {len(primary_thresholds) * len(meta_thresholds)} combinations...")
    print("   Primary thresholds:", [f"{t:.1%}" for t in primary_thresholds])
    print("   Meta thresholds:", [f"{t:.1%}" for t in meta_thresholds])

    # Test all combinations
    results = []

    for primary_t, meta_t in product(primary_thresholds, meta_thresholds):
        result = test_threshold_combination(
            strategy,
            bars_df,
            primary_t,
            meta_t,
            test_bars=100  # Reduced from 500 for faster testing
        )
        results.append(result)

        # Show progress for combinations that generate signals
        if result['total_signals'] > 0:
            print(f"   Primary={primary_t:.2%}, Meta={meta_t:.2%}: "
                  f"{result['total_signals']} signals ({result['signal_rate']:.1%}), "
                  f"Long={result['long']}, Short={result['short']}")

    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)

    # Show top combinations by signal count
    print("\n📊 TOP 10 COMBINATIONS BY SIGNAL COUNT:")
    top_by_signals = results_df.nlargest(10, 'total_signals')
    print(top_by_signals[['primary_threshold', 'meta_threshold', 'total_signals',
                          'long', 'short', 'signal_rate', 'avg_bet_size']].to_string(index=False))

    # Show combinations with good balance (neither too conservative nor too aggressive)
    print("\n⚖️  BALANCED COMBINATIONS (5-20% signal rate):")
    balanced = results_df[
        (results_df['signal_rate'] >= 0.05) &
        (results_df['signal_rate'] <= 0.20)
    ].sort_values('total_signals', ascending=False)

    if len(balanced) > 0:
        print(balanced[['primary_threshold', 'meta_threshold', 'total_signals',
                       'long', 'short', 'signal_rate', 'avg_bet_size']].to_string(index=False))
    else:
        print("   No combinations found in 5-20% signal rate range")

    # Find optimal combination
    # Prefer: moderate signal rate (10-15%), high avg bet size
    print("\n🎯 RECOMMENDED THRESHOLD:")

    # Filter to reasonable signal rates (1-30%)
    reasonable = results_df[
        (results_df['signal_rate'] >= 0.01) &
        (results_df['signal_rate'] <= 0.30)
    ].copy()

    if len(reasonable) > 0:
        # Score: balance between signal rate and bet size
        # Prefer moderate signal rate (10-15%) with high bet size
        reasonable['score'] = (
            reasonable['signal_rate'] * 0.5 +  # Signal frequency
            reasonable['avg_bet_size'] * 0.5    # Bet quality
        )

        best = reasonable.nlargest(1, 'score').iloc[0]

        print(f"   Primary Threshold: {best['primary_threshold']:.2%}")
        print(f"   Meta Threshold: {best['meta_threshold']:.2%}")
        print(f"   Signal Rate: {best['signal_rate']:.1%}")
        print(f"   Total Signals: {best['total_signals']}")
        print(f"   Long/Short: {best['long']}/{best['short']}")
        print(f"   Avg Bet Size: {best['avg_bet_size']:.3f}")

        print("\n💡 TO USE THESE THRESHOLDS:")
        print(f"   1. In risklabai_strategy.py, change default prob_threshold to {best['primary_threshold']:.4f}")
        print(f"   2. Modify meta model threshold from 0.5 to {best['meta_threshold']:.4f}")
    else:
        print("   No reasonable combinations found! Model may need retraining.")

    print("=" * 80)

    return results_df


if __name__ == "__main__":
    try:
        results = optimize_thresholds()
        print("\n✓ Threshold optimization complete")
    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
