#!/usr/bin/env python3
"""
Diagnostic Script: Why Are We Getting Zero Signals?

Traces through the entire RiskLabAI prediction pipeline to identify
where signal generation is failing.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy
from config.tick_config import TICK_DB_PATH, INITIAL_IMBALANCE_THRESHOLD
from data.tick_storage import TickStorage
from data.tick_to_bars import generate_bars_from_ticks

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def diagnose_prediction_pipeline():
    """Diagnose why we're getting zero signals."""

    print("\n" + "=" * 80)
    print("DIAGNOSTIC: ZERO SIGNALS INVESTIGATION")
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
    print("\n2. Loading recent bar data...")
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

    # Test on last 100 bars (multiple samples)
    print("\n3. Testing prediction on last 100 bars...")
    print("=" * 80)

    signals_distribution = {'long': 0, 'short': 0, 'none': 0}
    primary_outputs = []
    meta_probs = []
    bet_sizes = []

    for i in range(len(bars_df) - 100, len(bars_df)):
        historical_bars = bars_df.iloc[:i].copy()

        if len(historical_bars) < 100:
            continue

        # Call the predict method with optimized thresholds (now defaults)
        # Default thresholds are now: prob_threshold=0.015 (1.5%), meta_threshold=0.0001 (0.01%)
        signal, bet_size = strategy.predict(historical_bars)

        # Track signal distribution
        if signal == 1:
            signals_distribution['long'] += 1
        elif signal == -1:
            signals_distribution['short'] += 1
        else:
            signals_distribution['none'] += 1

        bet_sizes.append(bet_size)

        # Now let's dig deeper - call prepare_features and get raw model outputs
        features = strategy.prepare_features(historical_bars)

        if len(features) > 0:
            X = features.iloc[[-1]]

            # Get PRIMARY model raw prediction
            if strategy.primary_model is not None:
                primary_pred = strategy.primary_model.predict(X)[0]
                primary_outputs.append(primary_pred)

            # Get META model probability
            if strategy.meta_model is not None:
                meta_prob = strategy.meta_model.predict_proba(X)[0, 1]
                meta_probs.append(meta_prob)

    print("\n" + "=" * 80)
    print("DIAGNOSTIC RESULTS")
    print("=" * 80)

    print(f"\n📊 SIGNAL DISTRIBUTION (from predict() method):")
    print(f"   Long (1):  {signals_distribution['long']}")
    print(f"   Short (-1): {signals_distribution['short']}")
    print(f"   None (0):  {signals_distribution['none']}")
    print(f"   Total tested: {sum(signals_distribution.values())}")

    if primary_outputs:
        print(f"\n🔍 PRIMARY MODEL RAW OUTPUTS:")
        primary_arr = np.array(primary_outputs)
        unique, counts = np.unique(primary_arr, return_counts=True)
        print(f"   Unique values: {dict(zip(unique, counts))}")
        print(f"   Mean: {primary_arr.mean():.4f}")
        print(f"   Std: {primary_arr.std():.4f}")
        print(f"   Min: {primary_arr.min()}")
        print(f"   Max: {primary_arr.max()}")

    if meta_probs:
        print(f"\n📈 META MODEL PROBABILITIES:")
        meta_arr = np.array(meta_probs)
        print(f"   Mean: {meta_arr.mean():.4f}")
        print(f"   Std: {meta_arr.std():.4f}")
        print(f"   Min: {meta_arr.min():.4f}")
        print(f"   Max: {meta_arr.max():.4f}")
        print(f"   Above 0.5: {(meta_arr > 0.5).sum()} / {len(meta_arr)}")

    if bet_sizes:
        print(f"\n💰 BET SIZES (from predict()):")
        bet_arr = np.array(bet_sizes)
        print(f"   Mean: {bet_arr.mean():.4f}")
        print(f"   Non-zero: {(bet_arr > 0).sum()} / {len(bet_arr)}")
        print(f"   Above 0.5: {(bet_arr >= 0.5).sum()} / {len(bet_arr)}")

    # Deep dive into predict() logic
    print("\n" + "=" * 80)
    print("DETAILED PREDICTION ANALYSIS (Single Example)")
    print("=" * 80)

    # Get one example
    example_bars = bars_df.iloc[:len(bars_df)-50].copy()
    print(f"\nTesting with {len(example_bars)} bars...")

    # Step 1: Prepare features
    print("\n1. FEATURE PREPARATION:")
    features = strategy.prepare_features(example_bars)
    print(f"   Features shape: {features.shape if len(features) > 0 else 'EMPTY'}")

    if len(features) == 0:
        print("   ❌ ISSUE FOUND: prepare_features() returning empty DataFrame!")
        print("   This would cause predict() to return (0, 0.0)")
        return

    print(f"   ✓ Features prepared: {len(features)} rows, {len(features.columns)} columns")
    print(f"   Feature columns: {list(features.columns)[:10]}...")

    # Step 2: Get latest feature vector
    X = features.iloc[[-1]]
    print(f"\n2. LATEST FEATURE VECTOR:")
    print(f"   Shape: {X.shape}")
    print(f"   Non-null values: {X.notna().sum().sum()} / {X.size}")
    print(f"   Has NaN: {X.isna().any().any()}")
    print(f"   Has Inf: {np.isinf(X.values).any()}")

    # Step 3: Primary model prediction
    print(f"\n3. PRIMARY MODEL PREDICTION:")
    if strategy.primary_model is None:
        print("   ❌ ISSUE FOUND: primary_model is None!")
        return

    primary_pred = strategy.primary_model.predict(X)[0]
    print(f"   Raw prediction: {primary_pred}")
    print(f"   Type: {type(primary_pred)}")

    # Get probabilities if available
    if hasattr(strategy.primary_model, 'predict_proba'):
        primary_proba = strategy.primary_model.predict_proba(X)[0]
        print(f"   Probabilities: {primary_proba}")
        print(f"   Classes: {strategy.primary_model.classes_ if hasattr(strategy.primary_model, 'classes_') else 'N/A'}")

    # Step 4: Direction mapping
    print(f"\n4. DIRECTION MAPPING:")
    direction = primary_pred
    print(f"   Primary pred: {primary_pred} → Direction: {direction}")

    if direction == 0:
        print("   ⚠️  PRIMARY MODEL OUTPUTTING 0 (NO TRADE)")
        print("   This is why we get zero signals!")

    # Step 5: Meta model
    print(f"\n5. META MODEL EVALUATION:")
    if strategy.meta_model is None:
        print("   ❌ ISSUE FOUND: meta_model is None!")
        return

    meta_prob = strategy.meta_model.predict_proba(X)[0, 1]
    print(f"   Meta probability: {meta_prob:.4f}")
    print(f"   Threshold: 0.50")
    print(f"   Passes threshold: {meta_prob >= 0.5}")

    # Step 6: Final signal
    print(f"\n6. FINAL SIGNAL CALCULATION:")
    print(f"   Using optimized defaults: prob_threshold=0.015 (1.5%), meta_threshold=0.0001 (0.01%)...")
    signal, bet_size = strategy.predict(example_bars)
    print(f"   Signal: {signal}")
    print(f"   Bet size: {bet_size:.4f}")

    print("\n" + "=" * 80)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 80)

    if len(features) == 0:
        print("\n❌ ROOT CAUSE: prepare_features() returns empty DataFrame")
        print("   ACTION: Check feature engineering logic")
    elif direction == 0:
        print("\n❌ ROOT CAUSE: Primary model ALWAYS predicts 0 (no trade)")
        print("   Possible reasons:")
        print("   1. Model trained on different feature distribution")
        print("   2. Current data doesn't match training patterns")
        print("   3. Model is fundamentally too conservative")
        print("   4. Training data mismatch")
        print("\n   ACTIONS TO TRY:")
        print("   • Check training data vs prediction data distribution")
        print("   • Retrain with more balanced samples")
        print("   • Lower model decision threshold")
        print("   • Use probability-based signals instead of hard predictions")
    elif meta_prob < 0.5:
        print("\n❌ ROOT CAUSE: Meta model blocking with low probability")
        print("   ACTION: Lower meta threshold below 0.5")
    else:
        print("\n✓ Pipeline looks OK - investigate bet_size threshold")

    print("=" * 80)

    # Additional diagnostics
    print("\n" + "=" * 80)
    print("MODEL STATISTICS")
    print("=" * 80)

    # Check if model has feature importances
    if hasattr(strategy.primary_model, 'feature_importances_'):
        print("\nPrimary Model Feature Importances (Top 10):")
        importances = strategy.primary_model.feature_importances_
        if len(features.columns) == len(importances):
            feature_importance = pd.DataFrame({
                'feature': features.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            print(feature_importance.head(10).to_string(index=False))

    # Check model type
    print(f"\nPrimary Model Type: {type(strategy.primary_model).__name__}")
    print(f"Meta Model Type: {type(strategy.meta_model).__name__}")

    # Check if models have decision thresholds we can adjust
    if hasattr(strategy.primary_model, 'get_params'):
        print(f"\nPrimary Model Params: {strategy.primary_model.get_params()}")

    return {
        'signals': signals_distribution,
        'primary_outputs': primary_outputs,
        'meta_probs': meta_probs
    }


if __name__ == "__main__":
    try:
        results = diagnose_prediction_pipeline()
        print("\n✓ Diagnostic complete")
    except Exception as e:
        print(f"\n❌ Error during diagnosis: {e}")
        import traceback
        traceback.print_exc()
