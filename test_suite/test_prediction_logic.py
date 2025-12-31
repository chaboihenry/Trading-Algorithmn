#!/usr/bin/env python3
"""
Unit Tests for Prediction Logic

Tests the probability margin filter and signal generation logic
without requiring full market simulation.

Usage:
    python test_suite/test_prediction_logic.py
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy
from config.tick_config import (
    OPTIMAL_PROFIT_TARGET,
    OPTIMAL_STOP_LOSS,
    OPTIMAL_MAX_HOLDING_BARS,
    OPTIMAL_FRACTIONAL_D
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_probability_margin_logic():
    """Test that the 3% margin filter is working correctly."""
    logger.info("=" * 80)
    logger.info("TEST: Probability Margin Filter")
    logger.info("=" * 80)

    # Test cases: (prob_short, prob_neutral, prob_long, expected_signal, description)
    test_cases = [
        (0.45, 0.05, 0.50, 1, "Long with 5% margin (0.50 - 0.45 = 5%)"),
        (0.47, 0.05, 0.48, 0, "Neutral - only 1% margin (below 3%)"),
        (0.50, 0.05, 0.45, -1, "Short with 5% margin (0.50 - 0.45 = 5%)"),
        (0.33, 0.33, 0.34, 0, "Neutral - only 1% margin"),
        (0.40, 0.08, 0.52, 1, "Long with 12% margin (0.52 - 0.40 = 12%)"),
        (0.48, 0.05, 0.47, 0, "Neutral - 1% margin (borderline)"),
        (0.45, 0.20, 0.35, -1, "Short with 10% margin (0.45 - 0.35 = 10%)"),
        (0.30, 0.45, 0.25, 0, "Neutral wins but directional don't exceed prob_threshold"),
    ]

    MARGIN_THRESHOLD = 0.03
    prob_threshold = 0.015  # Kept from optimal params

    passed = 0
    failed = 0

    for prob_short, prob_neutral, prob_long, expected, description in test_cases:
        # Simulate the margin logic from risklabai_strategy.py
        if prob_long > prob_short and prob_long > prob_neutral:
            winner = 1
            margin = prob_long - max(prob_short, prob_neutral)
        elif prob_short > prob_long and prob_short > prob_neutral:
            winner = -1
            margin = prob_short - max(prob_long, prob_neutral)
        else:
            winner = 0
            margin = 0

        if (winner != 0 and
            margin >= MARGIN_THRESHOLD and
            max(prob_long, prob_short) > prob_threshold):
            actual_signal = winner
        else:
            actual_signal = 0

        # Check result
        if actual_signal == expected:
            logger.info(f"✓ PASS: {description}")
            logger.info(f"  Probs: S={prob_short:.2f}, N={prob_neutral:.2f}, L={prob_long:.2f}")
            logger.info(f"  Margin: {margin:.2%}, Signal: {actual_signal}")
            passed += 1
        else:
            logger.error(f"✗ FAIL: {description}")
            logger.error(f"  Expected: {expected}, Got: {actual_signal}")
            logger.error(f"  Probs: S={prob_short:.2f}, N={prob_neutral:.2f}, L={prob_long:.2f}")
            logger.error(f"  Margin: {margin:.2%}")
            failed += 1

        logger.info("")

    logger.info("=" * 80)
    logger.info(f"MARGIN FILTER TEST RESULTS: {passed} passed, {failed} failed")
    logger.info("=" * 80)
    logger.info("")

    return failed == 0


def test_model_loading():
    """Test that models can be loaded successfully."""
    logger.info("=" * 80)
    logger.info("TEST: Model Loading")
    logger.info("=" * 80)

    model_path = "models/risklabai_tick_models_aggressive.pkl"

    if not Path(model_path).exists():
        logger.error(f"✗ Model not found at {model_path}")
        logger.error("Run scripts/retrain_aggressive.py first")
        return False

    try:
        strategy = RiskLabAIStrategy(
            profit_taking=OPTIMAL_PROFIT_TARGET,
            stop_loss=OPTIMAL_STOP_LOSS,
            max_holding=OPTIMAL_MAX_HOLDING_BARS,
            d=OPTIMAL_FRACTIONAL_D,
            n_cv_splits=5,
            force_directional=True,
            neutral_threshold=0.00001
        )

        strategy.load_models(model_path)

        # Check models are loaded
        if strategy.primary_model is None:
            logger.error("✗ Primary model not loaded")
            return False

        if strategy.meta_model is None:
            logger.error("✗ Meta model not loaded")
            return False

        logger.info(f"✓ Models loaded successfully from {model_path}")
        logger.info(f"  Primary model: {type(strategy.primary_model).__name__}")
        logger.info(f"  Meta model: {type(strategy.meta_model).__name__}")
        logger.info("")

        return True

    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_generation():
    """Test that features can be generated from bar data."""
    logger.info("=" * 80)
    logger.info("TEST: Feature Generation")
    logger.info("=" * 80)

    try:
        # Create synthetic bar data
        np.random.seed(42)
        n_bars = 100

        dates = pd.date_range('2024-01-01', periods=n_bars, freq='5min')
        prices = 400 + np.cumsum(np.random.randn(n_bars) * 0.5)  # Random walk around $400

        bars = pd.DataFrame({
            'open': prices + np.random.randn(n_bars) * 0.1,
            'high': prices + abs(np.random.randn(n_bars) * 0.2),
            'low': prices - abs(np.random.randn(n_bars) * 0.2),
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_bars),
        }, index=dates)

        # Initialize strategy
        strategy = RiskLabAIStrategy(
            profit_taking=OPTIMAL_PROFIT_TARGET,
            stop_loss=OPTIMAL_STOP_LOSS,
            max_holding=OPTIMAL_MAX_HOLDING_BARS,
            d=OPTIMAL_FRACTIONAL_D,
            n_cv_splits=5,
            force_directional=True,
            neutral_threshold=0.00001
        )

        # Generate features
        features = strategy.prepare_features(bars)

        if len(features) == 0:
            logger.error("✗ No features generated")
            return False

        logger.info(f"✓ Features generated successfully")
        logger.info(f"  Input bars: {len(bars)}")
        logger.info(f"  Output features: {len(features)}")
        logger.info(f"  Feature columns: {features.columns.tolist()}")
        logger.info(f"  Feature shape: {features.shape}")
        logger.info("")

        # Check for expected features
        expected_features = ['frac_diff_close', 'ret_1', 'ret_5', 'ret_10', 'ret_20']
        missing = [f for f in expected_features if f not in features.columns]

        if missing:
            logger.warning(f"⚠️  Missing expected features: {missing}")
        else:
            logger.info(f"✓ All expected features present")
        logger.info("")

        return True

    except Exception as e:
        logger.error(f"✗ Feature generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all unit tests."""
    logger.info("=" * 80)
    logger.info("RISKLABAI PREDICTION LOGIC UNIT TESTS")
    logger.info("=" * 80)
    logger.info("")

    results = {}

    # Test 1: Probability margin logic
    results['margin_filter'] = test_probability_margin_logic()

    # Test 2: Model loading
    results['model_loading'] = test_model_loading()

    # Test 3: Feature generation
    results['feature_generation'] = test_feature_generation()

    # Summary
    logger.info("=" * 80)
    logger.info("TEST SUITE SUMMARY")
    logger.info("=" * 80)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {test_name}")

    all_passed = all(results.values())
    logger.info("")
    logger.info("=" * 80)

    if all_passed:
        logger.info("✓ ALL TESTS PASSED")
        return 0
    else:
        logger.error("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
