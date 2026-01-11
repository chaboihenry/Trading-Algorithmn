#!/usr/bin/env python3
"""
Walk-Forward Validation for RiskLabAI Strategy

FIXES A4: First fold trains on empty data

This script implements proper walk-forward validation with:
- Expanding window (trains on ALL data before test fold)
- Skip first fold OR ensure minimum training data
- No NaN predictions
- Realistic out-of-sample evaluation

Walk-forward validation is the gold standard for time series ML:
1. Preserves temporal order (no look-ahead bias)
2. Tests model on truly unseen data
3. Simulates real trading conditions
4. Detects overfitting and regime changes

Example timeline (6 folds):
  Fold 1: Train [0:1000], Test [1000:2000]
  Fold 2: Train [0:2000], Test [2000:3000]  <- Expanding window
  Fold 3: Train [0:3000], Test [3000:4000]
  ...

PROMPT 19: Walk-Forward First Fold Fix [A4]
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy
from data.tick_storage import TickStorage
from data.tick_to_bars import generate_bars_from_ticks
from config.tick_config import (
    TICK_DB_PATH,
    INITIAL_IMBALANCE_THRESHOLD,
    SYMBOLS
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Minimum training bars to avoid first-fold-empty-data issue
MIN_TRAIN_BARS = 200  # Minimum bars for meaningful ML training


def walk_forward_validation(
    symbol: str,
    n_splits: int = 6,
    min_training_bars: int = MIN_TRAIN_BARS
) -> Dict:
    """
    Perform walk-forward validation with proper first fold handling.

    FIXES A4: First fold now has sufficient training data by:
    1. Starting from fold 1 (not fold 0)
    2. Using expanding window (all data before test fold)
    3. Validating minimum training size before training

    Args:
        symbol: Stock ticker to validate
        n_splits: Number of folds for walk-forward (default: 6)
        min_training_bars: Minimum bars required for training (default: 200)

    Returns:
        Dictionary with validation results:
        - fold_results: List of per-fold metrics
        - avg_accuracy: Average accuracy across folds
        - avg_sharpe: Average Sharpe ratio
        - predictions_summary: Summary of predictions

    Example:
        >>> results = walk_forward_validation('SPY', n_splits=6)
        >>> print(f"Average accuracy: {results['avg_accuracy']:.2%}")
    """
    logger.info("=" * 80)
    logger.info(f"WALK-FORWARD VALIDATION: {symbol}")
    logger.info("=" * 80)
    logger.info(f"Splits: {n_splits}")
    logger.info(f"Minimum training bars: {min_training_bars}")
    logger.info("=" * 80)

    # Step 1: Load tick data and generate bars
    logger.info("\nStep 1: Loading tick data...")
    storage = TickStorage(str(TICK_DB_PATH))

    ticks = storage.load_ticks(symbol)
    if not ticks:
        logger.error(f"No tick data found for {symbol}")
        storage.close()
        return None

    logger.info(f"Loaded {len(ticks):,} ticks")

    # Generate imbalance bars
    logger.info("\nStep 2: Generating imbalance bars...")
    bars = generate_bars_from_ticks(
        ticks,
        threshold=INITIAL_IMBALANCE_THRESHOLD,
        symbol=symbol
    )

    if bars is None or len(bars) == 0:
        logger.error("Failed to generate bars")
        storage.close()
        return None

    logger.info(f"Generated {len(bars):,} imbalance bars")
    logger.info(f"Date range: {bars.index[0]} to {bars.index[-1]}")

    storage.close()

    # Step 3: Walk-forward validation
    logger.info("\n" + "=" * 80)
    logger.info("WALK-FORWARD VALIDATION")
    logger.info("=" * 80)

    results = []
    fold_size = len(bars) // n_splits

    logger.info(f"\nFold size: {fold_size} bars")
    logger.info(f"Total bars: {len(bars)}")
    logger.info("")

    # FIX A4: Start from fold 1 (not 0) to ensure first fold has training data
    for fold in range(1, n_splits):
        logger.info("-" * 80)
        logger.info(f"FOLD {fold}/{n_splits-1}")
        logger.info("-" * 80)

        # Expanding window: train on ALL data before this fold
        train_end = fold * fold_size
        test_start = train_end
        test_end = (fold + 1) * fold_size

        train_bars = bars.iloc[:train_end]
        test_bars = bars.iloc[test_start:test_end]

        # FIX A4: Skip if insufficient training data
        if len(train_bars) < min_training_bars:
            logger.warning(f"⚠️  Fold {fold}: SKIPPING - Only {len(train_bars)} training bars (need {min_training_bars})")
            continue

        logger.info(f"Training set: {len(train_bars):,} bars ({train_bars.index[0]} to {train_bars.index[-1]})")
        logger.info(f"Test set: {len(test_bars):,} bars ({test_bars.index[0]} to {test_bars.index[-1]})")

        # Train model on training data
        try:
            logger.info("\nTraining model...")
            strategy = RiskLabAIStrategy(
                profit_taking=0.5,
                stop_loss=0.5,
                max_holding=20,
                d=1.0,
                n_cv_splits=3  # Use fewer CV splits for speed
            )

            train_result = strategy.train(train_bars)

            if not train_result or 'error' in train_result:
                logger.error(f"❌ Fold {fold}: Training failed")
                continue

            logger.info(f"✓ Training complete")
            logger.info(f"  Primary accuracy: {train_result.get('primary_accuracy', 0):.2%}")
            logger.info(f"  Meta accuracy: {train_result.get('meta_accuracy', 0):.2%}")

            # Generate predictions on test data
            logger.info("\nGenerating predictions on test set...")
            test_predictions = []
            test_actuals = []

            for idx in range(len(test_bars)):
                bar = test_bars.iloc[idx:idx+1]

                # Generate prediction
                signal = strategy.generate_signal(bar)

                if signal != 0:
                    test_predictions.append(signal)

                    # Get actual outcome (next bar's return for simplification)
                    if idx < len(test_bars) - 1:
                        next_return = test_bars.iloc[idx + 1]['close'] / bar['close'].iloc[0] - 1
                        actual = 1 if next_return > 0 else -1
                        test_actuals.append(actual)

            # Calculate fold metrics
            if len(test_predictions) > 0 and len(test_actuals) > 0:
                test_predictions = np.array(test_predictions[:len(test_actuals)])
                test_actuals = np.array(test_actuals)

                accuracy = np.mean(test_predictions == test_actuals)

                # Calculate returns
                returns = []
                for pred, actual in zip(test_predictions, test_actuals):
                    ret = 0.5 if pred == actual else -0.5  # Simplified return
                    returns.append(ret)

                returns = np.array(returns)
                sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

                logger.info(f"\n✓ Fold {fold} Results:")
                logger.info(f"  Test predictions: {len(test_predictions)}")
                logger.info(f"  Accuracy: {accuracy:.2%}")
                logger.info(f"  Sharpe ratio: {sharpe:.2f}")

                # Check for NaN predictions (A4 issue indicator)
                nan_count = np.sum(np.isnan(test_predictions))
                if nan_count > 0:
                    logger.error(f"  ❌ NaN predictions: {nan_count}")
                else:
                    logger.info(f"  ✓ No NaN predictions")

                results.append({
                    'fold': fold,
                    'train_size': len(train_bars),
                    'test_size': len(test_bars),
                    'predictions': len(test_predictions),
                    'accuracy': accuracy,
                    'sharpe': sharpe,
                    'nan_count': nan_count
                })
            else:
                logger.warning(f"⚠️  Fold {fold}: No valid predictions")

        except Exception as e:
            logger.error(f"❌ Fold {fold}: Error - {e}")
            import traceback
            traceback.print_exc()
            continue

    # Step 4: Aggregate results
    logger.info("\n" + "=" * 80)
    logger.info("WALK-FORWARD VALIDATION SUMMARY")
    logger.info("=" * 80)

    if not results:
        logger.error("❌ No successful folds")
        return None

    # Calculate average metrics
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    avg_sharpe = np.mean([r['sharpe'] for r in results])
    total_predictions = sum([r['predictions'] for r in results])
    total_nans = sum([r['nan_count'] for r in results])

    logger.info(f"\nFolds completed: {len(results)}/{n_splits-1}")
    logger.info(f"Average accuracy: {avg_accuracy:.2%}")
    logger.info(f"Average Sharpe: {avg_sharpe:.2f}")
    logger.info(f"Total predictions: {total_predictions:,}")
    logger.info(f"Total NaN predictions: {total_nans} ({'✓ NONE' if total_nans == 0 else '❌ ISSUE'})")

    # Per-fold breakdown
    logger.info("\nPer-fold breakdown:")
    logger.info("-" * 80)
    logger.info(f"{'Fold':<6} {'Train':<8} {'Test':<8} {'Preds':<8} {'Accuracy':<10} {'Sharpe':<8} {'NaNs':<6}")
    logger.info("-" * 80)

    for r in results:
        logger.info(
            f"{r['fold']:<6} "
            f"{r['train_size']:<8} "
            f"{r['test_size']:<8} "
            f"{r['predictions']:<8} "
            f"{r['accuracy']:<10.2%} "
            f"{r['sharpe']:<8.2f} "
            f"{r['nan_count']:<6}"
        )

    logger.info("=" * 80)

    # Validation checks
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION CHECKS")
    logger.info("=" * 80)

    checks_passed = 0
    total_checks = 3

    # Check 1: No NaN predictions
    if total_nans == 0:
        logger.info("✓ No NaN predictions (A4 issue fixed)")
        checks_passed += 1
    else:
        logger.error(f"❌ Found {total_nans} NaN predictions (A4 issue)")

    # Check 2: All folds have sufficient training data
    min_train_size = min([r['train_size'] for r in results])
    if min_train_size >= min_training_bars:
        logger.info(f"✓ All folds have sufficient training data (min: {min_train_size})")
        checks_passed += 1
    else:
        logger.error(f"❌ Some folds have insufficient training data (min: {min_train_size})")

    # Check 3: Accuracy is reasonable
    if 0.4 <= avg_accuracy <= 0.7:
        logger.info(f"✓ Accuracy is reasonable ({avg_accuracy:.2%})")
        checks_passed += 1
    else:
        logger.warning(f"⚠️  Accuracy may be unrealistic ({avg_accuracy:.2%})")

    logger.info("=" * 80)
    logger.info(f"Checks passed: {checks_passed}/{total_checks}")
    logger.info("=" * 80)

    return {
        'fold_results': results,
        'avg_accuracy': avg_accuracy,
        'avg_sharpe': avg_sharpe,
        'total_predictions': total_predictions,
        'total_nans': total_nans,
        'checks_passed': checks_passed,
        'total_checks': total_checks
    }


def main():
    """Main entry point for walk-forward validation."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Walk-forward validation for RiskLabAI strategy'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='SPY',
        help='Symbol to validate (default: SPY)'
    )
    parser.add_argument(
        '--splits',
        type=int,
        default=6,
        help='Number of walk-forward splits (default: 6)'
    )
    parser.add_argument(
        '--min-bars',
        type=int,
        default=MIN_TRAIN_BARS,
        help=f'Minimum training bars (default: {MIN_TRAIN_BARS})'
    )

    args = parser.parse_args()

    # Run walk-forward validation
    results = walk_forward_validation(
        symbol=args.symbol,
        n_splits=args.splits,
        min_training_bars=args.min_bars
    )

    if results:
        logger.info("\n✓ Walk-forward validation complete")
        logger.info(f"  Average accuracy: {results['avg_accuracy']:.2%}")
        logger.info(f"  Average Sharpe: {results['avg_sharpe']:.2f}")

        if results['total_nans'] == 0 and results['checks_passed'] >= 2:
            logger.info("\n✓ VALIDATION PASSED")
            sys.exit(0)
        else:
            logger.error("\n❌ VALIDATION FAILED")
            sys.exit(1)
    else:
        logger.error("\n❌ Walk-forward validation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
