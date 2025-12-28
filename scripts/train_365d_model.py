#!/usr/bin/env python3
"""
Train Maximum RiskLabAI Model with Full Year Dataset

This script trains a model using ALL available tick data with optimized parameters
to maximize sample usage and model performance.

Strategy:
- Use ALL 1.5M ticks (full year)
- Lower min_samples threshold to use more data
- Compare to previous models
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy
from config.tick_config import INITIAL_IMBALANCE_THRESHOLD

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Train maximum model with full year dataset."""
    print("=" * 80)
    print("TRAINING MAXIMUM RISKLABAI MODEL")
    print("=" * 80)
    print("Strategy: Use ALL tick data with optimized parameters")
    print(f"Threshold: {INITIAL_IMBALANCE_THRESHOLD}")
    print("=" * 80)

    # Initialize strategy
    strategy = RiskLabAIStrategy(
        profit_taking=1.5,  # Slightly tighter for more signals
        stop_loss=1.5,
        max_holding=15,  # Longer holding period
        n_cv_splits=5
    )

    # Train from ALL tick data
    logger.info("Training from ALL tick data in database...")
    results = strategy.train_from_ticks(
        symbol='SPY',
        threshold=INITIAL_IMBALANCE_THRESHOLD,
        min_samples=20  # Much lower to use maximum data
    )

    if results['success']:
        # Save maximum model
        model_path = 'models/risklabai_tick_models_maximum.pkl'
        strategy.save_models(model_path)
        logger.info(f"✓ Maximum model saved to {model_path}")

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE - MAXIMUM MODEL")
        print("=" * 80)
        print(f"Primary Model Accuracy: {results['primary_accuracy']:.2%}")
        print(f"Meta Model Accuracy: {results['meta_accuracy']:.2%}")
        print(f"Samples Used: {results['n_samples']}")
        print(f"Model saved to: {model_path}")
        print("=" * 80)

        # Compare with previous models
        print("\nMODEL COMPARISON:")
        print("-" * 80)
        print("Model 1 (Original):   119 samples, 100.00% accuracy, 48.11% meta")
        print("Model 2 (Improved):  1006 samples,  97.61% accuracy, 54.17% meta")
        print(f"Model 3 (Maximum):  {results['n_samples']:4d} samples, {results['primary_accuracy']:6.2%} accuracy, {results['meta_accuracy']:6.2%} meta")
        print("-" * 80)

    else:
        logger.error("❌ Training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
