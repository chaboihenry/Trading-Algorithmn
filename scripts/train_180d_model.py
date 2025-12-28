#!/usr/bin/env python3
"""
Train Improved RiskLabAI Model with Full Tick Dataset

This script trains a new RiskLabAI model using all available tick data.
With more data, the model should identify more profitable trading opportunities.

Usage:
    python scripts/train_improved_model.py
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
    """Train improved model with full tick dataset."""
    print("=" * 80)
    print("TRAINING IMPROVED RISKLABAI MODEL")
    print("=" * 80)
    print(f"Using all available tick data from database")
    print(f"Threshold: {INITIAL_IMBALANCE_THRESHOLD}")
    print("=" * 80)

    # Initialize strategy
    strategy = RiskLabAIStrategy(
        profit_taking=2.0,
        stop_loss=2.0,
        max_holding=10,
        n_cv_splits=5
    )

    # Train from tick data
    logger.info("Training from tick data...")
    results = strategy.train_from_ticks(
        symbol='SPY',
        threshold=INITIAL_IMBALANCE_THRESHOLD,
        min_samples=50  # Lower threshold to use more data
    )

    if results['success']:
        # Save improved model
        model_path = 'models/risklabai_tick_models_improved.pkl'
        strategy.save_models(model_path)
        logger.info(f"✓ Improved model saved to {model_path}")

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Primary Model Accuracy: {results['primary_accuracy']:.2%}")
        print(f"Meta Model Accuracy: {results['meta_accuracy']:.2%}")
        print(f"Samples Used: {results['n_samples']}")
        print(f"Model saved to: {model_path}")
        print("=" * 80)
        print("\nTo use this model, update run_live_trading.py:")
        print("  'model_path': 'models/risklabai_tick_models_improved.pkl'")
        print("=" * 80)

    else:
        logger.error("❌ Training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
