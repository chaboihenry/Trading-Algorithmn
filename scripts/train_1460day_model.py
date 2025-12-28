#!/usr/bin/env python3
"""
Train 1460-Day (4-Year) RiskLabAI Model

Uses 4 years of tick data for maximum robustness.
This model should have the best generalization across market conditions.
"""

import sys
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy
from config.tick_config import INITIAL_IMBALANCE_THRESHOLD

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    print("=" * 80)
    print("TRAINING 1460-DAY (4-YEAR) MODEL")
    print("=" * 80)
    print("Using 6.4M ticks from 4 years of market data!")
    print("=" * 80)

    strategy = RiskLabAIStrategy(
        profit_taking=1.5,
        stop_loss=1.5,
        max_holding=15,
        n_cv_splits=5
    )

    results = strategy.train_from_ticks(
        symbol='SPY',
        threshold=INITIAL_IMBALANCE_THRESHOLD,
        min_samples=20
    )

    if results['success']:
        model_path = 'models/risklabai_tick_models_1460days.pkl'
        strategy.save_models(model_path)

        print("\n" + "=" * 80)
        print("1460-DAY MODEL COMPLETE - MAXIMUM DATA!")
        print("=" * 80)
        print(f"Primary Accuracy: {results['primary_accuracy']:.2%}")
        print(f"Meta Accuracy: {results['meta_accuracy']:.2%}")
        print(f"Samples: {results['n_samples']}")
        print(f"Saved to: {model_path}")
        print("=" * 80)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
