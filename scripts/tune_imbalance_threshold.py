#!/usr/bin/env python3
"""
Tune the imbalance bar threshold using purged k-fold cross-validation.

Usage:
    python scripts/tune_imbalance_threshold.py --symbol SPY
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune imbalance bar threshold")
    parser.add_argument("--symbol", help="Single symbol to test", required=True)
    parser.add_argument("--start-threshold", type=float, default=50.0)
    parser.add_argument("--end-threshold", type=float, default=200.0)
    parser.add_argument("--step", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "tune_imbalance_threshold.log"),
            logging.StreamHandler(),
        ],
    )

    thresholds = np.arange(args.start_threshold, args.end_threshold + args.step, args.step)
    results = []

    for threshold in thresholds:
        logger.info("=" * 80)
        logger.info(f"TESTING THRESHOLD: {threshold:.2f} for symbol {args.symbol}")
        logger.info("=" * 80)

        strategy = RiskLabAIStrategy(
            profit_taking=2.5,
            stop_loss=2.5,
            max_holding=10,
        )

        try:
            train_results = strategy.train_from_ticks(
                symbol=args.symbol,
                threshold=threshold,
                min_samples=100
            )

            if train_results.get("success"):
                cv_score = train_results.get("purged_cv_mean")
                if cv_score is not None:
                    results.append({"threshold": threshold, "cv_score": cv_score})
                    logger.info(f"✓ Threshold {threshold:.2f}: CV Score = {cv_score:.4f}")
                else:
                    logger.warning(f"✗ Threshold {threshold:.2f}: Training succeeded but no CV score returned.")
            else:
                logger.warning(f"✗ Threshold {threshold:.2f}: Training failed ({train_results.get('reason')})")

        except Exception as e:
            logger.error(f"✗ Threshold {threshold:.2f}: An exception occurred: {e}")

    if not results:
        logger.error("No successful runs. Cannot determine optimal threshold.")
        return 1

    best_result = max(results, key=lambda x: x["cv_score"])
    
    logger.info("=" * 80)
    logger.info("TUNING COMPLETE")
    logger.info("=" * 80)
    for result in results:
        logger.info(f"  - Threshold: {result['threshold']:.2f}, CV Score: {result['cv_score']:.4f}")
    
    logger.info("-" * 80)
    logger.info(f"🏆 Best Threshold: {best_result['threshold']:.2f} (CV Score: {best_result['cv_score']:.4f})")
    logger.info("=" * 80)
    logger.info(f"ACTION: Update INITIAL_IMBALANCE_THRESHOLD in config/tick_config.py to {best_result['threshold']:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
