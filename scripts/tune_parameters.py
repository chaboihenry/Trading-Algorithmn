#!/usr/bin/env python3
"""
Tune imbalance bar threshold and max holding period using purged k-fold CV.

Usage:
    python scripts/tune_parameters.py --symbol SPY
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Imbalance Threshold and Max Holding Period")
    parser.add_argument("--symbol", help="Single symbol to test", required=True)
    parser.add_argument("--threshold-start", type=float, default=50.0)
    parser.add_argument("--threshold-end", type=float, default=150.0)
    parser.add_argument("--threshold-step", type=float, default=25.0)
    parser.add_argument("--holding-periods", type=str, default="10,20,30", help="Comma-separated holding periods to test")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "tune_parameters.log"),
            logging.StreamHandler(),
        ],
    )

    thresholds = np.arange(args.threshold_start, args.threshold_end + args.threshold_step, args.threshold_step)
    holding_periods = [int(p) for p in args.holding_periods.split(',')]
    
    all_results = []

    for holding_period in holding_periods:
        for threshold in thresholds:
            logger.info("=" * 80)
            logger.info(f"TESTING: Threshold={threshold:.2f}, Max Holding={holding_period} for symbol {args.symbol}")
            logger.info("=" * 80)

            result = {
                "threshold": threshold,
                "holding_period": holding_period,
                "cv_score": None,
                "n_bars": 0,
                "n_samples": 0,
                "neutral_pct": None,
                "reason": "Success"
            }

            try:
                strategy = RiskLabAIStrategy(
                    profit_taking=2.5,
                    stop_loss=2.5,
                    max_holding=holding_period,
                )

                train_results = strategy.train_from_ticks(
                    symbol=args.symbol,
                    threshold=threshold,
                    min_samples=100
                )

                if train_results.get("success"):
                    result["cv_score"] = train_results.get("purged_cv_mean")
                    result["n_bars"] = train_results.get("bars_count")
                    result["n_samples"] = train_results.get("n_samples")
                    result["neutral_pct"] = train_results.get("neutral_pct")
                    
                    if result["cv_score"] is not None:
                        logger.info(f"✓ SUCCESS: Threshold={threshold:.2f}, Holding={holding_period} | CV Score={result['cv_score']:.4f}, Bars={result['n_bars']}, Neutral={result['neutral_pct']:.1%}")
                    else:
                        result["reason"] = "Training succeeded but no CV score returned."
                        logger.warning(f"✗ WARNING: Threshold={threshold:.2f}, Holding={holding_period} | {result['reason']}")

                else:
                    result["reason"] = train_results.get('reason', 'Unknown failure')
                    logger.warning(f"✗ FAILED: Threshold={threshold:.2f}, Holding={holding_period} | Reason: {result['reason']}")

            except Exception as e:
                result["reason"] = str(e)
                logger.error(f"✗ EXCEPTION: Threshold={threshold:.2f}, Holding={holding_period} | An exception occurred: {e}")
            
            all_results.append(result)

    if not all_results:
        logger.error("No runs were completed. Cannot determine optimal parameters.")
        return 1
        
    results_df = pd.DataFrame(all_results)
    
    logger.info("\n" + "=" * 80)
    logger.info("TUNING COMPLETE - FULL RESULTS")
    logger.info("=" * 80)
    print(results_df.to_string())
    
    successful_results = [r for r in all_results if r["cv_score"] is not None]

    if not successful_results:
        logger.error("\nNo successful runs with a CV score. Cannot determine optimal parameters.")
        return 1

    best_result = max(successful_results, key=lambda x: x["cv_score"])
    
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMAL PARAMETERS")
    logger.info("=" * 80)
    logger.info(f"🏆 Best Threshold: {best_result['threshold']:.2f}")
    logger.info(f"🏆 Best Holding Period: {best_result['holding_period']}")
    logger.info(f"   CV Score: {best_result['cv_score']:.4f}")
    logger.info(f"   Number of Bars: {best_result['n_bars']}")
    logger.info(f"   Neutral Label %: {best_result['neutral_pct']:.1%}")
    logger.info("=" * 80)
    logger.info(f"ACTION: Update INITIAL_IMBALANCE_THRESHOLD to {best_result['threshold']:.2f} and OPTIMAL_MAX_HOLDING_BARS to {best_result['holding_period']} in config/tick_config.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
