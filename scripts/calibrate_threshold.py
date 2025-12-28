#!/usr/bin/env python3
"""
Calibrate Tick Imbalance Bar Threshold

This script finds the optimal threshold for tick imbalance bars to achieve
a target number of bars per trading day.

Target:
- With IEX (free): ~5 bars/day
- With SIP (paid): ~50 bars/day

The script:
1. Loads historical tick data from database
2. Tests a range of thresholds
3. Generates bars for each threshold
4. Counts bars per day
5. Recommends the threshold closest to target

Usage:
    # Calibrate for SPY:
    python scripts/calibrate_threshold.py

    # Calibrate for specific symbol:
    python scripts/calibrate_threshold.py --symbol QQQ

    # Use specific date range:
    python scripts/calibrate_threshold.py --days 5
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import statistics

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.tick_config import (
    TICK_DB_PATH,
    SYMBOLS,
    TARGET_BARS_PER_DAY,
    FEED_NAME
)
from data.tick_storage import TickStorage
from data.tick_to_bars import generate_bars_from_ticks

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calibrate_threshold(
    symbol: str,
    days: int = 5
) -> Dict:
    """
    Find optimal threshold for a symbol.

    Args:
        symbol: Stock ticker to calibrate
        days: Number of days of data to use for calibration

    Returns:
        Dict with calibration results:
        {
            'optimal_threshold': 50.0,
            'bars_per_day': 48.2,
            'tested_thresholds': [...],
            'results': [...]
        }
    """
    logger.info("=" * 80)
    logger.info(f"CALIBRATING THRESHOLD FOR {symbol}")
    logger.info("=" * 80)
    logger.info(f"Data feed: {FEED_NAME}")
    logger.info(f"Target: {TARGET_BARS_PER_DAY} bars/day")
    logger.info(f"Days to analyze: {days}")

    # Load tick data
    storage = TickStorage(TICK_DB_PATH)

    # Get available date range
    date_range = storage.get_date_range(symbol)

    if not date_range:
        logger.error(f"❌ No tick data found for {symbol}")
        logger.error("Run scripts/backfill_ticks.py first to fetch historical data")
        storage.close()
        return None

    earliest, latest = date_range
    logger.info(f"Available data: {earliest} to {latest}")

    # Calculate how many days we have
    # Strip timezone info if present
    earliest_str = earliest.split('+')[0].split('T')[0] if 'T' in earliest or '+' in earliest else earliest.split()[0]
    latest_str = latest.split('+')[0].split('T')[0] if 'T' in latest or '+' in latest else latest.split()[0]

    start_date = datetime.fromisoformat(earliest_str)
    end_date = datetime.fromisoformat(latest_str)
    available_days = (end_date - start_date).days + 1

    if available_days < days:
        logger.warning(f"Only {available_days} days available, using all of them")
        days = available_days

    # Load ticks for calibration period (use most recent data)
    calibration_start = (end_date - timedelta(days=days)).isoformat()
    calibration_end = end_date.isoformat()

    logger.info(f"\nLoading ticks from {calibration_start} to {calibration_end}...")
    ticks = storage.load_ticks(symbol, start=calibration_start, end=calibration_end)

    if not ticks:
        logger.error(f"❌ No ticks found in calibration period")
        storage.close()
        return None

    logger.info(f"✓ Loaded {len(ticks):,} ticks")

    # Test a range of thresholds
    # For IEX: expect ~10% of ticks vs SIP, so test smaller thresholds
    # For SIP: test larger thresholds
    # We'll test from very small to very large and find the sweet spot

    # Start with rough estimate based on ticks per day
    ticks_per_day = len(ticks) / days
    logger.info(f"Ticks per day: {ticks_per_day:,.0f}")

    # Generate candidate thresholds to test
    # We want to bracket the optimal threshold
    # Rough estimate: threshold ≈ ticks_per_day / (target_bars * 2)
    rough_estimate = max(10, ticks_per_day / (TARGET_BARS_PER_DAY * 2))

    # Test thresholds around this estimate
    # Use logarithmic spacing for better coverage
    thresholds = [
        rough_estimate * 0.1,
        rough_estimate * 0.25,
        rough_estimate * 0.5,
        rough_estimate * 0.75,
        rough_estimate * 1.0,
        rough_estimate * 1.5,
        rough_estimate * 2.0,
        rough_estimate * 3.0,
        rough_estimate * 5.0,
        rough_estimate * 10.0
    ]

    # Round to reasonable values
    thresholds = [round(t, 1) for t in thresholds]

    logger.info(f"\nTesting {len(thresholds)} threshold values...")
    logger.info(f"Range: {min(thresholds):.1f} to {max(thresholds):.1f}")

    results = []

    for threshold in thresholds:
        # Generate bars with this threshold
        bars = generate_bars_from_ticks(ticks, threshold=threshold)

        if not bars:
            continue

        # Calculate bars per day
        bars_per_day = len(bars) / days

        # Calculate average ticks per bar
        avg_ticks_per_bar = len(ticks) / len(bars) if bars else 0

        results.append({
            'threshold': threshold,
            'total_bars': len(bars),
            'bars_per_day': bars_per_day,
            'avg_ticks_per_bar': avg_ticks_per_bar,
            'error': abs(bars_per_day - TARGET_BARS_PER_DAY)
        })

        logger.info(
            f"  Threshold {threshold:8.1f}: {len(bars):4d} bars "
            f"({bars_per_day:6.1f} bars/day, {avg_ticks_per_bar:7.1f} ticks/bar)"
        )

    storage.close()

    if not results:
        logger.error("❌ No valid results - try adjusting threshold range")
        return None

    # Find best threshold (closest to target)
    best_result = min(results, key=lambda x: x['error'])

    logger.info("\n" + "=" * 80)
    logger.info("CALIBRATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Optimal threshold: {best_result['threshold']:.1f}")
    logger.info(f"Bars per day: {best_result['bars_per_day']:.1f} (target: {TARGET_BARS_PER_DAY})")
    logger.info(f"Avg ticks per bar: {best_result['avg_ticks_per_bar']:.1f}")
    logger.info("=" * 80)

    return {
        'symbol': symbol,
        'optimal_threshold': best_result['threshold'],
        'bars_per_day': best_result['bars_per_day'],
        'avg_ticks_per_bar': best_result['avg_ticks_per_bar'],
        'tested_thresholds': thresholds,
        'all_results': results
    }


def main():
    """
    Main entry point for calibration script.
    """
    parser = argparse.ArgumentParser(
        description='Calibrate tick imbalance bar threshold'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default=SYMBOLS[0],
        help=f'Symbol to calibrate (default: {SYMBOLS[0]})'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=5,
        help='Number of days to use for calibration (default: 5)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("TICK IMBALANCE BAR THRESHOLD CALIBRATION")
    print("=" * 80)
    print(f"Symbol: {args.symbol}")
    print(f"Data feed: {FEED_NAME}")
    print(f"Target: {TARGET_BARS_PER_DAY} bars/day")
    print("=" * 80)

    # Run calibration
    result = calibrate_threshold(args.symbol, args.days)

    if result:
        print("\n" + "=" * 80)
        print("RECOMMENDATION")
        print("=" * 80)
        print(f"\nUpdate config/tick_config.py:")
        print(f"  INITIAL_IMBALANCE_THRESHOLD = {result['optimal_threshold']:.1f}")
        print(f"\nThis will produce approximately {result['bars_per_day']:.0f} bars per day")
        print("=" * 80)
    else:
        print("\n❌ Calibration failed")
        print("Make sure you have tick data in the database:")
        print("  python scripts/backfill_ticks.py --symbol", args.symbol)
        sys.exit(1)


if __name__ == "__main__":
    main()
