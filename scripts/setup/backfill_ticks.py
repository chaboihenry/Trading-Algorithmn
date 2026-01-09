#!/usr/bin/env python3
"""
Backfill Historical Tick Data

This script fetches historical tick data from Alpaca and stores it in the database.
It's designed to be:
- Resumable: If interrupted, it picks up where it left off
- Efficient: Skips days that are already fetched
- Informative: Shows progress and estimates time remaining

The script works with both IEX (free) and SIP (paid) data - just set USE_SIP
in config/tick_config.py.

OOP Concepts:
- Uses composition: Combines TickStorage and AlpacaTickClient classes
- Procedural script structure with functions for each step
- Error handling to make script robust

Usage:
    # Backfill all configured symbols:
    python scripts/setup/backfill_ticks.py

    # Backfill specific symbol:
    python scripts/setup/backfill_ticks.py --symbol SPY

    # Backfill last 30 days:
    python scripts/setup/backfill_ticks.py --days 30

    # Force re-fetch (ignore existing data):
    python scripts/setup/backfill_ticks.py --force
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.tick_config import (
    TICK_DB_PATH,
    SYMBOLS,
    BACKFILL_DAYS,
    FEED_NAME
)
from data.tick_storage import TickStorage
from data.alpaca_tick_client import AlpacaTickClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_trading_days(start_date: datetime, end_date: datetime) -> List[datetime]:
    """
    Get list of trading days (weekdays) between two dates.

    This filters out weekends. In a production system, you'd also filter
    out market holidays (New Year's, Christmas, etc.), but for now we'll
    just skip weekends.

    Args:
        start_date: First date
        end_date: Last date (inclusive)

    Returns:
        List of datetime objects representing trading days

    Example:
        >>> from datetime import datetime
        >>> days = get_trading_days(datetime(2024, 1, 1), datetime(2024, 1, 7))
        >>> print(f"{len(days)} trading days")  # Should be 5 (Mon-Fri)
    """
    trading_days = []
    current = start_date

    while current <= end_date:
        # Skip weekends (Monday=0, Sunday=6)
        if current.weekday() < 5:
            trading_days.append(current)
        current += timedelta(days=1)

    return trading_days


def estimate_time_remaining(
    elapsed_seconds: float,
    completed: int,
    total: int
) -> str:
    """
    Estimate time remaining based on current progress.

    Args:
        elapsed_seconds: Time spent so far
        completed: Number of items completed
        total: Total number of items

    Returns:
        Human-readable time estimate (e.g., "5m 30s", "1h 15m")
    """
    if completed == 0:
        return "unknown"

    avg_time_per_item = elapsed_seconds / completed
    remaining_items = total - completed
    remaining_seconds = avg_time_per_item * remaining_items

    # Convert to human-readable format
    if remaining_seconds < 60:
        return f"{int(remaining_seconds)}s"
    elif remaining_seconds < 3600:
        minutes = int(remaining_seconds / 60)
        seconds = int(remaining_seconds % 60)
        return f"{minutes}m {seconds}s"
    else:
        hours = int(remaining_seconds / 3600)
        minutes = int((remaining_seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


def backfill_symbol(
    symbol: str,
    days: int,
    force: bool = False
) -> Tuple[int, int]:
    """
    Backfill tick data for a single symbol.

    Args:
        symbol: Stock ticker
        days: Number of days to backfill
        force: If True, re-fetch even if data exists

    Returns:
        Tuple of (total_ticks_fetched, days_processed)
    """
    logger.info("=" * 80)
    logger.info(f"BACKFILLING {symbol}")
    logger.info("=" * 80)

    # Initialize components
    storage = TickStorage(TICK_DB_PATH)
    client = AlpacaTickClient()

    # Calculate date range
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)

    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Data feed: {FEED_NAME}")

    # Get trading days in range
    trading_days = get_trading_days(start_date, end_date)
    logger.info(f"Trading days to process: {len(trading_days)}")

    # Check existing data
    if not force:
        status = storage.get_backfill_status(symbol)
        if status:
            logger.info(f"Existing data found:")
            logger.info(f"  Range: {status['earliest_timestamp']} to {status['latest_timestamp']}")
            logger.info(f"  Total ticks: {status['total_ticks']:,}")
            logger.info(f"  Last updated: {status['last_updated']}")

            # Filter out days we already have
            # Strip timezone info to avoid comparison errors
            earliest_str = status['earliest_timestamp'].split('+')[0].split('T')[0] if 'T' in status['earliest_timestamp'] or '+' in status['earliest_timestamp'] else status['earliest_timestamp'].split()[0]
            latest_str = status['latest_timestamp'].split('+')[0].split('T')[0] if 'T' in status['latest_timestamp'] or '+' in status['latest_timestamp'] else status['latest_timestamp'].split()[0]

            existing_start = datetime.fromisoformat(earliest_str)
            existing_end = datetime.fromisoformat(latest_str)

            trading_days = [
                day for day in trading_days
                if day < existing_start or day > existing_end
            ]

            if trading_days:
                logger.info(f"Fetching {len(trading_days)} missing days")
            else:
                logger.info(f"✓ All days already fetched (use --force to re-fetch)")
                storage.close()
                return (0, 0)

    # Fetch data day by day
    total_ticks = 0
    days_processed = 0
    start_time = datetime.now()

    for i, trading_day in enumerate(trading_days, 1):
        try:
            logger.info(f"\nDay {i}/{len(trading_days)}: {trading_day.date()}")

            # Fetch ticks for this day
            day_ticks = client.fetch_day_ticks(
                symbol,
                trading_day,
                extended_hours=True
            )

            if day_ticks:
                # Save to database
                saved = storage.save_ticks(symbol, day_ticks)
                total_ticks += saved
                days_processed += 1

                logger.info(f"  ✓ Saved {saved:,} ticks")
            else:
                logger.warning(f"  ⚠ No ticks found (market closed or data unavailable)")

            # Show progress
            elapsed = (datetime.now() - start_time).total_seconds()
            eta = estimate_time_remaining(elapsed, i, len(trading_days))

            logger.info(f"  Progress: {i}/{len(trading_days)} days ({i/len(trading_days)*100:.1f}%)")
            logger.info(f"  Total ticks: {total_ticks:,}")
            logger.info(f"  Elapsed: {int(elapsed)}s, ETA: {eta}")

        except KeyboardInterrupt:
            logger.warning("\n\nBackfill interrupted by user")
            break
        except Exception as e:
            logger.error(f"  ❌ Error fetching {trading_day.date()}: {e}")
            continue

    # Update backfill status
    if days_processed > 0:
        date_range = storage.get_date_range(symbol)
        if date_range:
            earliest, latest = date_range
            total_stored = storage.get_tick_count(symbol)

            storage.update_backfill_status(
                symbol,
                earliest,
                latest,
                total_stored
            )

            logger.info("\n" + "=" * 80)
            logger.info(f"BACKFILL COMPLETE FOR {symbol}")
            logger.info("=" * 80)
            logger.info(f"Fetched: {total_ticks:,} new ticks")
            logger.info(f"Total stored: {total_stored:,} ticks")
            logger.info(f"Date range: {earliest} to {latest}")
            logger.info("=" * 80)

    storage.close()
    return (total_ticks, days_processed)


def main():
    """
    Main entry point for backfill script.

    Parses command-line arguments and runs backfill for each symbol.
    """
    parser = argparse.ArgumentParser(
        description='Backfill historical tick data from Alpaca'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        help=f'Symbol to backfill (default: all configured symbols: {", ".join(SYMBOLS)})'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=BACKFILL_DAYS,
        help=f'Number of days to backfill (default: {BACKFILL_DAYS})'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-fetch even if data exists'
    )

    args = parser.parse_args()

    # Determine which symbols to backfill
    if args.symbol:
        symbols_to_backfill = [args.symbol.upper()]
    else:
        symbols_to_backfill = SYMBOLS

    logger.info("=" * 80)
    logger.info("TICK DATA BACKFILL")
    logger.info("=" * 80)
    logger.info(f"Symbols: {', '.join(symbols_to_backfill)}")
    logger.info(f"Days: {args.days}")
    logger.info(f"Data feed: {FEED_NAME}")
    logger.info(f"Force re-fetch: {args.force}")
    logger.info("=" * 80)

    # Test connection first
    client = AlpacaTickClient()
    if not client.test_connection():
        logger.error("\n❌ Connection test failed!")
        logger.error("Check your Alpaca API credentials and subscription level")
        sys.exit(1)

    # Backfill each symbol
    grand_total_ticks = 0
    grand_total_days = 0

    for symbol in symbols_to_backfill:
        try:
            ticks, days = backfill_symbol(symbol, args.days, args.force)
            grand_total_ticks += ticks
            grand_total_days += days
        except Exception as e:
            logger.error(f"\n❌ Failed to backfill {symbol}: {e}")
            continue

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Symbols processed: {len(symbols_to_backfill)}")
    logger.info(f"Trading days fetched: {grand_total_days}")
    logger.info(f"Total ticks: {grand_total_ticks:,}")
    logger.info("=" * 80)
    logger.info("\n✓ Tick data is ready!")
    logger.info("  Next step: Run scripts/calibrate_threshold.py to find optimal bar threshold")


if __name__ == "__main__":
    main()
