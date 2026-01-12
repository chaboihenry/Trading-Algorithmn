#!/usr/bin/env python3
"""
Backfill Historical Tick Data

This script fetches historical tick data from Alpaca and stores it in the database.
It's designed to be:
- Resumable: If interrupted, it picks up where it left off
- Efficient: Skips days that are already fetched
- Informative: Shows progress and estimates time remaining
- Adaptive: Automatically removes ultra-low liquidity symbols after backfill

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

    # Disable automatic cleanup of low-liquidity symbols:
    python scripts/setup/backfill_ticks.py --no-auto-cleanup
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.tick_config import (
    TICK_DB_PATH,
    SYMBOLS,
    BACKFILL_DAYS,
    FEED_NAME
)
from data.tick_storage import TickStorage
from data.alpaca_tick_client import AlpacaTickClient
from utils.market_calendar import get_trading_days, is_trading_day, MARKET_TZ

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# NOTE: get_trading_days() is now imported from utils.market_calendar
# It uses Alpaca's Calendar API to get accurate trading days (no hardcoded holidays!)
# The old implementation only skipped weekends - the new one also skips:
# - NYSE holidays (New Year's, MLK Day, Presidents Day, Good Friday, Memorial Day,
#   Juneteenth, Independence Day, Labor Day, Thanksgiving, Christmas)
# - Early closes (Christmas Eve, day before Thanksgiving)


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


def verify_data_sufficiency(symbol: str, min_ticks: int = 1_000_000) -> bool:
    """
    Verify symbol has enough data for ML training.

    Args:
        symbol: Stock ticker to check
        min_ticks: Minimum required ticks (default: 1M for sufficient ML training)

    Returns:
        bool: True if data is sufficient, False otherwise
    """
    storage = TickStorage(str(TICK_DB_PATH))
    count = storage.get_tick_count(symbol)
    storage.close()

    if count < min_ticks:
        logger.error(f"❌ {symbol}: {count:,} ticks (need {min_ticks:,})")
        return False
    logger.info(f"✓ {symbol}: {count:,} ticks - SUFFICIENT")
    return True


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
    storage = TickStorage(str(TICK_DB_PATH))
    client = AlpacaTickClient()

    # Calculate date range in Eastern Time (market's timezone)
    end_date = datetime.now(MARKET_TZ).date()
    start_date = end_date - timedelta(days=days)

    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Data feed: {FEED_NAME}")

    # Get trading days in range (uses Alpaca Calendar API - no hardcoded holidays!)
    trading_days = get_trading_days(start_date, end_date)
    logger.info(f"Trading days to process: {len(trading_days)} (weekends and holidays excluded)")

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

            existing_start = datetime.fromisoformat(earliest_str).date()
            existing_end = datetime.fromisoformat(latest_str).date()

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
            logger.info(f"\nDay {i}/{len(trading_days)}: {trading_day}")

            # Convert date to datetime for API call
            trading_datetime = datetime(
                trading_day.year,
                trading_day.month,
                trading_day.day,
                tzinfo=MARKET_TZ
            )

            # Fetch ticks for this day
            day_ticks = client.fetch_day_ticks(
                symbol,
                trading_datetime,
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
            logger.error(f"  ❌ Error fetching {trading_day}: {e}")
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

            # Verify data sufficiency for ML training
            logger.info("\nVerifying data sufficiency...")
            verify_data_sufficiency(symbol)

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
    parser.add_argument(
        '--no-auto-cleanup',
        action='store_true',
        help='Disable automatic cleanup of ultra-low liquidity symbols (<200 bars)'
    )
    parser.add_argument(
        '--min-bars',
        type=int,
        default=200,
        help='Minimum bars required for auto-cleanup (default: 200)'
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

    # Adaptive cleanup: Auto-remove symbols with insufficient bars
    if not args.no_auto_cleanup and grand_total_ticks > 0:
        logger.info("\n" + "=" * 80)
        logger.info("ADAPTIVE CLEANUP: Checking symbol liquidity...")
        logger.info("=" * 80)
        logger.info(f"Removing symbols with <{args.min_bars} imbalance bars")
        logger.info("(Insufficient data for ML training)\n")

        try:
            # Import cleanup module
            sys.path.insert(0, str(project_root / "scripts"))
            from auto_cleanup_low_liquidity import check_symbol_liquidity, delete_symbol_data

            # Check all symbols in database
            storage = TickStorage(str(TICK_DB_PATH))
            conn = storage._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT symbol FROM backfill_status ORDER BY symbol")
            all_symbols = [row[0] for row in cursor.fetchall()]

            symbols_to_delete = []
            for symbol in all_symbols:
                result = check_symbol_liquidity(storage, symbol, args.min_bars)
                if result['should_delete']:
                    symbols_to_delete.append(result)
                    logger.warning(
                        f"  ❌ {symbol}: {result['bar_count']} bars - "
                        f"DELETING ({result['reason']})"
                    )

            # Delete insufficient symbols
            if symbols_to_delete:
                logger.info(f"\n🗑️  Deleting {len(symbols_to_delete)} ultra-low liquidity symbols...")
                deleted_count = 0
                deleted_ticks = 0

                for r in symbols_to_delete:
                    if delete_symbol_data(storage, r['symbol']):
                        deleted_count += 1
                        deleted_ticks += r['total_ticks']

                logger.info(f"✓ Deleted {deleted_count} symbols")
                logger.info(f"✓ Freed ~{deleted_ticks:,} tick records")

                # Vacuum database
                logger.info("\n💾 Vacuuming database to reclaim space...")
                conn.execute("VACUUM")
                logger.info("✓ Database vacuumed successfully")

                logger.info("\n" + "=" * 80)
                logger.info(f"FINAL: {len(all_symbols) - deleted_count} symbols remain in database")
                logger.info("=" * 80)
            else:
                logger.info("\n✓ All symbols have sufficient liquidity - no cleanup needed!")

            storage.close()

        except Exception as e:
            logger.error(f"\n❌ Auto-cleanup failed: {e}")
            logger.info("You can manually run: python scripts/auto_cleanup_low_liquidity.py")

    logger.info("\n  Next step: Run scripts/calibrate_threshold.py to find optimal bar threshold")


if __name__ == "__main__":
    main()
