#!/usr/bin/env python3
"""
Adaptive Symbol Cleanup - Automatic Deletion of Ultra-Low Liquidity Symbols

This script automatically deletes symbols from the database that don't generate
sufficient imbalance bars for ML trading (<200 bars).

WHEN TO RUN:
- After completing a full backfill session
- Periodically to clean up the database
- Integrate into backfill scripts for automatic cleanup

HOW IT WORKS:
1. Scans all symbols in database
2. Generates bars for each symbol (in-memory, no storage)
3. Counts bars
4. Automatically deletes symbols with <200 bars
5. Vacuums database to reclaim space

ADAPTIVE THRESHOLD:
The MIN_BARS threshold can be adjusted based on your ML requirements.
Default: 200 bars (minimum for reliable ML training)

Usage:
    # Run with default threshold (200 bars):
    python scripts/auto_cleanup_low_liquidity.py

    # Run with custom threshold:
    python scripts/auto_cleanup_low_liquidity.py --min-bars 250

    # Dry run (see what would be deleted):
    python scripts/auto_cleanup_low_liquidity.py --dry-run
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.tick_storage import TickStorage
from data.tick_to_bars import generate_bars_from_ticks
from config.tick_config import TICK_DB_PATH, INITIAL_IMBALANCE_THRESHOLD

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ADAPTIVE THRESHOLD - Adjust this based on your ML requirements
DEFAULT_MIN_BARS = 200  # Minimum bars required for ML trading


def check_symbol_liquidity(storage: TickStorage, symbol: str, min_bars: int) -> Dict:
    """
    Check if a symbol has sufficient liquidity (generates enough bars).

    Args:
        storage: TickStorage instance
        symbol: Stock ticker
        min_bars: Minimum bars required

    Returns:
        Dictionary with symbol analysis
    """
    try:
        # Get tick count
        status = storage.get_backfill_status(symbol)
        if not status or status['total_ticks'] == 0:
            return {
                'symbol': symbol,
                'total_ticks': 0,
                'bar_count': 0,
                'should_delete': True,
                'reason': 'NO_DATA'
            }

        total_ticks = status['total_ticks']

        # Load ticks
        ticks = storage.load_ticks(symbol)
        if not ticks:
            return {
                'symbol': symbol,
                'total_ticks': total_ticks,
                'bar_count': 0,
                'should_delete': True,
                'reason': 'LOAD_ERROR'
            }

        # Generate bars (in-memory only)
        bars = generate_bars_from_ticks(
            ticks,
            threshold=INITIAL_IMBALANCE_THRESHOLD
        )

        if not bars:
            return {
                'symbol': symbol,
                'total_ticks': total_ticks,
                'bar_count': 0,
                'should_delete': True,
                'reason': 'BAR_GEN_ERROR'
            }

        bar_count = len(bars)
        should_delete = bar_count < min_bars

        return {
            'symbol': symbol,
            'total_ticks': total_ticks,
            'bar_count': bar_count,
            'should_delete': should_delete,
            'reason': f'BELOW_THRESHOLD (< {min_bars})' if should_delete else 'OK'
        }

    except Exception as e:
        logger.error(f"Error checking {symbol}: {e}")
        return {
            'symbol': symbol,
            'total_ticks': 0,
            'bar_count': 0,
            'should_delete': True,
            'reason': f'EXCEPTION: {str(e)}'
        }


def delete_symbol_data(storage: TickStorage, symbol: str) -> bool:
    """
    Delete all data for a symbol from the database.

    Args:
        storage: TickStorage instance
        symbol: Stock ticker to delete

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"🗑️  Deleting {symbol} from database...")

    conn = None
    try:
        conn = storage._get_connection()
        cursor = conn.cursor()

        # Delete from ticks table
        cursor.execute("DELETE FROM ticks WHERE symbol = ?", (symbol,))
        ticks_deleted = cursor.rowcount

        # Delete from imbalance_bars table
        cursor.execute("DELETE FROM imbalance_bars WHERE symbol = ?", (symbol,))
        bars_deleted = cursor.rowcount

        # Delete from backfill_status table
        cursor.execute("DELETE FROM backfill_status WHERE symbol = ?", (symbol,))

        conn.commit()

        logger.info(f"✓ Deleted {symbol}: {ticks_deleted:,} ticks, {bars_deleted:,} bars")
        return True

    except Exception as e:
        logger.error(f"❌ Error deleting {symbol}: {e}")
        if conn:
            conn.rollback()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Adaptive cleanup: Auto-delete symbols with insufficient bars'
    )
    parser.add_argument(
        '--min-bars',
        type=int,
        default=DEFAULT_MIN_BARS,
        help=f'Minimum bars required (default: {DEFAULT_MIN_BARS})'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("ADAPTIVE SYMBOL CLEANUP - AUTO-DELETE LOW LIQUIDITY")
    logger.info("=" * 80)
    logger.info(f"Mode: {'DRY RUN' if args.dry_run else 'DELETE MODE'}")
    logger.info(f"Database: {TICK_DB_PATH}")
    logger.info(f"Minimum bars threshold: {args.min_bars}")
    logger.info("=" * 80)

    # Initialize storage
    storage = TickStorage(str(TICK_DB_PATH))

    # Get all symbols
    conn = storage._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT symbol FROM backfill_status ORDER BY symbol")
    symbols = [row[0] for row in cursor.fetchall()]

    logger.info(f"\nScanning {len(symbols)} symbols...\n")

    # Check each symbol
    symbols_to_delete = []
    symbols_to_keep = []

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] Checking {symbol}...")

        result = check_symbol_liquidity(storage, symbol, args.min_bars)

        if result['should_delete']:
            symbols_to_delete.append(result)
            logger.warning(
                f"  ❌ {symbol}: {result['bar_count']} bars - "
                f"WILL DELETE ({result['reason']})"
            )
        else:
            symbols_to_keep.append(result)
            logger.info(
                f"  ✓ {symbol}: {result['bar_count']} bars - OK"
            )

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SCAN SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total symbols: {len(symbols)}")
    logger.info(f"  ✓ Keep: {len(symbols_to_keep)} symbols (≥{args.min_bars} bars)")
    logger.info(f"  ❌ Delete: {len(symbols_to_delete)} symbols (<{args.min_bars} bars)")

    # Delete if not dry run
    if symbols_to_delete:
        logger.info("\n" + "-" * 80)
        logger.info("SYMBOLS TO DELETE:")
        logger.info("-" * 80)
        logger.info(f"{'Symbol':<8} {'Ticks':<12} {'Bars':<8} {'Reason'}")
        logger.info("-" * 80)

        for r in sorted(symbols_to_delete, key=lambda x: x['bar_count']):
            logger.info(
                f"{r['symbol']:<8} "
                f"{r['total_ticks']:<12,} "
                f"{r['bar_count']:<8} "
                f"{r['reason']}"
            )

        if not args.dry_run:
            logger.info("\n" + "=" * 80)
            logger.info("DELETING SYMBOLS...")
            logger.info("=" * 80)

            deleted_count = 0
            deleted_ticks = 0

            for r in symbols_to_delete:
                if delete_symbol_data(storage, r['symbol']):
                    deleted_count += 1
                    deleted_ticks += r['total_ticks']

            logger.info("")
            logger.info(f"✓ Deleted {deleted_count}/{len(symbols_to_delete)} symbols")
            logger.info(f"✓ Freed ~{deleted_ticks:,} tick records")

            # Vacuum database
            logger.info("\nVacuuming database to reclaim space...")
            try:
                conn.execute("VACUUM")
                logger.info("✓ Database vacuumed successfully")
            except Exception as e:
                logger.error(f"❌ Error vacuuming: {e}")
        else:
            logger.info("\n" + "=" * 80)
            logger.info("DRY RUN - No deletions performed")
            logger.info("=" * 80)
            logger.info(f"Would delete {len(symbols_to_delete)} symbols")
            logger.info(f"Would free ~{sum(r['total_ticks'] for r in symbols_to_delete):,} tick records")
            logger.info("\nTo actually delete, run:")
            logger.info(f"  python scripts/auto_cleanup_low_liquidity.py --min-bars {args.min_bars}")
    else:
        logger.info("\n✓ All symbols have sufficient liquidity - nothing to delete!")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("FINAL STATUS")
    logger.info("=" * 80)
    if args.dry_run:
        logger.info(f"Current symbols: {len(symbols)}")
        logger.info(f"Would remain: {len(symbols_to_keep)}")
    else:
        logger.info(f"Symbols in database: {len(symbols_to_keep)}")
        logger.info(f"All symbols meet minimum threshold (≥{args.min_bars} bars)")
    logger.info("=" * 80)

    storage.close()

    return {
        'total': len(symbols),
        'kept': len(symbols_to_keep),
        'deleted': len(symbols_to_delete) if not args.dry_run else 0,
        'would_delete': len(symbols_to_delete) if args.dry_run else 0
    }


if __name__ == '__main__':
    main()
