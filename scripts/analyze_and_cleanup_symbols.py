#!/usr/bin/env python3
"""
Analyze Symbol Liquidity and Cleanup Low-Quality Data

This script:
1. Generates imbalance bars for all symbols in database
2. Counts bars per symbol
3. Classifies symbols into liquidity tiers
4. Removes ultra-low liquidity symbols (<200 bars) to save space
5. Reports comprehensive statistics

Liquidity Tiers:
- Tier 1 (High): 400+ bars - Ready for trading
- Tier 2 (Medium): 250-399 bars - Usable
- Tier 3 (Low): 200-249 bars - Marginal
- Tier 4 (Ultra-low): <200 bars - DELETE (not suitable for ML)

Usage:
    # Dry run (don't delete anything, just report):
    python scripts/analyze_and_cleanup_symbols.py --dry-run

    # Actually delete ultra-low liquidity symbols:
    python scripts/analyze_and_cleanup_symbols.py --delete
"""

import sys
import logging
from pathlib import Path
import argparse

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

# Thresholds for liquidity tiers
MIN_BARS_HIGH_LIQUIDITY = 400      # Tier 1: Excellent
MIN_BARS_MEDIUM_LIQUIDITY = 250    # Tier 2: Good
MIN_BARS_LOW_LIQUIDITY = 200       # Tier 3: Marginal
# Below 200 = Tier 4: Ultra-low (DELETE)


def analyze_symbol_bars(storage: TickStorage, symbol: str) -> dict:
    """
    Generate bars for a symbol and analyze liquidity.

    Args:
        storage: TickStorage instance
        symbol: Stock ticker

    Returns:
        Dictionary with symbol analysis:
        {
            'symbol': str,
            'total_ticks': int,
            'bar_count': int,
            'tier': int (1-4),
            'tier_name': str,
            'ticks_per_bar': float,
            'suitable_for_trading': bool
        }
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Analyzing {symbol}")
    logger.info(f"{'='*60}")

    # Get backfill status
    status = storage.get_backfill_status(symbol)
    if not status or status['total_ticks'] == 0:
        logger.warning(f"❌ {symbol}: No tick data found")
        return {
            'symbol': symbol,
            'total_ticks': 0,
            'bar_count': 0,
            'tier': 4,
            'tier_name': 'NO_DATA',
            'ticks_per_bar': 0,
            'suitable_for_trading': False
        }

    total_ticks = status['total_ticks']
    logger.info(f"Total ticks: {total_ticks:,}")

    # Load ticks
    logger.info("Loading ticks...")
    ticks = storage.load_ticks(symbol)
    if not ticks:
        logger.warning(f"❌ {symbol}: Failed to load ticks")
        return {
            'symbol': symbol,
            'total_ticks': total_ticks,
            'bar_count': 0,
            'tier': 4,
            'tier_name': 'LOAD_ERROR',
            'ticks_per_bar': 0,
            'suitable_for_trading': False
        }

    logger.info(f"Loaded {len(ticks):,} ticks")

    # Generate imbalance bars
    logger.info("Generating imbalance bars...")
    try:
        bars = generate_bars_from_ticks(
            ticks,
            threshold=INITIAL_IMBALANCE_THRESHOLD
        )

        if bars is None or len(bars) == 0:
            logger.warning(f"❌ {symbol}: Failed to generate bars")
            return {
                'symbol': symbol,
                'total_ticks': total_ticks,
                'bar_count': 0,
                'tier': 4,
                'tier_name': 'BAR_GEN_ERROR',
                'ticks_per_bar': 0,
                'suitable_for_trading': False
            }

        bar_count = len(bars)
        ticks_per_bar = total_ticks / bar_count if bar_count > 0 else 0

        # Classify into tier
        if bar_count >= MIN_BARS_HIGH_LIQUIDITY:
            tier = 1
            tier_name = "HIGH_LIQUIDITY"
            suitable = True
        elif bar_count >= MIN_BARS_MEDIUM_LIQUIDITY:
            tier = 2
            tier_name = "MEDIUM_LIQUIDITY"
            suitable = True
        elif bar_count >= MIN_BARS_LOW_LIQUIDITY:
            tier = 3
            tier_name = "LOW_LIQUIDITY"
            suitable = True
        else:
            tier = 4
            tier_name = "ULTRA_LOW_LIQUIDITY"
            suitable = False

        logger.info(f"✓ Generated {bar_count:,} bars")
        logger.info(f"  Ticks per bar: {ticks_per_bar:.0f}")
        logger.info(f"  Tier: {tier} - {tier_name}")
        logger.info(f"  Suitable for trading: {'YES ✓' if suitable else 'NO ❌'}")

        return {
            'symbol': symbol,
            'total_ticks': total_ticks,
            'bar_count': bar_count,
            'tier': tier,
            'tier_name': tier_name,
            'ticks_per_bar': ticks_per_bar,
            'suitable_for_trading': suitable
        }

    except Exception as e:
        logger.error(f"❌ {symbol}: Error generating bars: {e}")
        import traceback
        traceback.print_exc()
        return {
            'symbol': symbol,
            'total_ticks': total_ticks,
            'bar_count': 0,
            'tier': 4,
            'tier_name': 'EXCEPTION',
            'ticks_per_bar': 0,
            'suitable_for_trading': False
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
    logger.info(f"🗑️  Deleting all data for {symbol}...")

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

        logger.info(f"✓ Deleted {ticks_deleted:,} ticks, {bars_deleted:,} bars for {symbol}")
        return True

    except Exception as e:
        logger.error(f"❌ Error deleting {symbol}: {e}")
        if conn:
            conn.rollback()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Analyze symbol liquidity and cleanup low-quality data'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Analyze only, do not delete anything (default)'
    )
    parser.add_argument(
        '--delete',
        action='store_true',
        help='Actually delete ultra-low liquidity symbols (<200 bars)'
    )

    args = parser.parse_args()

    # Default to dry-run if neither specified
    delete_mode = args.delete

    logger.info("=" * 80)
    logger.info("SYMBOL LIQUIDITY ANALYSIS AND CLEANUP")
    logger.info("=" * 80)
    logger.info(f"Mode: {'DELETE ULTRA-LOW LIQUIDITY' if delete_mode else 'DRY RUN (no deletions)'}")
    logger.info(f"Database: {TICK_DB_PATH}")
    logger.info("")
    logger.info("Liquidity Tiers:")
    logger.info(f"  Tier 1 (High): {MIN_BARS_HIGH_LIQUIDITY}+ bars - Excellent for trading")
    logger.info(f"  Tier 2 (Medium): {MIN_BARS_MEDIUM_LIQUIDITY}-{MIN_BARS_HIGH_LIQUIDITY-1} bars - Good for trading")
    logger.info(f"  Tier 3 (Low): {MIN_BARS_LOW_LIQUIDITY}-{MIN_BARS_MEDIUM_LIQUIDITY-1} bars - Marginal for trading")
    logger.info(f"  Tier 4 (Ultra-low): <{MIN_BARS_LOW_LIQUIDITY} bars - NOT suitable (will DELETE)")
    logger.info("=" * 80)

    # Initialize storage
    storage = TickStorage(str(TICK_DB_PATH))

    # Get all symbols from backfill_status
    conn = storage._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT symbol FROM backfill_status ORDER BY symbol")
    symbols = [row[0] for row in cursor.fetchall()]

    logger.info(f"\nFound {len(symbols)} symbols in database")
    logger.info("")

    # Analyze each symbol
    results = []
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"\n[{i}/{len(symbols)}] Processing {symbol}...")
        analysis = analyze_symbol_bars(storage, symbol)
        results.append(analysis)

    # Categorize results
    tier1 = [r for r in results if r['tier'] == 1]
    tier2 = [r for r in results if r['tier'] == 2]
    tier3 = [r for r in results if r['tier'] == 3]
    tier4 = [r for r in results if r['tier'] == 4]

    suitable = [r for r in results if r['suitable_for_trading']]
    unsuitable = [r for r in results if not r['suitable_for_trading']]

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"\nTotal symbols analyzed: {len(results)}")
    logger.info("")
    logger.info("BY TIER:")
    logger.info(f"  Tier 1 (High Liquidity): {len(tier1)} symbols")
    logger.info(f"  Tier 2 (Medium Liquidity): {len(tier2)} symbols")
    logger.info(f"  Tier 3 (Low Liquidity): {len(tier3)} symbols")
    logger.info(f"  Tier 4 (Ultra-low Liquidity): {len(tier4)} symbols")
    logger.info("")
    logger.info("SUITABILITY:")
    logger.info(f"  ✓ Suitable for trading (≥{MIN_BARS_LOW_LIQUIDITY} bars): {len(suitable)} symbols")
    logger.info(f"  ❌ NOT suitable (<{MIN_BARS_LOW_LIQUIDITY} bars): {len(unsuitable)} symbols")

    # Show unsuitable symbols
    if unsuitable:
        logger.info("\n" + "-" * 80)
        logger.info("ULTRA-LOW LIQUIDITY SYMBOLS (Will be deleted in --delete mode):")
        logger.info("-" * 80)
        logger.info(f"{'Symbol':<8} {'Ticks':<12} {'Bars':<8} {'Ticks/Bar':<12} {'Status'}")
        logger.info("-" * 80)

        for r in sorted(unsuitable, key=lambda x: x['bar_count'], reverse=True):
            logger.info(
                f"{r['symbol']:<8} "
                f"{r['total_ticks']:<12,} "
                f"{r['bar_count']:<8} "
                f"{r['ticks_per_bar']:<12,.0f} "
                f"{r['tier_name']}"
            )

    # Show suitable symbols by tier
    logger.info("\n" + "-" * 80)
    logger.info(f"TIER 1 - HIGH LIQUIDITY ({len(tier1)} symbols):")
    logger.info("-" * 80)
    if tier1:
        for r in sorted(tier1, key=lambda x: x['bar_count'], reverse=True):
            logger.info(f"  {r['symbol']:<8} {r['bar_count']:>4} bars  ({r['total_ticks']:>10,} ticks)")

    logger.info("\n" + "-" * 80)
    logger.info(f"TIER 2 - MEDIUM LIQUIDITY ({len(tier2)} symbols):")
    logger.info("-" * 80)
    if tier2:
        for r in sorted(tier2, key=lambda x: x['bar_count'], reverse=True):
            logger.info(f"  {r['symbol']:<8} {r['bar_count']:>4} bars  ({r['total_ticks']:>10,} ticks)")

    logger.info("\n" + "-" * 80)
    logger.info(f"TIER 3 - LOW LIQUIDITY ({len(tier3)} symbols):")
    logger.info("-" * 80)
    if tier3:
        for r in sorted(tier3, key=lambda x: x['bar_count'], reverse=True):
            logger.info(f"  {r['symbol']:<8} {r['bar_count']:>4} bars  ({r['total_ticks']:>10,} ticks)")

    # Initialize counters
    deleted_count = 0
    deleted_ticks = 0

    # Delete ultra-low liquidity symbols if in delete mode
    if delete_mode and unsuitable:
        logger.info("\n" + "=" * 80)
        logger.info("DELETING ULTRA-LOW LIQUIDITY SYMBOLS")
        logger.info("=" * 80)

        for r in unsuitable:
            if delete_symbol_data(storage, r['symbol']):
                deleted_count += 1
                deleted_ticks += r['total_ticks']

        logger.info("")
        logger.info(f"✓ Deleted {deleted_count} symbols")
        logger.info(f"✓ Freed up ~{deleted_ticks:,} tick records")

        # Vacuum database to reclaim space
        logger.info("\nVacuuming database to reclaim space...")
        try:
            conn.execute("VACUUM")
            logger.info("✓ Database vacuumed successfully")
        except Exception as e:
            logger.error(f"❌ Error vacuuming database: {e}")

    elif unsuitable:
        logger.info("\n" + "=" * 80)
        logger.info("DRY RUN - No deletions performed")
        logger.info("=" * 80)
        logger.info(f"Would delete {len(unsuitable)} symbols if run with --delete")
        logger.info(f"Would free up ~{sum(r['total_ticks'] for r in unsuitable):,} tick records")
        logger.info("")
        logger.info("To actually delete these symbols, run:")
        logger.info("  python scripts/analyze_and_cleanup_symbols.py --delete")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Symbols suitable for trading: {len(suitable)}")
    logger.info(f"  - Tier 1 (High): {len(tier1)}")
    logger.info(f"  - Tier 2 (Medium): {len(tier2)}")
    logger.info(f"  - Tier 3 (Low): {len(tier3)}")
    logger.info("")
    logger.info(f"Symbols NOT suitable: {len(unsuitable)} {'(DELETED)' if delete_mode else '(would be deleted with --delete)'}")
    logger.info("=" * 80)

    storage.close()

    return {
        'total': len(results),
        'tier1': len(tier1),
        'tier2': len(tier2),
        'tier3': len(tier3),
        'tier4': len(tier4),
        'suitable': len(suitable),
        'unsuitable': len(unsuitable),
        'deleted': deleted_count if delete_mode else 0
    }


if __name__ == '__main__':
    main()
