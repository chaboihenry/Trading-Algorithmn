#!/usr/bin/env python3
"""
Backfill Status Diagnostic Script

Analyzes the current state of tick data backfill and provides actionable insights:
- What symbols have been backfilled
- How much data each symbol has
- Which symbols need more data
- Recommendations for completing backfill

Usage:
    python scripts/diagnose_backfill_status.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

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
from utils.market_calendar import get_trading_days, MARKET_TZ

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Minimum ticks for ML training (from backfill_ticks.py)
MIN_TICKS_SUFFICIENT = 1_000_000


def analyze_symbol_status(storage: TickStorage, symbol: str) -> dict:
    """
    Analyze the backfill status for a single symbol.

    Returns:
        Dict with status information
    """
    status = storage.get_backfill_status(symbol)
    tick_count = storage.get_tick_count(symbol)

    if not status or tick_count == 0:
        return {
            'symbol': symbol,
            'status': 'NOT_STARTED',
            'tick_count': 0,
            'earliest': None,
            'latest': None,
            'days_covered': 0,
            'sufficient': False,
            'percentage': 0.0
        }

    # Parse dates
    try:
        earliest_str = status['earliest_timestamp'].split('+')[0].split('T')[0] if 'T' in status['earliest_timestamp'] or '+' in status['earliest_timestamp'] else status['earliest_timestamp'].split()[0]
        latest_str = status['latest_timestamp'].split('+')[0].split('T')[0] if 'T' in status['latest_timestamp'] or '+' in status['latest_timestamp'] else status['latest_timestamp'].split()[0]

        earliest_date = datetime.fromisoformat(earliest_str).date()
        latest_date = datetime.fromisoformat(latest_str).date()
        days_covered = (latest_date - earliest_date).days + 1
    except:
        earliest_date = None
        latest_date = None
        days_covered = 0

    # Determine status
    if tick_count >= MIN_TICKS_SUFFICIENT:
        status_str = 'SUFFICIENT'
    elif tick_count > 0:
        status_str = 'IN_PROGRESS'
    else:
        status_str = 'EMPTY'

    percentage = (tick_count / MIN_TICKS_SUFFICIENT) * 100

    return {
        'symbol': symbol,
        'status': status_str,
        'tick_count': tick_count,
        'earliest': earliest_date,
        'latest': latest_date,
        'days_covered': days_covered,
        'sufficient': tick_count >= MIN_TICKS_SUFFICIENT,
        'percentage': percentage
    }


def calculate_missing_days(storage: TickStorage, symbol: str, target_days: int) -> int:
    """
    Calculate how many trading days are missing for a symbol.

    Args:
        storage: TickStorage instance
        symbol: Stock ticker
        target_days: Target number of days to backfill

    Returns:
        Number of missing trading days
    """
    status = storage.get_backfill_status(symbol)

    if not status:
        # No data at all - need all days
        end_date = datetime.now(MARKET_TZ).date()
        start_date = end_date - timedelta(days=target_days)
        trading_days = get_trading_days(start_date, end_date)
        return len(trading_days)

    # Calculate target date range
    end_date = datetime.now(MARKET_TZ).date()
    start_date = end_date - timedelta(days=target_days)

    # Get what we have
    try:
        earliest_str = status['earliest_timestamp'].split('+')[0].split('T')[0] if 'T' in status['earliest_timestamp'] or '+' in status['earliest_timestamp'] else status['earliest_timestamp'].split()[0]
        latest_str = status['latest_timestamp'].split('+')[0].split('T')[0] if 'T' in status['latest_timestamp'] or '+' in status['latest_timestamp'] else status['latest_timestamp'].split()[0]

        existing_start = datetime.fromisoformat(earliest_str).date()
        existing_end = datetime.fromisoformat(latest_str).date()
    except:
        # Can't parse dates - assume all missing
        trading_days = get_trading_days(start_date, end_date)
        return len(trading_days)

    # Calculate missing days (before existing_start or after existing_end)
    all_trading_days = get_trading_days(start_date, end_date)
    missing_days = [
        day for day in all_trading_days
        if day < existing_start or day > existing_end
    ]

    return len(missing_days)


def estimate_ticks_per_day(symbol: str) -> int:
    """
    Estimate average ticks per day for a symbol based on liquidity.

    Rough estimates:
    - High liquidity (SPY, QQQ, AAPL): ~50,000 ticks/day
    - Medium liquidity (most stocks): ~10,000 ticks/day
    - Low liquidity (small caps): ~2,000 ticks/day
    """
    # High liquidity symbols
    high_liquidity = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'GOOGL', 'META']

    # Medium-high liquidity
    medium_high = ['DIA', 'IWM', 'XLF', 'GLD', 'TLT']

    if symbol in high_liquidity:
        return 50000  # Very liquid
    elif symbol in medium_high:
        return 20000  # Medium-high
    else:
        return 10000  # Default assumption


def main():
    """Main diagnostic function."""

    print("=" * 100)
    print("TICK DATA BACKFILL STATUS DIAGNOSTIC")
    print("=" * 100)
    print(f"Database: {TICK_DB_PATH}")
    print(f"Data feed: {FEED_NAME}")
    print(f"Target backfill days: {BACKFILL_DAYS}")
    print(f"Symbols configured: {len(SYMBOLS)}")
    print(f"Minimum ticks for training: {MIN_TICKS_SUFFICIENT:,}")
    print("=" * 100)

    # Initialize storage
    storage = TickStorage(str(TICK_DB_PATH))

    # Analyze each symbol
    print("\n" + "=" * 100)
    print("SYMBOL-BY-SYMBOL ANALYSIS")
    print("=" * 100)
    print(f"{'Symbol':<8} {'Status':<15} {'Ticks':<15} {'Progress':<12} {'Date Range':<30} {'Days':<6}")
    print("-" * 100)

    all_statuses = []

    for symbol in SYMBOLS:
        status = analyze_symbol_status(storage, symbol)
        all_statuses.append(status)

        # Format output
        symbol_str = status['symbol']
        status_str = status['status']
        ticks_str = f"{status['tick_count']:,}"
        progress_str = f"{status['percentage']:.1f}%"

        if status['earliest'] and status['latest']:
            date_range_str = f"{status['earliest']} to {status['latest']}"
        else:
            date_range_str = "No data"

        days_str = str(status['days_covered'])

        # Color code status
        if status['sufficient']:
            status_display = f"✓ {status_str}"
        elif status['status'] == 'IN_PROGRESS':
            status_display = f"⏳ {status_str}"
        else:
            status_display = f"✗ {status_str}"

        print(f"{symbol_str:<8} {status_display:<15} {ticks_str:<15} {progress_str:<12} {date_range_str:<30} {days_str:<6}")

    # Summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)

    sufficient_count = sum(1 for s in all_statuses if s['sufficient'])
    in_progress_count = sum(1 for s in all_statuses if s['status'] == 'IN_PROGRESS' and not s['sufficient'])
    not_started_count = sum(1 for s in all_statuses if s['status'] == 'NOT_STARTED')

    total_ticks = sum(s['tick_count'] for s in all_statuses)
    avg_ticks = total_ticks / len(all_statuses) if all_statuses else 0

    print(f"Symbols with sufficient data: {sufficient_count}/{len(SYMBOLS)} ({sufficient_count/len(SYMBOLS)*100:.1f}%)")
    print(f"Symbols in progress: {in_progress_count}")
    print(f"Symbols not started: {not_started_count}")
    print(f"Total ticks collected: {total_ticks:,}")
    print(f"Average ticks per symbol: {avg_ticks:,.0f}")

    # Problematic symbols
    print("\n" + "=" * 100)
    print("SYMBOLS NEEDING ATTENTION")
    print("=" * 100)

    problematic = [s for s in all_statuses if not s['sufficient']]

    if not problematic:
        print("✓ All symbols have sufficient data!")
    else:
        print(f"{'Symbol':<8} {'Current':<15} {'Needed':<15} {'Missing':<15} {'Est. Days Needed':<20}")
        print("-" * 100)

        for status in problematic:
            symbol = status['symbol']
            current = status['tick_count']
            needed = MIN_TICKS_SUFFICIENT
            missing = max(0, needed - current)

            # Estimate days needed
            est_ticks_per_day = estimate_ticks_per_day(symbol)
            days_needed = missing / est_ticks_per_day if est_ticks_per_day > 0 else 0

            print(f"{symbol:<8} {current:,}{'':>5} {needed:,}{'':>5} {missing:,}{'':>5} {days_needed:,.0f} days")

    # Calculate missing days for backfill
    print("\n" + "=" * 100)
    print("BACKFILL COMPLETENESS")
    print("=" * 100)

    end_date = datetime.now(MARKET_TZ).date()
    start_date = end_date - timedelta(days=BACKFILL_DAYS)
    total_trading_days = len(get_trading_days(start_date, end_date))

    print(f"Target date range: {start_date} to {end_date}")
    print(f"Total trading days in range: {total_trading_days}")
    print("")

    for symbol in SYMBOLS:
        missing_days = calculate_missing_days(storage, symbol, BACKFILL_DAYS)
        covered_days = total_trading_days - missing_days
        completeness = (covered_days / total_trading_days * 100) if total_trading_days > 0 else 0

        if missing_days == 0:
            print(f"{symbol:<8} ✓ Complete ({covered_days}/{total_trading_days} days, {completeness:.1f}%)")
        elif missing_days < 10:
            print(f"{symbol:<8} ⚠ Nearly complete ({covered_days}/{total_trading_days} days, {completeness:.1f}%, {missing_days} days missing)")
        else:
            print(f"{symbol:<8} ✗ Incomplete ({covered_days}/{total_trading_days} days, {completeness:.1f}%, {missing_days} days missing)")

    # Recommendations
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)

    if sufficient_count == len(SYMBOLS):
        print("✓ All symbols have sufficient data for ML training!")
        print("\nYou can proceed to:")
        print("  1. Run model training: python scripts/train_all_symbols.py")
        print("  2. Start live trading: python main.py")
    elif in_progress_count > 0:
        print("⏳ Backfill is in progress. Please wait for completion.")
        print("\nTo monitor progress:")
        print("  - Watch the backfill script output")
        print("  - Re-run this diagnostic periodically: python scripts/diagnose_backfill_status.py")
        print("\nIf backfill is stuck:")
        print("  - Check for API rate limits")
        print("  - Check your Alpaca subscription (IEX vs SIP)")
        print("  - Some symbols may be less liquid and have fewer ticks")
    else:
        print("✗ Some symbols have not started backfilling.")
        print("\nTo start/resume backfill:")
        print("  - All symbols: python scripts/setup/backfill_ticks.py")
        print("  - Specific symbol: python scripts/setup/backfill_ticks.py --symbol MMC")
        print("  - Force re-fetch: python scripts/setup/backfill_ticks.py --force")

    # Less liquid symbols warning
    low_tick_symbols = [s for s in all_statuses if 0 < s['tick_count'] < MIN_TICKS_SUFFICIENT / 2]
    if low_tick_symbols:
        print("\n⚠️  LOW LIQUIDITY WARNING")
        print("-" * 100)
        print("These symbols have low tick counts even after backfill:")
        for s in low_tick_symbols:
            print(f"  - {s['symbol']}: {s['tick_count']:,} ticks")
        print("\nPossible reasons:")
        print("  1. IEX feed (free) has less data than SIP feed (paid)")
        print("  2. Symbol is less liquid (fewer trades per day)")
        print("  3. Need longer backfill period (increase BACKFILL_DAYS in config)")
        print("\nOptions:")
        print("  - Upgrade to SIP data feed for more complete data")
        print("  - Increase BACKFILL_DAYS to 180 or 365 days")
        print("  - Remove low-liquidity symbols from SYMBOLS list")

    print("\n" + "=" * 100)

    storage.close()


if __name__ == "__main__":
    main()
