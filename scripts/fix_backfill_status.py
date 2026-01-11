#!/usr/bin/env python3
"""
Fix Backfill Status Database Issue

ROOT CAUSE IDENTIFIED:
The backfill_status table contains stale entries claiming symbols have been
backfilled when they haven't. When backfill_ticks.py runs, it checks this table
and returns early if data exists, preventing new data from being saved.

This script:
1. Verifies actual tick counts vs backfill_status claims
2. Removes inconsistent entries
3. Optionally clears all backfill_status for fresh start

Usage:
    # Check for inconsistencies:
    python scripts/fix_backfill_status.py --check

    # Fix inconsistencies (remove stale entries):
    python scripts/fix_backfill_status.py --fix

    # Clear ALL backfill_status (nuclear option):
    python scripts/fix_backfill_status.py --clear-all

    # Fix for specific symbol:
    python scripts/fix_backfill_status.py --fix --symbol QQQ
"""

import sqlite3
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.tick_config import TICK_DB_PATH


def check_inconsistencies(symbol_filter=None):
    """Check for inconsistencies between backfill_status and actual ticks."""
    print("=" * 80)
    print("CHECKING BACKFILL_STATUS INCONSISTENCIES")
    print("=" * 80)

    conn = sqlite3.connect(str(TICK_DB_PATH))
    cursor = conn.cursor()

    # Get all backfill_status entries
    if symbol_filter:
        cursor.execute("SELECT * FROM backfill_status WHERE symbol = ?", (symbol_filter,))
    else:
        cursor.execute("SELECT * FROM backfill_status ORDER BY symbol")

    backfill_entries = cursor.fetchall()

    inconsistencies = []
    total_checked = 0

    for symbol, earliest, latest, claimed_ticks, last_updated in backfill_entries:
        total_checked += 1

        # Get actual tick count
        cursor.execute("SELECT COUNT(*) FROM ticks WHERE symbol = ?", (symbol,))
        actual_ticks = cursor.fetchone()[0]

        if actual_ticks != claimed_ticks:
            inconsistencies.append({
                'symbol': symbol,
                'claimed': claimed_ticks,
                'actual': actual_ticks,
                'diff': actual_ticks - claimed_ticks,
                'earliest': earliest,
                'latest': latest,
                'last_updated': last_updated
            })

    conn.close()

    print(f"\nChecked {total_checked} symbols in backfill_status")
    print(f"Found {len(inconsistencies)} inconsistencies\n")

    if inconsistencies:
        print("INCONSISTENT ENTRIES:")
        print("-" * 80)
        for entry in inconsistencies:
            status = "STALE" if entry['actual'] == 0 else "MISMATCH"
            print(f"{entry['symbol']:10} [{status:8}] Claimed: {entry['claimed']:>10,} | Actual: {entry['actual']:>10,} | Diff: {entry['diff']:>10,}")

        print()
        print("CRITICAL SYMBOLS (no actual data despite backfill_status claim):")
        critical = [e for e in inconsistencies if e['actual'] == 0]
        if critical:
            for entry in critical:
                print(f"  ❌ {entry['symbol']}: backfill_status says {entry['claimed']:,} ticks, but database has 0")
        else:
            print("  None")

    else:
        print("✓ No inconsistencies found!")

    return inconsistencies


def fix_inconsistencies(symbol_filter=None, dry_run=False):
    """Remove stale backfill_status entries where actual ticks = 0."""
    inconsistencies = check_inconsistencies(symbol_filter)

    if not inconsistencies:
        print("\n✓ Nothing to fix!")
        return

    # Filter to only stale entries (actual_ticks = 0)
    stale_entries = [e for e in inconsistencies if e['actual'] == 0]

    if not stale_entries:
        print("\n✓ No stale entries to remove (all mismatches have some data)")
        return

    print("\n" + "=" * 80)
    print("FIXING STALE BACKFILL_STATUS ENTRIES")
    print("=" * 80)

    conn = sqlite3.connect(str(TICK_DB_PATH))
    cursor = conn.cursor()

    removed_count = 0

    for entry in stale_entries:
        symbol = entry['symbol']

        if dry_run:
            print(f"[DRY RUN] Would remove: {symbol} ({entry['claimed']:,} claimed ticks)")
        else:
            cursor.execute("DELETE FROM backfill_status WHERE symbol = ?", (symbol,))
            removed_count += 1
            print(f"✓ Removed: {symbol} ({entry['claimed']:,} claimed ticks)")

    if not dry_run:
        conn.commit()
        print(f"\n✓ Removed {removed_count} stale entries")

    conn.close()

    if dry_run:
        print("\n(DRY RUN - no changes made. Run without --dry-run to apply changes)")


def clear_all_backfill_status(confirm=False):
    """Clear ALL backfill_status entries (nuclear option)."""
    if not confirm:
        print("ERROR: --clear-all requires --confirm flag for safety")
        return

    print("=" * 80)
    print("⚠️  CLEARING ALL BACKFILL_STATUS ENTRIES")
    print("=" * 80)

    conn = sqlite3.connect(str(TICK_DB_PATH))
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM backfill_status")
    count = cursor.fetchone()[0]

    cursor.execute("DELETE FROM backfill_status")
    conn.commit()

    conn.close()

    print(f"✓ Cleared {count} entries from backfill_status")
    print("\nYou can now run backfill_ticks.py and it will re-fetch everything")


def main():
    parser = argparse.ArgumentParser(
        description="Fix backfill_status database inconsistencies"
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check for inconsistencies (read-only)'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Fix inconsistencies by removing stale entries'
    )
    parser.add_argument(
        '--clear-all',
        action='store_true',
        help='Clear ALL backfill_status entries (requires --confirm)'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        help='Limit operations to specific symbol'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Required confirmation for --clear-all'
    )

    args = parser.parse_args()

    if not any([args.check, args.fix, args.clear_all]):
        parser.print_help()
        print("\nERROR: Must specify --check, --fix, or --clear-all")
        sys.exit(1)

    if args.clear_all:
        clear_all_backfill_status(confirm=args.confirm)
    elif args.fix:
        fix_inconsistencies(symbol_filter=args.symbol, dry_run=args.dry_run)
    elif args.check:
        check_inconsistencies(symbol_filter=args.symbol)


if __name__ == "__main__":
    main()
