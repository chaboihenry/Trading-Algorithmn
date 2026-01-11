#!/usr/bin/env python3
"""
Database Diagnostic Script

Inspects the tick database to identify data integrity issues.
"""

import sqlite3
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.tick_config import TICK_DB_PATH

def diagnose_database():
    """Run comprehensive database diagnostics."""
    print("=" * 80)
    print("DATABASE DIAGNOSTICS")
    print("=" * 80)
    print(f"Database: {TICK_DB_PATH}")
    print(f"Exists: {TICK_DB_PATH.exists()}")
    print(f"Size: {TICK_DB_PATH.stat().st_size / 1024 / 1024:.2f} MB")
    print()

    conn = sqlite3.connect(str(TICK_DB_PATH))
    cursor = conn.cursor()

    # 1. Check ticks table
    print("TICKS TABLE:")
    print("-" * 80)

    cursor.execute("SELECT COUNT(*) FROM ticks")
    total_ticks = cursor.fetchone()[0]
    print(f"Total rows: {total_ticks:,}")

    cursor.execute("SELECT DISTINCT symbol FROM ticks ORDER BY symbol")
    symbols = [row[0] for row in cursor.fetchall()]
    print(f"Symbols with data: {symbols}")

    for symbol in symbols:
        cursor.execute("SELECT COUNT(*) FROM ticks WHERE symbol = ?", (symbol,))
        count = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM ticks WHERE symbol = ?", (symbol,))
        min_ts, max_ts = cursor.fetchone()

        print(f"  {symbol}: {count:,} ticks ({min_ts} to {max_ts})")

    print()

    # 2. Check imbalance_bars table
    print("IMBALANCE_BARS TABLE:")
    print("-" * 80)

    cursor.execute("SELECT COUNT(*) FROM imbalance_bars")
    total_bars = cursor.fetchone()[0]
    print(f"Total rows: {total_bars:,}")

    if total_bars > 0:
        cursor.execute("SELECT DISTINCT symbol FROM imbalance_bars ORDER BY symbol")
        bar_symbols = [row[0] for row in cursor.fetchall()]
        print(f"Symbols with bars: {bar_symbols}")

        for symbol in bar_symbols:
            cursor.execute("SELECT COUNT(*) FROM imbalance_bars WHERE symbol = ?", (symbol,))
            count = cursor.fetchone()[0]
            print(f"  {symbol}: {count:,} bars")
    else:
        print("  (empty)")

    print()

    # 3. Check backfill_status table
    print("BACKFILL_STATUS TABLE:")
    print("-" * 80)

    cursor.execute("SELECT * FROM backfill_status ORDER BY symbol")
    rows = cursor.fetchall()

    if rows:
        print(f"Total entries: {len(rows)}")
        for row in rows:
            symbol, earliest, latest, total, last_updated = row
            print(f"  {symbol}:")
            print(f"    Range: {earliest} to {latest}")
            print(f"    Claimed ticks: {total:,}")
            print(f"    Last updated: {last_updated}")

            # Verify against actual ticks
            cursor.execute("SELECT COUNT(*) FROM ticks WHERE symbol = ?", (symbol,))
            actual_count = cursor.fetchone()[0]

            if actual_count != total:
                print(f"    ⚠️  MISMATCH: backfill_status says {total:,}, actual is {actual_count:,}")
    else:
        print("  (empty)")

    print()

    # 4. Check database integrity
    print("DATABASE INTEGRITY:")
    print("-" * 80)

    cursor.execute("PRAGMA integrity_check")
    integrity = cursor.fetchone()[0]
    print(f"Integrity check: {integrity}")

    cursor.execute("PRAGMA foreign_key_check")
    fk_errors = cursor.fetchall()
    if fk_errors:
        print(f"Foreign key errors: {len(fk_errors)}")
        for error in fk_errors[:5]:
            print(f"  {error}")
    else:
        print("Foreign key check: OK")

    print()

    # 5. Test a simple insert
    print("CONNECTION TEST:")
    print("-" * 80)

    try:
        cursor.execute("""
            INSERT INTO ticks (symbol, timestamp, price, size)
            VALUES ('TEST', '2025-01-01 00:00:00', 100.0, 100)
        """)
        print("Test insert: OK (not committed)")

        conn.rollback()
        print("Rollback: OK")

        cursor.execute("SELECT COUNT(*) FROM ticks WHERE symbol='TEST'")
        test_count = cursor.fetchone()[0]
        print(f"After rollback: {test_count} TEST rows (should be 0)")

    except Exception as e:
        print(f"Test insert failed: {e}")

    conn.close()

    print()
    print("=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    diagnose_database()
