#!/usr/bin/env python3
"""
Initialize Tick Data Database Tables

This script creates all necessary tables in the existing tick data database.
You created the database file manually using DB Browser for SQLite, but it's
empty (no tables). This script adds the required table structure.

Tables created:
1. ticks - Stores individual trade ticks from Alpaca
2. imbalance_bars - Stores generated tick imbalance bars
3. backfill_status - Tracks which data has been fetched

OOP Concepts:
- Uses sqlite3 module (built into Python) for database operations
- Connection object: Represents database connection
- Cursor object: Executes SQL commands
- Context manager (with statement): Ensures proper cleanup

Usage:
    python scripts/init_tick_tables.py

The script is idempotent - safe to run multiple times. If tables already
exist, it will skip creation.
"""

import sqlite3
import sys
from pathlib import Path

# Add project root to Python path to import our config
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.tick_config import TICK_DB_PATH


def create_tables():
    """
    Create all required tables in the tick database.

    This function:
    1. Connects to the existing database
    2. Creates tables using CREATE TABLE IF NOT EXISTS (safe to re-run)
    3. Creates indexes for fast queries
    4. Commits changes and closes connection

    Tables:
    - ticks: Raw tick data from Alpaca
    - imbalance_bars: Generated tick imbalance bars (OHLCV format)
    - backfill_status: Tracking table to avoid re-fetching data
    """
    print("=" * 80)
    print("INITIALIZING TICK DATA DATABASE")
    print("=" * 80)

    # Check that database path exists
    db_path = Path(TICK_DB_PATH)

    # Create empty database file if it doesn't exist
    # (In your case, you already created it with DB Browser)
    if not db_path.exists():
        print(f"Database file not found at: {TICK_DB_PATH}")
        print(f"Creating new database...")
        db_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Found existing database at: {TICK_DB_PATH}")

    # Connect to database
    # sqlite3.connect() creates a Connection object
    conn = sqlite3.connect(TICK_DB_PATH)

    # Create a Cursor object to execute SQL commands
    # Think of cursor as the "worker" that carries out database operations
    cursor = conn.cursor()

    print("\nCreating tables...")

    # -------------------------------------------------------------------
    # TABLE 1: ticks
    # Stores individual trade ticks from Alpaca SIP or IEX feed
    # -------------------------------------------------------------------
    print("  [1/3] Creating 'ticks' table...")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ticks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            price REAL NOT NULL,
            size INTEGER NOT NULL,
            exchange TEXT,
            trade_id TEXT,
            UNIQUE(symbol, timestamp, trade_id)
        )
    """)

    # Index for fast queries by symbol and timestamp
    # Without this index, queries would scan the entire table (slow!)
    # With index, database can jump directly to relevant rows
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_ticks_symbol_timestamp
        ON ticks(symbol, timestamp)
    """)

    print("     ✓ 'ticks' table created")
    print("     ✓ Index on (symbol, timestamp) created")

    # -------------------------------------------------------------------
    # TABLE 2: imbalance_bars
    # Stores tick imbalance bars generated from raw ticks
    # -------------------------------------------------------------------
    print("  [2/3] Creating 'imbalance_bars' table...")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS imbalance_bars (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            bar_start TEXT NOT NULL,
            bar_end TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            tick_count INTEGER,
            imbalance REAL,
            UNIQUE(symbol, bar_end)
        )
    """)

    # Index for querying bars by symbol and time
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_bars_symbol_end
        ON imbalance_bars(symbol, bar_end)
    """)

    print("     ✓ 'imbalance_bars' table created")
    print("     ✓ Index on (symbol, bar_end) created")

    # -------------------------------------------------------------------
    # TABLE 3: backfill_status
    # Tracks which symbols have been backfilled and date ranges
    # -------------------------------------------------------------------
    print("  [3/3] Creating 'backfill_status' table...")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS backfill_status (
            symbol TEXT PRIMARY KEY,
            earliest_timestamp TEXT,
            latest_timestamp TEXT,
            total_ticks INTEGER,
            last_updated TEXT
        )
    """)

    print("     ✓ 'backfill_status' table created")

    # Commit all changes to disk
    # This makes all table creations permanent
    conn.commit()

    print("\n" + "=" * 80)
    print("DATABASE INITIALIZATION COMPLETE")
    print("=" * 80)

    # Get database stats
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table'
        ORDER BY name
    """)
    tables = cursor.fetchall()

    print(f"\nDatabase: {TICK_DB_PATH}")
    print(f"Tables created: {len(tables)}")
    for table in tables:
        print(f"  - {table[0]}")

    # Get database file size
    db_size = db_path.stat().st_size
    print(f"\nDatabase size: {db_size:,} bytes ({db_size / 1024:.2f} KB)")

    # Close connection (cleanup)
    conn.close()

    print("\n✓ Ready for tick data operations!")
    print("  Next step: Run scripts/backfill_ticks.py to fetch historical data")


def verify_tables():
    """
    Verify that all expected tables exist and show their structure.

    This is a diagnostic function to confirm everything was created correctly.
    """
    print("\n" + "=" * 80)
    print("VERIFYING TABLE STRUCTURE")
    print("=" * 80)

    conn = sqlite3.connect(TICK_DB_PATH)
    cursor = conn.cursor()

    expected_tables = ['ticks', 'imbalance_bars', 'backfill_status']

    for table_name in expected_tables:
        print(f"\nTable: {table_name}")
        print("-" * 40)

        # Get column information
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()

        for col in columns:
            col_id, col_name, col_type, not_null, default_val, is_pk = col
            pk_marker = " [PRIMARY KEY]" if is_pk else ""
            null_marker = " NOT NULL" if not_null else ""
            print(f"  {col_name:20} {col_type:10}{null_marker}{pk_marker}")

        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        print(f"\nCurrent rows: {row_count}")

    conn.close()


if __name__ == "__main__":
    """
    Main entry point when script is run directly.

    The if __name__ == "__main__" pattern is a Python idiom that lets
    you write code that only runs when the script is executed directly,
    not when it's imported as a module.
    """
    try:
        create_tables()
        verify_tables()

        print("\n" + "=" * 80)
        print("SUCCESS! Database is ready for tick data.")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error initializing database: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
