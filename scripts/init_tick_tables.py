"""
Initialize SQLite tables for tick storage.

Usage:
    python scripts/init_tick_tables.py
"""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


def init_tables(db_path: Path) -> None:
    """Create required tables and indexes if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                price REAL NOT NULL,
                size INTEGER NOT NULL,
                exchange TEXT,
                trade_id TEXT,
                tick_id TEXT NOT NULL,
                UNIQUE(symbol, tick_id)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS backfill_status (
                symbol TEXT PRIMARY KEY,
                earliest_timestamp INTEGER,
                latest_timestamp INTEGER,
                total_ticks INTEGER,
                last_updated INTEGER
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS imbalance_bars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                bar_start INTEGER NOT NULL,
                bar_end INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                tick_count INTEGER NOT NULL,
                imbalance REAL,
                UNIQUE(symbol, bar_start, bar_end)
            )
            """
        )

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_ticks_symbol_time ON ticks(symbol, timestamp)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_bars_symbol_end ON imbalance_bars(symbol, bar_end)"
        )

        conn.commit()
        logger.info(f"✓ Initialized tick tables at {db_path}")
    finally:
        conn.close()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from config.tick_config import TICK_DB_PATH
    init_tables(TICK_DB_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
