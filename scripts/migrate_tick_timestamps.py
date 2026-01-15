#!/usr/bin/env python3
"""
Migrate tick tables to epoch-millisecond timestamps and deterministic tick ids.

Usage:
    python scripts/migrate_tick_timestamps.py
"""

import logging
import sqlite3
import hashlib
from typing import Dict, Any

from config.tick_config import TICK_DB_PATH
from data.tick_storage import to_epoch_ms

logger = logging.getLogger(__name__)


def table_info(cursor, table: str) -> Dict[str, Any]:
    cursor.execute(f"PRAGMA table_info({table})")
    return {row[1]: row[2] for row in cursor.fetchall()}


def migrate_ticks(cursor, conn) -> None:
    logger.info("Migrating ticks table...")
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ticks_new (
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

    read_cursor = conn.cursor()
    write_cursor = conn.cursor()

    read_cursor.execute("SELECT symbol, timestamp, price, size, exchange, trade_id FROM ticks")
    rows = read_cursor.fetchmany(10_000)
    inserted = 0
    batches = 0
    while rows:
        payload = []
        for symbol, timestamp, price, size, exchange, trade_id in rows:
            ts_ms = to_epoch_ms(timestamp)
            if trade_id:
                tick_id = str(trade_id)
            else:
                key = f"{symbol}|{ts_ms}|{float(price):.6f}|{int(size)}|{exchange or ''}"
                tick_id = hashlib.sha1(key.encode("utf-8")).hexdigest()
            payload.append((symbol, ts_ms, price, size, exchange, trade_id, tick_id))

        write_cursor.executemany(
            """
            INSERT OR IGNORE INTO ticks_new
            (symbol, timestamp, price, size, exchange, trade_id, tick_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        inserted += write_cursor.rowcount
        batches += 1
        if batches % 25 == 0:
            logger.info(f"  Migrated {batches * 10_000:,} rows (inserted {inserted:,})")
        rows = read_cursor.fetchmany(10_000)

    logger.info(f"ticks_new rows inserted: {inserted}")
    conn.commit()


def migrate_backfill_status(cursor) -> None:
    logger.info("Migrating backfill_status table...")
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS backfill_status_new (
            symbol TEXT PRIMARY KEY,
            earliest_timestamp INTEGER,
            latest_timestamp INTEGER,
            total_ticks INTEGER,
            last_updated INTEGER
        )
        """
    )

    cursor.execute("SELECT symbol, earliest_timestamp, latest_timestamp, total_ticks, last_updated FROM backfill_status")
    rows = cursor.fetchall()
    payload = []
    for symbol, earliest, latest, total_ticks, last_updated in rows:
        payload.append(
            (
                symbol,
                to_epoch_ms(earliest) if earliest is not None else None,
                to_epoch_ms(latest) if latest is not None else None,
                total_ticks,
                to_epoch_ms(last_updated) if last_updated is not None else None,
            )
        )

    cursor.executemany(
        """
        INSERT OR REPLACE INTO backfill_status_new
        (symbol, earliest_timestamp, latest_timestamp, total_ticks, last_updated)
        VALUES (?, ?, ?, ?, ?)
        """,
        payload,
    )


def migrate_imbalance_bars(cursor) -> None:
    logger.info("Migrating imbalance_bars table...")
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS imbalance_bars_new (
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
        """
        SELECT symbol, bar_start, bar_end, open, high, low, close, volume, tick_count, imbalance
        FROM imbalance_bars
        """
    )
    rows = cursor.fetchall()
    payload = []
    for row in rows:
        symbol, bar_start, bar_end, open_, high, low, close, volume, tick_count, imbalance = row
        payload.append(
            (
                symbol,
                to_epoch_ms(bar_start),
                to_epoch_ms(bar_end),
                open_,
                high,
                low,
                close,
                volume,
                tick_count,
                imbalance,
            )
        )

    cursor.executemany(
        """
        INSERT OR REPLACE INTO imbalance_bars_new
        (symbol, bar_start, bar_end, open, high, low, close, volume, tick_count, imbalance)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )


def finalize_migration(cursor, conn) -> None:
    cursor.execute("DROP TABLE IF EXISTS ticks")
    cursor.execute("ALTER TABLE ticks_new RENAME TO ticks")

    cursor.execute("DROP TABLE IF EXISTS backfill_status")
    cursor.execute("ALTER TABLE backfill_status_new RENAME TO backfill_status")

    cursor.execute("DROP TABLE IF EXISTS imbalance_bars")
    cursor.execute("ALTER TABLE imbalance_bars_new RENAME TO imbalance_bars")

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticks_symbol_time ON ticks(symbol, timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bars_symbol_end ON imbalance_bars(symbol, bar_end)")

    conn.commit()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not TICK_DB_PATH.exists():
        logger.error(f"Database not found at {TICK_DB_PATH}")
        return 1

    conn = sqlite3.connect(str(TICK_DB_PATH))
    try:
        cursor = conn.cursor()

        if "ticks" not in {row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")}:
            logger.error("No ticks table found. Run init_tick_tables.py first.")
            return 1

        ticks_info = table_info(cursor, "ticks")
        if "tick_id" in ticks_info and "INT" in ticks_info.get("timestamp", "").upper():
            logger.info("ticks table already in epoch-ms format with tick_id. Skipping migration.")
            return 0

        migrate_ticks(cursor, conn)
        if "backfill_status" in {row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")}:
            migrate_backfill_status(cursor)
        if "imbalance_bars" in {row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")}:
            migrate_imbalance_bars(cursor)

        finalize_migration(cursor, conn)
        logger.info("✓ Migration complete")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
