"""
Tick Data Storage Module

This module handles all database operations for tick data persistence.
Think of it as the "warehouse manager" - it knows how to store ticks,
retrieve them, and keep track of what's been collected.

OOP Concepts Used:
1. Class: TickStorage encapsulates all database operations
2. Methods: Functions that belong to the class and operate on its data
3. Context Manager (with statement): Ensures database connections are properly closed
4. Type Hints: Documents what type of data each parameter expects

Why use a class here?
- Encapsulation: All database logic in one place
- State management: Keeps track of database path
- Reusability: Create multiple instances for different databases if needed
- Clean interface: Users don't need to know SQL, just call methods
"""

import sqlite3
import logging
import hashlib
import pytz
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# TIMEZONE UTILITIES
# =============================================================================
ET_TZ = pytz.timezone('America/New_York')
UTC_TZ = pytz.UTC


def to_utc(dt):
    """Convert any datetime to UTC."""
    if dt.tzinfo is None:
        # Assume ET if naive
        dt = ET_TZ.localize(dt)
    return dt.astimezone(UTC_TZ)


def to_et(dt):
    """Convert any datetime to Eastern Time."""
    if dt.tzinfo is None:
        dt = UTC_TZ.localize(dt)
    return dt.astimezone(ET_TZ)


def to_epoch_ms(value) -> int:
    """Convert datetime/ISO string/epoch to epoch milliseconds (UTC)."""
    if isinstance(value, (int, float)):
        return int(value)

    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError as exc:
            raise ValueError(f"Invalid timestamp string: {value}") from exc
    elif isinstance(value, datetime):
        dt = value
    else:
        raise TypeError(f"Unsupported timestamp type: {type(value)}")

    dt = to_utc(dt)
    return int(dt.timestamp() * 1000)


def epoch_ms_to_datetime(value: int) -> datetime:
    """Convert epoch milliseconds to timezone-aware datetime in UTC."""
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc)


class TickStorage:
    """
    Manages tick data persistence in SQLite database.

    This class is like a librarian for your tick data - it knows how to:
    - Save new ticks to the database (save_ticks)
    - Find and load old ticks (load_ticks)
    - Track what's already stored (get_backfill_status, update_backfill_status)
    - Count how many ticks you have (get_tick_count)

    The class uses a "lazy connection" pattern - it doesn't connect to the
    database until you actually need it. This saves resources.

    Attributes:
        db_path (str): Path to the SQLite database file
        _connection (Optional[sqlite3.Connection]): Database connection (created on demand)

    Example usage:
        >>> storage = TickStorage(str(TICK_DB_PATH))
        >>> ticks = [
        ...     {'symbol': 'SPY', 'timestamp': '2024-01-01 09:30:00', 'price': 450.0, 'size': 100},
        ...     {'symbol': 'SPY', 'timestamp': '2024-01-01 09:30:01', 'price': 450.1, 'size': 50}
        ... ]
        >>> storage.save_ticks('SPY', ticks)
        >>> loaded = storage.load_ticks('SPY', start='2024-01-01', end='2024-01-02')
    """

    def __init__(self, db_path: str):
        """
        Initialize the tick storage with database path.

        Args:
            db_path: Full path to SQLite database file
                     Uses TICK_DB_PATH from config (configurable via DATA_PATH env var)

        Note:
            This doesn't open a database connection yet. The connection
            is created "lazily" when you first need it (in _get_connection).
        """
        self.db_path = db_path
        self._connection = None  # Lazy connection - created when needed

        # Verify database exists
        if not Path(db_path).exists():
            raise FileNotFoundError(
                f"Database not found at {db_path}. "
                f"Run scripts/init_tick_tables.py first to create it."
            )

        logger.debug(f"TickStorage initialized for: {db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get or create database connection (lazy loading pattern).

        This method creates a database connection the first time it's called,
        then reuses that same connection for all future calls. This is more
        efficient than creating a new connection for every operation.

        Thread Safety & Performance Optimizations:
        - timeout=30.0: Prevents immediate failures under load
        - check_same_thread=False: Allows multi-threaded access (use with care)
        - isolation_level='DEFERRED': Delays locks until needed
        - WAL mode: Write-Ahead Logging for better concurrency
        - synchronous=NORMAL: Balanced safety/performance

        Returns:
            sqlite3.Connection: Active database connection

        OOP Pattern: This is a "getter" method that ensures the connection
        exists before returning it. It's private (starts with _) because
        users of this class don't need to call it directly.
        """
        if self._connection is None:
            self._connection = sqlite3.connect(
                self.db_path,
                timeout=30.0,              # Wait up to 30s for lock (prevents immediate failures)
                check_same_thread=False,   # Allow multi-threaded access
                isolation_level='DEFERRED' # Delay locks until write actually happens
            )
            # Set row factory to return rows as dictionaries (easier to work with)
            self._connection.row_factory = sqlite3.Row

            # Enable Write-Ahead Logging for better concurrency
            # Multiple readers can access DB while one writer is writing
            self._connection.execute("PRAGMA journal_mode=WAL")

            # Balanced synchronous mode - safer than OFF, faster than FULL
            self._connection.execute("PRAGMA synchronous=NORMAL")

            # Performance tuning for large tick datasets
            self._connection.execute("PRAGMA temp_store=MEMORY")
            # Negative cache_size sets size in KB (e.g., -200000 ≈ 200MB)
            self._connection.execute("PRAGMA cache_size=-200000")

            self._verify_schema()

            logger.debug("Database connection established with WAL mode")

        return self._connection

    def _verify_schema(self) -> None:
        """Verify required tables/columns exist for the current schema."""
        cursor = self._connection.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ticks'"
        )
        if cursor.fetchone() is None:
            raise RuntimeError(
                "ticks table not found. Run scripts/init_tick_tables.py first."
            )

        cursor.execute("PRAGMA table_info(ticks)")
        columns = {row[1] for row in cursor.fetchall()}
        if "tick_id" not in columns:
            raise RuntimeError(
                "Legacy tick schema detected (missing tick_id). "
                "Run scripts/migrate_tick_timestamps.py to upgrade."
            )

    def _validate_tick(self, tick: dict, symbol: str) -> bool:
        """
        Validate tick data before saving to database.

        This prevents corrupt/invalid data from polluting the database.
        Invalid ticks are logged but skipped (not saved).

        Validation Rules:
        1. Price must be > 0 (no negative or zero prices)
        2. Price must be < $100,000 (catches obvious errors like $45,000,000)
        3. Size must be > 0 (no zero-share trades)

        Args:
            tick: Tick dictionary with 'price' and 'size' fields
            symbol: Stock symbol (for logging)

        Returns:
            bool: True if tick is valid, False if should be skipped

        Example:
            >>> tick1 = {'price': 450.0, 'size': 100}
            >>> storage._validate_tick(tick1, 'SPY')  # True
            >>> tick2 = {'price': -10.0, 'size': 100}
            >>> storage._validate_tick(tick2, 'SPY')  # False (negative price)
        """
        price = tick.get('price', 0)
        size = tick.get('size', 0)

        # Validation 1: Price must be positive
        if price <= 0:
            logger.warning(f"{symbol}: Invalid price {price} (must be > 0)")
            return False

        # Validation 2: Price sanity check (catch obvious errors)
        if price > 100000:
            logger.warning(f"{symbol}: Suspicious price {price} (> $100,000)")
            return False

        # Validation 3: Size must be positive
        if size <= 0:
            logger.warning(f"{symbol}: Invalid size {size} (must be > 0)")
            return False

        return True

    def _build_tick_id(
        self,
        symbol: str,
        timestamp_ms: int,
        price: float,
        size: int,
        exchange: Optional[str],
        trade_id: Optional[str]
    ) -> str:
        """
        Build a deterministic tick identifier.

        If trade_id is present, use it directly. Otherwise, hash a stable
        tuple of fields to ensure idempotent inserts without a trade_id.
        """
        if trade_id:
            return str(trade_id)

        key = f"{symbol}|{timestamp_ms}|{price:.6f}|{size}|{exchange or ''}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()

    def save_ticks(self, symbol: str, ticks: List[Dict]) -> int:
        """
        Save multiple ticks to database in one efficient batch operation.

        Uses executemany() which is much faster than inserting one tick at a time.
        For example, saving 10,000 ticks:
        - Individual inserts: ~10 seconds
        - Batch insert: ~0.1 seconds (100x faster!)

        Data Validation:
        - Invalid ticks (price<=0, price>$100k, size<=0) are filtered out
        - Filtered ticks are logged but not saved
        - This prevents corrupt data from entering the database

        Time Handling:
        - All timestamps are converted to UTC epoch milliseconds before storage
        - Naive timestamps are assumed to be in Eastern Time (ET)
        - This ensures consistency across different data sources

        Args:
            symbol: Stock ticker like "SPY", "QQQ", "IWM"
            ticks: List of tick dictionaries, each containing:
                   - timestamp: datetime or ISO string (converted to UTC)
                   - price: float (trade price)
                   - size: int (number of shares)
                   - exchange: str (optional, which exchange executed the trade)
                   - trade_id: str (optional, unique trade identifier)

        Returns:
            int: Number of ticks successfully saved (after validation)

        Example:
            >>> storage = TickStorage(db_path)
            >>> ticks = [
            ...     {'timestamp': '2024-01-01 09:30:00', 'price': 450.0, 'size': 100},
            ...     {'timestamp': '2024-01-01 09:30:01', 'price': 450.1, 'size': 50}
            ... ]
            >>> count = storage.save_ticks('SPY', ticks)
            >>> print(f"Saved {count} ticks")
        """
        if not ticks:
            logger.debug(f"No ticks to save for {symbol}")
            return 0

        # STEP 1: Validate ticks before processing
        # Filter out invalid ticks (bad prices, bad sizes, etc.)
        valid_ticks = [tick for tick in ticks if self._validate_tick(tick, symbol)]

        if len(valid_ticks) < len(ticks):
            invalid_count = len(ticks) - len(valid_ticks)
            logger.warning(
                f"{symbol}: Filtered {invalid_count}/{len(ticks)} invalid ticks "
                f"({invalid_count/len(ticks)*100:.1f}% rejected)"
            )

        if not valid_ticks:
            logger.warning(f"{symbol}: No valid ticks to save (all filtered)")
            return 0

        conn = self._get_connection()
        cursor = conn.cursor()

        # STEP 2: Get count BEFORE insert (for accurate reporting)
        # cursor.rowcount is unreliable with INSERT OR IGNORE
        count_before = self.get_tick_count(symbol)

        # STEP 3: Prepare data for batch insert
        # Convert list of dicts to list of tuples for SQL INSERT
        insert_data = []
        for tick in valid_ticks:
            # Convert timestamp to epoch milliseconds (UTC)
            timestamp_ms = to_epoch_ms(tick['timestamp'])
            exchange = tick.get('exchange')
            trade_id = tick.get('trade_id')
            tick_id = self._build_tick_id(
                symbol=symbol,
                timestamp_ms=timestamp_ms,
                price=tick['price'],
                size=tick['size'],
                exchange=exchange,
                trade_id=trade_id
            )

            insert_data.append((
                symbol,
                timestamp_ms,
                tick['price'],
                tick['size'],
                exchange,  # Optional field
                trade_id,  # Optional field
                tick_id
            ))

        try:
            # STEP 4: INSERT OR IGNORE
            # Skip ticks that already exist (based on UNIQUE constraint)
            # This makes the operation idempotent - safe to run multiple times
            cursor.executemany("""
                INSERT OR IGNORE INTO ticks
                (symbol, timestamp, price, size, exchange, trade_id, tick_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, insert_data)

            # Commit changes to disk (make them permanent)
            conn.commit()

            # STEP 5: Get count AFTER insert for accurate reporting
            # cursor.rowcount is UNRELIABLE with INSERT OR IGNORE (SQLite bug)
            # Instead, we calculate: saved = count_after - count_before
            count_after = self.get_tick_count(symbol)
            saved_count = count_after - count_before

            # Log results with detailed breakdown
            duplicate_count = len(valid_ticks) - saved_count
            logger.info(
                f"{symbol}: Saved {saved_count}/{len(ticks)} ticks "
                f"(filtered: {len(ticks) - len(valid_ticks)}, duplicates: {duplicate_count})"
            )

            return saved_count

        except sqlite3.Error as e:
            # If something goes wrong, roll back the transaction
            conn.rollback()
            logger.error(f"Error saving ticks for {symbol}: {e}")
            raise

    def load_ticks(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = None,
        timestamp_format: str = "epoch_ms"
    ) -> List[Tuple]:
        """
        Load ticks from database for a given symbol and time range.

        This returns ticks in a simple format: list of (timestamp, price, size) tuples.
        This format is efficient and works well with the bar generator.

        Args:
            symbol: Stock ticker
            start: Start timestamp (ISO string, datetime, or epoch ms)
            end: End timestamp (ISO string, datetime, or epoch ms)
            limit: Maximum number of ticks to return (for testing with small samples)
            timestamp_format: "epoch_ms", "datetime", or "iso"

        Returns:
            List of tuples: [(timestamp, price, size), ...]

        Example:
            >>> storage = TickStorage(db_path)
            >>> ticks = storage.load_ticks('SPY', start='2024-01-01', end='2024-01-02')
            >>> for timestamp, price, size in ticks[:5]:
            ...     print(f"{timestamp}: ${price} x {size}")
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Build SQL query based on parameters
        query = "SELECT timestamp, price, size FROM ticks WHERE symbol = ?"
        params = [symbol]

        if start:
            query += " AND timestamp >= ?"
            params.append(to_epoch_ms(start))

        if end:
            query += " AND timestamp <= ?"
            params.append(to_epoch_ms(end))

        query += " ORDER BY timestamp ASC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)

        # Fetch all rows and convert to list of tuples
        rows = cursor.fetchall()

        logger.debug(f"Loaded {len(rows)} ticks for {symbol}")

        if timestamp_format == "epoch_ms":
            return [(row['timestamp'], row['price'], row['size']) for row in rows]
        if timestamp_format == "datetime":
            return [(epoch_ms_to_datetime(row['timestamp']), row['price'], row['size']) for row in rows]
        if timestamp_format == "iso":
            return [
                (epoch_ms_to_datetime(row['timestamp']).isoformat(), row['price'], row['size'])
                for row in rows
            ]

        raise ValueError(f"Unsupported timestamp_format: {timestamp_format}")

    def get_backfill_status(self, symbol: str) -> Optional[Dict]:
        """
        Check the backfill status for a symbol.

        This tells you what data you already have, so you don't re-fetch it.

        Args:
            symbol: Stock ticker

        Returns:
            Dict with status info, or None if symbol hasn't been backfilled yet:
            {
                'symbol': 'SPY',
                'earliest_timestamp': '2024-01-01 04:00:00',
                'latest_timestamp': '2024-03-01 20:00:00',
                'total_ticks': 5000000,
                'last_updated': '2024-03-02 10:30:00'
            }

        Example:
            >>> storage = TickStorage(db_path)
            >>> status = storage.get_backfill_status('SPY')
            >>> if status:
            ...     print(f"Have {status['total_ticks']} ticks from {status['earliest_timestamp']}")
            ... else:
            ...     print("No data yet - run backfill")
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM backfill_status WHERE symbol = ?
        """, (symbol,))

        row = cursor.fetchone()

        if row:
            return dict(row)  # Convert sqlite3.Row to dict
        return None

    def update_backfill_status(
        self,
        symbol: str,
        earliest,
        latest,
        count: int
    ):
        """
        Update the backfill status for a symbol.

        Call this after fetching new tick data to keep track of what you have.

        Time Handling:
        - Timestamps are stored in UTC epoch milliseconds for consistency
        - Input timestamps can be ISO strings, datetimes, or epoch ms

        Args:
            symbol: Stock ticker
            earliest: Earliest timestamp in the data
            latest: Latest timestamp in the data
            count: Total number of ticks stored

        Example:
            >>> storage = TickStorage(db_path)
            >>> storage.update_backfill_status(
            ...     'SPY',
            ...     earliest='2024-01-01 04:00:00',
            ...     latest='2024-03-01 20:00:00',
            ...     count=5000000
            ... )
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get current timestamp in UTC epoch ms
        now = int(datetime.now(UTC_TZ).timestamp() * 1000)

        # INSERT OR REPLACE: Update if exists, insert if doesn't exist
        cursor.execute("""
            INSERT OR REPLACE INTO backfill_status
            (symbol, earliest_timestamp, latest_timestamp, total_ticks, last_updated)
            VALUES (?, ?, ?, ?, ?)
        """, (symbol, to_epoch_ms(earliest), to_epoch_ms(latest), count, now))

        conn.commit()

        logger.info(
            f"Updated backfill status for {symbol}: "
            f"{count} ticks from {to_epoch_ms(earliest)} to {to_epoch_ms(latest)}"
        )

    def get_tick_count(self, symbol: str) -> int:
        """
        Get total number of ticks stored for a symbol.

        This is useful for monitoring and validation.

        Args:
            symbol: Stock ticker

        Returns:
            int: Total tick count

        Example:
            >>> storage = TickStorage(db_path)
            >>> count = storage.get_tick_count('SPY')
            >>> print(f"SPY has {count:,} ticks stored")
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COUNT(*) FROM ticks WHERE symbol = ?
        """, (symbol,))

        return cursor.fetchone()[0]

    def get_date_range(self, symbol: str) -> Optional[Tuple[int, int]]:
        """
        Get the date range of ticks stored for a symbol.

        Returns:
            Tuple of (earliest_ms, latest_ms) timestamps, or None if no data

        Example:
            >>> storage = TickStorage(db_path)
            >>> range = storage.get_date_range('SPY')
            >>> if range:
            ...     earliest, latest = range
            ...     print(f"Data from {earliest} to {latest}")
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT MIN(timestamp), MAX(timestamp)
            FROM ticks
            WHERE symbol = ?
        """, (symbol,))

        row = cursor.fetchone()

        if row[0] and row[1]:
            return (row[0], row[1])
        return None

    def save_bars(self, symbol: str, bars: List[Dict]) -> int:
        """
        Save generated imbalance bars to database.

        After generating bars from ticks, save them here for fast access.
        This avoids regenerating bars every time you need them.

        Args:
            symbol: Stock ticker
            bars: List of bar dictionaries with OHLCV data

        Returns:
            int: Number of bars saved

        Example:
            >>> bars = [
            ...     {
            ...         'bar_start': '2024-01-01 09:30:00',
            ...         'bar_end': '2024-01-01 10:15:00',
            ...         'open': 450.0, 'high': 451.0, 'low': 449.5, 'close': 450.5,
            ...         'volume': 100000, 'tick_count': 500, 'imbalance': 0.15
            ...     }
            ... ]
            >>> storage.save_bars('SPY', bars)
        """
        if not bars:
            return 0

        conn = self._get_connection()
        cursor = conn.cursor()

        insert_data = []
        for bar in bars:
            insert_data.append(
                (
                    symbol,
                    to_epoch_ms(bar['bar_start']),
                    to_epoch_ms(bar['bar_end']),
                    bar['open'],
                    bar['high'],
                    bar['low'],
                    bar['close'],
                    bar['volume'],
                    bar['tick_count'],
                    bar.get('imbalance', 0.0)
                )
            )

        try:
            cursor.executemany("""
                INSERT OR REPLACE INTO imbalance_bars
                (symbol, bar_start, bar_end, open, high, low, close, volume, tick_count, imbalance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, insert_data)

            conn.commit()
            saved_count = cursor.rowcount

            logger.info(f"Saved {saved_count} bars for {symbol}")
            return saved_count

        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Error saving bars for {symbol}: {e}")
            raise

    def load_bars(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = None,
        timestamp_format: str = "datetime"
    ) -> pd.DataFrame:
        """
        Load imbalance bars from database.

        Args:
            symbol: Stock ticker
            start: Start timestamp (ISO string, datetime, or epoch ms)
            end: End timestamp (ISO string, datetime, or epoch ms)
            limit: Maximum number of bars to return
            timestamp_format: "datetime", "iso", or "epoch_ms"

        Returns:
            DataFrame indexed by bar_end with OHLCV columns.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = """
            SELECT bar_start, bar_end, open, high, low, close, volume, tick_count, imbalance
            FROM imbalance_bars
            WHERE symbol = ?
        """
        params = [symbol]

        if start:
            query += " AND bar_end >= ?"
            params.append(to_epoch_ms(start))
        if end:
            query += " AND bar_end <= ?"
            params.append(to_epoch_ms(end))

        query += " ORDER BY bar_end ASC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=[
            "bar_start", "bar_end", "open", "high", "low", "close",
            "volume", "tick_count", "imbalance"
        ])

        if timestamp_format == "epoch_ms":
            df.set_index("bar_end", inplace=True)
        elif timestamp_format == "iso":
            df["bar_end"] = df["bar_end"].apply(lambda x: epoch_ms_to_datetime(x).isoformat())
            df.set_index("bar_end", inplace=True)
        elif timestamp_format == "datetime":
            df["bar_end"] = df["bar_end"].apply(epoch_ms_to_datetime)
            df.set_index("bar_end", inplace=True)
        else:
            raise ValueError(f"Unsupported timestamp_format: {timestamp_format}")

        return df

    def close(self):
        """
        Close the database connection.

        Call this when you're completely done with the storage object.
        In practice, Python will close it automatically when the object
        is garbage collected, but explicit is better than implicit!

        Example:
            >>> storage = TickStorage(db_path)
            >>> # ... do work ...
            >>> storage.close()
        """
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.debug("Database connection closed")

    def __del__(self):
        """
        Destructor - called when object is being destroyed.

        This ensures the database connection is closed even if you forget
        to call close() explicitly.

        OOP Concept: __del__ is a "magic method" (also called "dunder method")
        that Python calls automatically when an object is about to be deleted.
        """
        self.close()
