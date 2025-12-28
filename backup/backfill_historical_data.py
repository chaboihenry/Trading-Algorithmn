"""
Historical Data Backfill Script

This script backfills the database with historical market data from January 1, 2020
to the present. It populates:
1. raw_price_data - OHLCV price data
2. technical_indicators - RSI, MACD, Bollinger Bands, etc.
3. volatility_metrics - Historical volatility, ATR, etc.
4. ml_features - Aggregated features for ML models

The script is idempotent - safe to run multiple times without duplicating data.

Usage:
    python backfill_historical_data.py

    Optional flags:
    --start-date YYYY-MM-DD  (default: 2020-01-01)
    --end-date YYYY-MM-DD    (default: today)
    --symbols AAPL,MSFT,...  (default: all symbols in assets table)
"""

import sqlite3
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import argparse
from typing import List, Tuple, Optional
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = '/Volumes/Vault/85_assets_prediction.db'
DEFAULT_START_DATE = '2020-01-01'


class HistoricalDataBackfiller:
    """
    Manages the backfilling of historical market data into the database.

    This class handles:
    - Fetching historical price data from Yahoo Finance
    - Calculating technical indicators
    - Computing volatility metrics
    - Aggregating ML features
    - Inserting data into the database (idempotently)
    """

    def __init__(self, db_path: str = DB_PATH):
        """
        Initialize the backfiller.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.conn = None

    def connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path)
        logger.info(f"Connected to database: {self.db_path}")

    def disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def __enter__(self):
        """Context manager entry - establish connection."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close connection automatically."""
        self.disconnect()
        return False  # Don't suppress exceptions

    def get_symbols(self, custom_symbols: Optional[List[str]] = None) -> List[str]:
        """
        Get list of symbols to backfill.

        Args:
            custom_symbols: Optional list of specific symbols to backfill

        Returns:
            List of ticker symbols
        """
        if custom_symbols:
            return custom_symbols

        # Get all symbols from assets table
        query = "SELECT symbol_ticker FROM assets ORDER BY symbol_ticker"
        df = pd.read_sql_query(query, self.conn)
        symbols = df['symbol_ticker'].tolist()
        logger.info(f"Found {len(symbols)} symbols in assets table")
        return symbols

    def get_existing_date_range(self, symbol: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Check what data already exists for a symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Tuple of (earliest_date, latest_date) or (None, None) if no data
        """
        query = """
        SELECT MIN(price_date) as min_date, MAX(price_date) as max_date
        FROM raw_price_data
        WHERE symbol_ticker = ?
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (symbol,))
        result = cursor.fetchone()

        if result and result[0]:
            return result[0], result[1]
        return None, None

    def fetch_historical_prices(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data from Yahoo Finance.

        Args:
            symbol: Ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV data or None if fetch fails
        """
        try:
            logger.info(f"Fetching {symbol} from {start_date} to {end_date}")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=True)

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None

            # Prepare data for insertion
            df = df.reset_index()
            df['symbol_ticker'] = symbol
            df['price_date'] = df['Date'].dt.strftime('%Y-%m-%d')

            # Rename columns to match database schema
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Select only needed columns
            df = df[['symbol_ticker', 'price_date', 'open', 'high', 'low', 'close', 'volume']]

            logger.info(f"Fetched {len(df)} days of data for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None

    def insert_price_data(self, df: pd.DataFrame) -> int:
        """
        Insert price data into raw_price_data table.

        Args:
            df: DataFrame with price data

        Returns:
            Number of rows inserted
        """
        if df is None or df.empty:
            return 0

        # Create table if it doesn't exist
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS raw_price_data (
            symbol_ticker TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (symbol_ticker, date)
        )
        """
        self.conn.execute(create_table_sql)

        # Insert data (ignore duplicates)
        inserted = 0
        for _, row in df.iterrows():
            try:
                self.conn.execute("""
                    INSERT OR IGNORE INTO raw_price_data
                    (symbol_ticker, price_date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['symbol_ticker'],
                    row['price_date'],
                    row['open'],
                    row['high'],
                    row['low'],
                    row['close'],
                    row['volume']
                ))
                inserted += self.conn.cursor().rowcount
            except sqlite3.IntegrityError:
                # Duplicate entry, skip
                continue

        self.conn.commit()
        return inserted

    def calculate_technical_indicators(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ):
        """
        Calculate and store technical indicators.

        Calculates:
        - RSI (14-period)
        - MACD (12, 26, 9)
        - Bollinger Bands (20-period, 2 std)
        - ATR (14-period)

        Args:
            symbol: Ticker symbol
            start_date: Start date for calculation
            end_date: End date for calculation
        """
        # Fetch price data with extra buffer for indicator calculation
        buffer_start = (pd.to_datetime(start_date) - timedelta(days=100)).strftime('%Y-%m-%d')

        query = """
        SELECT price_date as date, close, high, low, volume
        FROM raw_price_data
        WHERE symbol_ticker = ? AND price_date >= ? AND price_date <= ?
        ORDER BY price_date
        """
        df = pd.read_sql_query(query, self.conn, params=(symbol, buffer_start, end_date))

        if len(df) < 50:  # Need minimum data for indicators
            logger.warning(f"Insufficient data for {symbol} technical indicators")
            return

        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # Calculate MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Calculate Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width']

        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr_14'] = true_range.rolling(14).mean()

        # Filter to requested date range
        df = df[df.index >= start_date]
        df = df.reset_index()
        df['symbol_ticker'] = symbol

        # Insert indicators into existing schema (don't recreate table)
        for _, row in df.iterrows():
            try:
                self.conn.execute("""
                    INSERT OR REPLACE INTO technical_indicators
                    (symbol_ticker, indicator_date, rsi_14, macd, macd_signal, macd_histogram,
                     bb_upper, bb_middle, bb_lower, atr_14)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    row['date'].strftime('%Y-%m-%d'),
                    row['rsi_14'],
                    row['macd'],
                    row['macd_signal'],
                    row['macd_histogram'],
                    row['bb_upper'],
                    row['bb_middle'],
                    row['bb_lower'],
                    row['atr_14']
                ))
            except Exception as e:
                logger.error(f"Error inserting technical indicators for {symbol}: {e}")
                continue

        self.conn.commit()
        logger.info(f"Calculated technical indicators for {symbol}")

    def calculate_volatility_metrics(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ):
        """
        Calculate and store volatility metrics.

        Calculates:
        - Historical volatility (10, 20, 30, 50, 100 day windows)
        - Rolling returns

        Args:
            symbol: Ticker symbol
            start_date: Start date for calculation
            end_date: End date for calculation
        """
        # Fetch price data with buffer
        buffer_start = (pd.to_datetime(start_date) - timedelta(days=150)).strftime('%Y-%m-%d')

        query = """
        SELECT price_date as date, close
        FROM raw_price_data
        WHERE symbol_ticker = ? AND price_date >= ? AND price_date <= ?
        ORDER BY price_date
        """
        df = pd.read_sql_query(query, self.conn, params=(symbol, buffer_start, end_date))

        if len(df) < 100:
            logger.warning(f"Insufficient data for {symbol} volatility metrics")
            return

        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        # Calculate returns
        df['return_1d'] = df['close'].pct_change()
        df['return_5d'] = df['close'].pct_change(5)
        df['return_20d'] = df['close'].pct_change(20)

        # Calculate close-to-close volatility for different windows
        df['close_to_close_vol_10d'] = df['return_1d'].rolling(10).std() * np.sqrt(252)
        df['close_to_close_vol_20d'] = df['return_1d'].rolling(20).std() * np.sqrt(252)
        df['close_to_close_vol_60d'] = df['return_1d'].rolling(60).std() * np.sqrt(252)

        # Filter to requested date range
        df = df[df.index >= start_date]
        df = df.reset_index()
        df['symbol_ticker'] = symbol

        # Insert metrics into existing schema
        for _, row in df.iterrows():
            try:
                self.conn.execute("""
                    INSERT OR REPLACE INTO volatility_metrics
                    (symbol_ticker, vol_date, close_to_close_vol_10d, close_to_close_vol_20d, close_to_close_vol_60d)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    symbol,
                    row['date'].strftime('%Y-%m-%d'),
                    row.get('close_to_close_vol_10d'),
                    row.get('close_to_close_vol_20d'),
                    row.get('close_to_close_vol_60d')
                ))
            except Exception as e:
                logger.error(f"Error inserting volatility metrics for {symbol}: {e}")
                continue

        self.conn.commit()
        logger.info(f"Calculated volatility metrics for {symbol}")

    def backfill_symbol(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> bool:
        """
        Backfill all data for a single symbol.

        Args:
            symbol: Ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check existing data
            existing_start, existing_end = self.get_existing_date_range(symbol)

            if existing_start and existing_end:
                logger.info(f"{symbol}: Existing data from {existing_start} to {existing_end}")

                # Determine what needs to be backfilled
                if existing_start <= start_date and existing_end >= end_date:
                    logger.info(f"{symbol}: Already has complete data, recalculating indicators")
                    # Recalculate indicators for entire range
                    self.calculate_technical_indicators(symbol, start_date, end_date)
                    self.calculate_volatility_metrics(symbol, start_date, end_date)
                    return True

            # Fetch and insert price data
            df = self.fetch_historical_prices(symbol, start_date, end_date)
            if df is not None:
                inserted = self.insert_price_data(df)
                logger.info(f"{symbol}: Inserted {inserted} new price records")

            # Calculate and store indicators
            self.calculate_technical_indicators(symbol, start_date, end_date)
            self.calculate_volatility_metrics(symbol, start_date, end_date)

            # Small delay to avoid rate limiting
            time.sleep(0.5)

            return True

        except Exception as e:
            logger.error(f"Error backfilling {symbol}: {e}")
            return False

    def backfill_all(
        self,
        start_date: str = DEFAULT_START_DATE,
        end_date: Optional[str] = None,
        symbols: Optional[List[str]] = None
    ):
        """
        Backfill data for all symbols.

        Args:
            start_date: Start date (default: 2020-01-01)
            end_date: End date (default: today)
            symbols: Optional list of specific symbols to backfill
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        logger.info("=" * 80)
        logger.info("STARTING HISTORICAL DATA BACKFILL")
        logger.info("=" * 80)
        logger.info(f"Date range: {start_date} to {end_date}")

        self.connect()

        try:
            symbols_to_backfill = self.get_symbols(symbols)
            total = len(symbols_to_backfill)
            successful = 0
            failed = 0

            logger.info(f"Backfilling {total} symbols...")
            logger.info("-" * 80)

            for i, symbol in enumerate(symbols_to_backfill, 1):
                logger.info(f"[{i}/{total}] Processing {symbol}...")

                if self.backfill_symbol(symbol, start_date, end_date):
                    successful += 1
                else:
                    failed += 1

            logger.info("=" * 80)
            logger.info("BACKFILL COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Successful: {successful}")
            logger.info(f"Failed: {failed}")
            logger.info(f"Total: {total}")

        finally:
            self.disconnect()


def main():
    """Main entry point for the backfill script."""
    parser = argparse.ArgumentParser(description='Backfill historical market data')
    parser.add_argument(
        '--start-date',
        type=str,
        default=DEFAULT_START_DATE,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date (YYYY-MM-DD), defaults to today'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        default=None,
        help='Comma-separated list of symbols (defaults to all in assets table)'
    )

    args = parser.parse_args()

    # Parse symbols if provided
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]

    # Run backfill
    backfiller = HistoricalDataBackfiller()
    backfiller.backfill_all(
        start_date=args.start_date,
        end_date=args.end_date,
        symbols=symbols
    )


if __name__ == "__main__":
    main()
