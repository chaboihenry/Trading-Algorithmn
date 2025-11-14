"""
Enhanced Price Data Collector
==============================
Uses Yahoo Finance as primary source with intelligent rate limiting to avoid data gaps

Key Improvements:
- Yahoo Finance primary (allows frequent data collection)
- Intelligent rate limiting (60 calls/minute)
- Batch processing with progress tracking
- Polygon.io fallback for missing data
- Consistent lookback windows via DataConfig
"""

import sqlite3
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Optional
import logging
import time
import sys
from pathlib import Path

# Add data_collectors to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from data_infrastructure import DataConfig, get_rate_limit_manager, get_data_quality_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedPriceDataCollector:
    """
    Enhanced price data collector with Yahoo Finance primary source

    Features:
    - Yahoo Finance as primary (higher rate limits, better for frequent collection)
    - Intelligent rate limiting (prevents API violations)
    - Polygon.io fallback for missing data
    - Consistent date ranges via DataConfig
    - Batch processing with progress tracking
    """

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db",
                 use_yfinance_primary: bool = True) -> None:
        """
        Initialize the collector

        Args:
            db_path: Path to database
            use_yfinance_primary: Use Yahoo Finance as primary source (recommended)
        """
        self.db_path = db_path
        self.use_yfinance_primary = use_yfinance_primary

        # API configuration
        self.polygon_api_key = "GqaA97fQfGJTiMc0KX4_kpUhuuhpd5NW"
        self.polygon_base_url = "https://api.polygon.io"

        # Use consistent date ranges from DataConfig
        start_date, end_date = DataConfig.get_date_range('historical')
        self.start_date = start_date
        self.end_date = end_date

        # Get singletons
        self.rate_limiter = get_rate_limit_manager()
        self.data_quality = get_data_quality_manager()

        logger.info(f"Initialized EnhancedPriceDataCollector")
        logger.info(f"Primary source: {'Yahoo Finance' if use_yfinance_primary else 'Polygon.io'}")
        logger.info(f"Date range: {self.start_date.date()} to {self.end_date.date()} ({DataConfig.HISTORICAL_DAYS} days)")

    def _get_db_connection(self) -> sqlite3.Connection:
        """Create and return database connection"""
        return sqlite3.connect(self.db_path)

    def _get_all_tickers(self) -> List[str]:
        """Get all tickers from the assets table"""
        try:
            conn = self._get_db_connection()
            query = "SELECT symbol_ticker FROM assets WHERE symbol_ticker IS NOT NULL ORDER BY symbol_ticker"
            df = pd.read_sql(query, conn)
            conn.close()
            tickers = df['symbol_ticker'].tolist()
            logger.info(f"Retrieved {len(tickers)} tickers from assets table")
            return tickers
        except Exception as e:
            logger.error(f"Error retrieving tickers: {str(e)}")
            raise

    def collect_yfinance(self, ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Collect daily OHLCV data using Yahoo Finance (PRIMARY METHOD)

        Yahoo Finance advantages:
        - Higher rate limits (60 calls/minute vs 5 for Polygon free tier)
        - Better for frequent intraday updates
        - No cost
        - Reliable historical data

        Args:
            ticker: Stock/ETF ticker symbol
            start_date: Start date for data collection
            end_date: End date for data collection

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            # Rate limiting
            self.rate_limiter.wait_if_needed('yfinance')

            logger.info(f"Collecting {ticker} from Yahoo Finance ({start_date.date()} to {end_date.date()})")

            # Download data from yfinance
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)

            if data.empty:
                logger.warning(f"No data returned from Yahoo Finance for {ticker}")
                return None

            # Handle MultiIndex columns (when downloading multiple tickers)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Reset index to get date as column
            data = data.reset_index()

            # Rename columns to match database schema
            data = data.rename(columns={
                'Date': 'price_date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adj_close'
            })

            # Add ticker column
            data['symbol_ticker'] = ticker

            # Convert date to string format
            data['price_date'] = pd.to_datetime(data['price_date']).dt.strftime('%Y-%m-%d')

            # Calculate dollar volume
            data['dollar_volume'] = data['close'] * data['volume']

            # Yahoo Finance doesn't provide trades_count
            data['trades_count'] = None

            # Select columns for database
            columns = [
                'symbol_ticker', 'price_date', 'open', 'high', 'low',
                'close', 'volume', 'adj_close', 'dollar_volume', 'trades_count'
            ]
            data = data[columns]

            # Validate data quality
            is_valid, error_msg = self.data_quality.validate_data_quality(
                data,
                required_cols=['symbol_ticker', 'price_date', 'close', 'volume'],
                min_rows=DataConfig.MINIMUM_DATA_POINTS
            )

            if not is_valid:
                logger.warning(f"{ticker}: Data quality issue - {error_msg}")
                return None

            logger.info(f"✓ {ticker}: Collected {len(data)} rows from Yahoo Finance")
            return data

        except Exception as e:
            logger.error(f"✗ {ticker}: Yahoo Finance error - {str(e)}")
            return None

    def collect_polygon(self, ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Collect daily OHLCV data using Polygon.io (FALLBACK METHOD)

        Args:
            ticker: Stock/ETF ticker symbol
            start_date: Start date for data collection
            end_date: End date for data collection

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            # Rate limiting
            self.rate_limiter.wait_if_needed('polygon')

            logger.info(f"Collecting {ticker} from Polygon.io (fallback)")

            # Polygon.io aggregates endpoint
            url = f"{self.polygon_base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"

            params = {
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000,
                'apiKey': self.polygon_api_key
            }

            response = requests.get(url, params=params, timeout=30)

            if response.status_code != 200:
                logger.warning(f"{ticker}: Polygon API status {response.status_code}")
                return None

            data_json = response.json()

            if data_json.get('status') != 'OK' or not data_json.get('results'):
                logger.warning(f"{ticker}: No data from Polygon")
                return None

            # Convert to DataFrame
            results = data_json['results']
            data = pd.DataFrame(results)

            # Polygon.io column mapping
            data = data.rename(columns={
                't': 'timestamp',
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                'vw': 'vwap',
                'n': 'trades_count'
            })

            # Convert timestamp to date
            data['price_date'] = pd.to_datetime(data['timestamp'], unit='ms').dt.strftime('%Y-%m-%d')

            # Add ticker column
            data['symbol_ticker'] = ticker

            # Calculate dollar volume
            data['dollar_volume'] = data['close'] * data['volume']

            # Use close as adj_close (Polygon aggregates are already adjusted)
            data['adj_close'] = data['close']

            # Select columns for database
            columns = [
                'symbol_ticker', 'price_date', 'open', 'high', 'low',
                'close', 'volume', 'adj_close', 'dollar_volume', 'trades_count'
            ]
            data = data[columns]

            logger.info(f"✓ {ticker}: Collected {len(data)} rows from Polygon.io")
            return data

        except Exception as e:
            logger.error(f"✗ {ticker}: Polygon error - {str(e)}")
            return None

    def collect_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Collect data for a single ticker with primary/fallback logic

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with OHLCV data or None if both sources failed
        """
        # Try primary source
        if self.use_yfinance_primary:
            data = self.collect_yfinance(ticker, self.start_date, self.end_date)
            if data is not None:
                return data
            # Fallback to Polygon
            logger.info(f"{ticker}: Trying Polygon.io fallback")
            return self.collect_polygon(ticker, self.start_date, self.end_date)
        else:
            data = self.collect_polygon(ticker, self.start_date, self.end_date)
            if data is not None:
                return data
            # Fallback to Yahoo Finance
            logger.info(f"{ticker}: Trying Yahoo Finance fallback")
            return self.collect_yfinance(ticker, self.start_date, self.end_date)

    def save_to_database(self, data: pd.DataFrame) -> None:
        """
        Save price data to database

        Args:
            data: DataFrame with price data
        """
        try:
            conn = self._get_db_connection()

            # Create table if not exists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS raw_price_data (
                    price_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol_ticker TEXT NOT NULL,
                    price_date DATE NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    adj_close REAL,
                    dollar_volume REAL,
                    trades_count INTEGER,
                    UNIQUE(symbol_ticker, price_date)
                )
            """)

            # Create index for faster queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_price_ticker_date ON raw_price_data(symbol_ticker, price_date)")

            # Insert or replace data
            data.to_sql('raw_price_data', conn, if_exists='append', index=False, method='multi')

            conn.commit()
            conn.close()

        except sqlite3.IntegrityError:
            # Duplicate entries - use INSERT OR REPLACE instead
            conn = self._get_db_connection()
            for _, row in data.iterrows():
                conn.execute("""
                    INSERT OR REPLACE INTO raw_price_data
                    (symbol_ticker, price_date, open, high, low, close, volume, adj_close, dollar_volume, trades_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, tuple(row))
            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error saving to database: {str(e)}")
            raise

    def collect_all_tickers(self, batch_size: int = 10) -> None:
        """
        Collect price data for all tickers with batch processing

        Args:
            batch_size: Number of tickers to save per batch (for progress tracking)
        """
        tickers = self._get_all_tickers()
        total = len(tickers)

        logger.info(f"\n{'='*60}")
        logger.info(f"COLLECTING PRICE DATA FOR {total} TICKERS")
        logger.info(f"{'='*60}\n")

        start_time = time.time()
        success_count = 0
        failed_tickers = []

        for i, ticker in enumerate(tickers, 1):
            logger.info(f"\n[{i}/{total}] Processing {ticker}...")

            try:
                data = self.collect_ticker_data(ticker)

                if data is not None and not data.empty:
                    self.save_to_database(data)
                    success_count += 1
                    logger.info(f"✓ {ticker}: Saved {len(data)} rows to database")
                else:
                    failed_tickers.append(ticker)
                    logger.warning(f"✗ {ticker}: Failed to collect data")

                # Progress update every batch
                if i % batch_size == 0:
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    remaining = (total - i) / rate if rate > 0 else 0
                    logger.info(f"\nProgress: {i}/{total} ({i/total*100:.1f}%) | "
                               f"Success: {success_count} | Failed: {len(failed_tickers)} | "
                               f"Rate: {rate:.1f} tickers/sec | ETA: {remaining/60:.1f} min\n")

            except Exception as e:
                logger.error(f"✗ {ticker}: Unexpected error - {str(e)}")
                failed_tickers.append(ticker)

        # Final summary
        elapsed = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"COLLECTION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total tickers: {total}")
        logger.info(f"Successful: {success_count} ({success_count/total*100:.1f}%)")
        logger.info(f"Failed: {len(failed_tickers)} ({len(failed_tickers)/total*100:.1f}%)")
        logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")
        logger.info(f"Average rate: {total/elapsed:.1f} tickers/second")

        if failed_tickers:
            logger.warning(f"\nFailed tickers: {', '.join(failed_tickers)}")


def main():
    """Main execution"""
    collector = EnhancedPriceDataCollector(use_yfinance_primary=True)

    # Collect all tickers with batch progress tracking
    collector.collect_all_tickers(batch_size=10)


if __name__ == "__main__":
    main()
