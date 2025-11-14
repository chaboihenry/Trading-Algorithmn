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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PriceDataCollector:
    """
    Collects daily OHLCV price data for all 85 assets

    Database: /Volumes/Vault/85_assets_prediction.db
    Table: raw_price_data
    Data Source: Yahoo Finance (primary) with Polygon.io fallback
    Time Period: Consistent with DataConfig (2 years for historical data)

    IMPROVEMENTS:
    - Yahoo Finance primary (60 calls/min vs Polygon 5 calls/min)
    - Intelligent rate limiting via RateLimitManager
    - Data quality validation
    - Consistent lookback windows via DataConfig
    """

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db",
                 use_yfinance_primary: bool = True) -> None:
        """Initialize the collector with database path"""
        self.db_path = db_path
        self.use_yfinance_primary = use_yfinance_primary

        # Polygon.io API key
        self.polygon_api_key = "GqaA97fQfGJTiMc0KX4_kpUhuuhpd5NW"
        self.polygon_base_url = "https://api.polygon.io"

        # Use consistent date ranges from DataConfig
        start_date, end_date = DataConfig.get_date_range('historical')
        self.start_date = start_date
        self.end_date = end_date

        # Get singletons
        self.rate_limiter = get_rate_limit_manager()
        self.data_quality = get_data_quality_manager()

        logger.info(f"Initialized PriceDataCollector")
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

    def collect_daily_ohlcv(self, ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Collect daily OHLCV data for a single ticker using Polygon.io

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

            logger.info(f"Collecting data for {ticker} from Polygon.io")

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
                logger.warning(f"Polygon API returned status {response.status_code} for {ticker}")
                return None

            data_json = response.json()

            if data_json.get('status') != 'OK' or not data_json.get('results'):
                logger.warning(f"No data returned from Polygon for {ticker}")
                return None

            # Convert to DataFrame
            results = data_json['results']
            data = pd.DataFrame(results)

            # Polygon.io column mapping
            # t = timestamp (ms), o = open, h = high, l = low, c = close, v = volume, vw = volume weighted, n = trades
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

            # Polygon doesn't provide adj_close directly in aggregates, use close
            data['adj_close'] = data['close']

            # Select only the columns needed for database
            columns = [
                'symbol_ticker', 'price_date', 'open', 'high', 'low',
                'close', 'volume', 'adj_close', 'dollar_volume', 'trades_count'
            ]
            data = data[columns]

            logger.info(f"Collected {len(data)} rows for {ticker} from Polygon.io")
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error collecting data for {ticker}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error collecting data for {ticker}: {str(e)}")
            return None

    def collect_daily_ohlcv_yfinance(self, ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Collect daily OHLCV data using yfinance (primary/fallback method)

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

            logger.info(f"Collecting data for {ticker} from Yahoo Finance")

            # Download data from yfinance
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)

            if data.empty:
                logger.warning(f"No data returned from yfinance for {ticker}")
                return None

            # Handle MultiIndex columns
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

            # yfinance doesn't provide trades_count, set to None
            data['trades_count'] = None

            # Select only the columns needed for database
            columns = [
                'symbol_ticker', 'price_date', 'open', 'high', 'low',
                'close', 'volume', 'adj_close', 'dollar_volume', 'trades_count'
            ]
            data = data[columns]

            logger.info(f"Collected {len(data)} rows for {ticker} from yfinance")
            return data

        except Exception as e:
            logger.error(f"Error collecting data for {ticker} from yfinance: {str(e)}")
            return None

    def collect_all_tickers(self, batch_size: int = 10) -> pd.DataFrame:
        """
        Collect price data for all tickers from the assets table

        Args:
            batch_size: Number of tickers to process before logging progress

        Returns:
            Combined DataFrame with all price data
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"COLLECTING PRICE DATA FOR ALL TICKERS")
        logger.info(f"Primary source: {'Yahoo Finance' if self.use_yfinance_primary else 'Polygon.io'}")
        logger.info(f"{'='*60}\n")

        # Get all tickers from database
        tickers = self._get_all_tickers()

        all_data = []
        success_count = 0
        failed_tickers = []
        polygon_count = 0
        yfinance_count = 0
        start_time = time.time()

        for idx, ticker in enumerate(tickers, 1):
            logger.info(f"\n[{idx}/{len(tickers)}] Processing {ticker}...")

            # Try primary source first
            if self.use_yfinance_primary:
                ticker_data = self.collect_daily_ohlcv_yfinance(ticker, self.start_date, self.end_date)
                if ticker_data is not None and not ticker_data.empty:
                    # Validate data quality
                    is_valid, error_msg = self.data_quality.validate_data_quality(
                        ticker_data,
                        required_cols=['symbol_ticker', 'price_date', 'close', 'volume'],
                        min_rows=DataConfig.MINIMUM_DATA_POINTS
                    )
                    if is_valid:
                        all_data.append(ticker_data)
                        success_count += 1
                        yfinance_count += 1
                        logger.info(f"✓ {ticker}: Collected {len(ticker_data)} rows from Yahoo Finance")
                    else:
                        logger.warning(f"{ticker}: Data quality issue - {error_msg}, trying Polygon fallback")
                        ticker_data = None

                # Fallback to Polygon if Yahoo failed
                if ticker_data is None:
                    logger.info(f"{ticker}: Trying Polygon.io fallback")
                    ticker_data = self.collect_daily_ohlcv(ticker, self.start_date, self.end_date)
                    if ticker_data is not None and not ticker_data.empty:
                        all_data.append(ticker_data)
                        success_count += 1
                        polygon_count += 1
                        logger.info(f"✓ {ticker}: Collected {len(ticker_data)} rows from Polygon.io")
                    else:
                        failed_tickers.append(ticker)
                        logger.warning(f"✗ {ticker}: Failed to collect data")
            else:
                # Polygon primary, Yahoo fallback
                ticker_data = self.collect_daily_ohlcv(ticker, self.start_date, self.end_date)
                if ticker_data is not None and not ticker_data.empty:
                    all_data.append(ticker_data)
                    success_count += 1
                    polygon_count += 1
                    logger.info(f"✓ {ticker}: Collected {len(ticker_data)} rows from Polygon.io")
                else:
                    logger.info(f"{ticker}: Trying Yahoo Finance fallback")
                    ticker_data = self.collect_daily_ohlcv_yfinance(ticker, self.start_date, self.end_date)
                    if ticker_data is not None and not ticker_data.empty:
                        all_data.append(ticker_data)
                        success_count += 1
                        yfinance_count += 1
                        logger.info(f"✓ {ticker}: Collected {len(ticker_data)} rows from Yahoo Finance")
                    else:
                        failed_tickers.append(ticker)
                        logger.warning(f"✗ {ticker}: Failed to collect data")

            # Log progress every batch_size tickers
            if idx % batch_size == 0:
                elapsed = time.time() - start_time
                rate = idx / elapsed if elapsed > 0 else 0
                remaining = (len(tickers) - idx) / rate if rate > 0 else 0
                logger.info(f"\nProgress: {idx}/{len(tickers)} ({idx/len(tickers)*100:.1f}%) | "
                           f"Success: {success_count} | Failed: {len(failed_tickers)} | "
                           f"Rate: {rate:.1f} tickers/sec | ETA: {remaining/60:.1f} min\n")

        # Combine all data
        if not all_data:
            logger.warning("No data collected for any ticker")
            return pd.DataFrame()

        combined_df = pd.concat(all_data, axis=0, ignore_index=True)
        combined_df = combined_df.sort_values(['symbol_ticker', 'price_date'])

        # Final summary
        elapsed = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"COLLECTION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total tickers: {len(tickers)}")
        logger.info(f"Successful: {success_count} ({success_count/len(tickers)*100:.1f}%)")
        logger.info(f"  - Yahoo Finance: {yfinance_count}")
        logger.info(f"  - Polygon.io: {polygon_count}")
        logger.info(f"Failed: {len(failed_tickers)} ({len(failed_tickers)/len(tickers)*100:.1f}%)")
        if failed_tickers:
            logger.info(f"Failed tickers: {', '.join(failed_tickers)}")
        logger.info(f"Total rows collected: {len(combined_df)}")
        logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")
        logger.info(f"Average rate: {len(tickers)/elapsed:.1f} tickers/second")
        logger.info(f"{'='*60}\n")

        return combined_df

    def populate_price_data_table(self, replace: bool = False) -> None:
        """
        Populate the raw_price_data table in the database

        Args:
            replace: If True, replace existing data. If False, append only new data.
        """
        logger.info("Starting raw_price_data table population")

        try:
            # Collect all price data
            price_df = self.collect_all_tickers()

            if price_df.empty:
                logger.error("No price data collected. Aborting database insertion.")
                return

            # Connect to database
            conn = self._get_db_connection()
            cursor = conn.cursor()

            # Check existing data
            cursor.execute("SELECT COUNT(*) FROM raw_price_data")
            existing_count = cursor.fetchone()[0]

            if existing_count > 0 and not replace:
                logger.info(f"Found {existing_count} existing records. Appending new data only.")

                # Get existing ticker-date combinations
                existing_df = pd.read_sql(
                    "SELECT symbol_ticker, price_date FROM raw_price_data",
                    conn
                )
                existing_keys = set(
                    zip(existing_df['symbol_ticker'], existing_df['price_date'])
                )

                # Filter to only new records
                new_records_mask = ~price_df.apply(
                    lambda row: (row['symbol_ticker'], row['price_date']) in existing_keys,
                    axis=1
                )
                new_price_df = price_df[new_records_mask]

                if len(new_price_df) > 0:
                    new_price_df.to_sql('raw_price_data', conn, if_exists='append', index=False)
                    logger.info(f"Inserted {len(new_price_df)} new records")
                else:
                    logger.info("No new records to insert")
            else:
                # Replace or insert all
                if_exists = 'replace' if replace else 'append'
                price_df.to_sql('raw_price_data', conn, if_exists=if_exists, index=False)
                logger.info(f"{'Replaced with' if replace else 'Inserted'} {len(price_df)} records")

            # Verify insertion and show summary
            cursor.execute("SELECT COUNT(*) FROM raw_price_data")
            final_count = cursor.fetchone()[0]
            logger.info(f"raw_price_data table now contains {final_count} total records")

            # Show summary statistics
            summary_df = pd.read_sql("""
                SELECT
                    symbol_ticker,
                    COUNT(*) as record_count,
                    MIN(price_date) as start_date,
                    MAX(price_date) as end_date,
                    ROUND(AVG(close), 2) as avg_close,
                    ROUND(AVG(volume), 0) as avg_volume
                FROM raw_price_data
                GROUP BY symbol_ticker
                ORDER BY symbol_ticker
            """, conn)

            logger.info(f"\nPrice Data Summary (first 10 tickers):")
            logger.info(f"\n{summary_df.head(10).to_string(index=False)}")

            # Overall date range
            date_range_df = pd.read_sql("""
                SELECT
                    MIN(price_date) as earliest_date,
                    MAX(price_date) as latest_date,
                    COUNT(DISTINCT price_date) as unique_dates,
                    COUNT(DISTINCT symbol_ticker) as unique_tickers
                FROM raw_price_data
            """, conn)

            logger.info(f"\nOverall Statistics:")
            logger.info(f"\n{date_range_df.to_string(index=False)}")

            conn.close()
            logger.info("Successfully populated raw_price_data table")

        except Exception as e:
            logger.error(f"Error populating raw_price_data table: {str(e)}")
            raise

    def get_price_data_summary(self) -> pd.DataFrame:
        """Get summary of price data in the database"""
        try:
            conn = self._get_db_connection()
            query = """
                SELECT
                    symbol_ticker,
                    COUNT(*) as days,
                    MIN(price_date) as start_date,
                    MAX(price_date) as end_date
                FROM raw_price_data
                GROUP BY symbol_ticker
                ORDER BY symbol_ticker
            """
            df = pd.read_sql(query, conn)
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Error getting price data summary: {str(e)}")
            raise


if __name__ == "__main__":
    # Initialize collector with Yahoo Finance as primary source (faster, better rate limits)
    collector = PriceDataCollector(use_yfinance_primary=True)

    print(f"\n{'='*60}")
    print(f"Price Data Collection Script")
    print(f"{'='*60}")
    print(f"Database: {collector.db_path}")
    print(f"Date Range: {collector.start_date.date()} to {collector.end_date.date()}")
    print(f"Data Source: yfinance")

    # Populate database
    print(f"\n{'='*60}")
    print("Collecting price data for all assets...")
    print(f"{'='*60}\n")

    collector.populate_price_data_table(replace=True)

    print(f"\n{'='*60}")
    print("Price data collection complete!")
    print(f"{'='*60}\n")