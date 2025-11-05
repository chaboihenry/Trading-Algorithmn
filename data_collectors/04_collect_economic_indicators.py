import sqlite3
import pandas as pd
import yfinance as yf
import requests
from fredapi import Fred
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EconomicIndicatorCollector:
    """
    Collects economic indicators and market data

    Database: /Volumes/Vault/85_assets_prediction.db
    Table: economic_indicators
    Data Sources:
        - yfinance: VIX data, Treasury yields, market indices, dollar index, commodities
        - FRED API (optional): Fed funds rate, inflation, unemployment, GDP, consumer confidence
    Time Period: 3 years (October 2022 - October 2025)
    """

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db") -> None:
        """Initialize the collector with database path and API keys"""
        self.db_path = db_path

        # FRED API key
        self.fred_api_key = "294bbe7f3078eb3b530c16ebad2b4d60"
        self.fred = Fred(api_key=self.fred_api_key)

        # Date range for 3 years of data
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=3*365)  # Approximately 3 years

        logger.info(f"Initialized EconomicIndicatorCollector")
        logger.info(f"Date range: {self.start_date.date()} to {self.end_date.date()}")

    def _get_db_connection(self) -> sqlite3.Connection:
        """Create and return database connection"""
        return sqlite3.connect(self.db_path)

    def collect_vix_data(self) -> pd.DataFrame:
        """Collect VIX and VIX futures data from yfinance"""
        try:
            logger.info("Collecting VIX data from yfinance")

            # VIX spot
            vix = yf.download('^VIX', start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)

            if vix.empty:
                logger.error("VIX download returned empty DataFrame")
                return pd.DataFrame()

            # Handle MultiIndex columns
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)

            vix_df = pd.DataFrame({
                'Date': vix.index,
                'vix': vix['Close'].values if 'Close' in vix.columns else vix.iloc[:, 0].values
            })

            # VIX futures for term structure (1m, 3m, 6m)
            # Note: These tickers may not have full historical data
            try:
                vix1m = yf.download('^VIX1D', start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)
                if not vix1m.empty:
                    if isinstance(vix1m.columns, pd.MultiIndex):
                        vix1m.columns = vix1m.columns.get_level_values(0)
                    vix_df = vix_df.merge(
                        pd.DataFrame({'Date': vix1m.index, 'vix_1m': vix1m['Close'].values if 'Close' in vix1m.columns else vix1m.iloc[:, 0].values}),
                        on='Date', how='left'
                    )
            except Exception as e:
                logger.warning(f"Could not fetch VIX1M data: {str(e)}")

            try:
                vix3m = yf.download('^VIX3M', start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)
                if not vix3m.empty:
                    if isinstance(vix3m.columns, pd.MultiIndex):
                        vix3m.columns = vix3m.columns.get_level_values(0)
                    vix_df = vix_df.merge(
                        pd.DataFrame({'Date': vix3m.index, 'vix_3m': vix3m['Close'].values if 'Close' in vix3m.columns else vix3m.iloc[:, 0].values}),
                        on='Date', how='left'
                    )
            except Exception as e:
                logger.warning(f"Could not fetch VIX3M data: {str(e)}")

            try:
                vix6m = yf.download('^VIX6M', start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)
                if not vix6m.empty:
                    if isinstance(vix6m.columns, pd.MultiIndex):
                        vix6m.columns = vix6m.columns.get_level_values(0)
                    vix_df = vix_df.merge(
                        pd.DataFrame({'Date': vix6m.index, 'vix_6m': vix6m['Close'].values if 'Close' in vix6m.columns else vix6m.iloc[:, 0].values}),
                        on='Date', how='left'
                    )
            except Exception as e:
                logger.warning(f"Could not fetch VIX6M data: {str(e)}")

            # Calculate VIX change
            vix_df['vix_change'] = vix_df['vix'].pct_change(fill_method=None)

            logger.info(f"Collected {len(vix_df)} rows of VIX data")
            return vix_df

        except Exception as e:
            logger.error(f"Error collecting VIX data: {str(e)}")
            return pd.DataFrame()

    def collect_treasury_data(self) -> pd.DataFrame:
        """Collect Treasury yield data from yfinance (more reliable than FRED API)"""
        try:
            logger.info("Collecting Treasury yield data from yfinance")

            # Treasury yield tickers from yfinance
            treasury_symbols = {
                '^TNX': 'treasury_10y',    # 10-Year Treasury Yield
                '^FVX': 'treasury_5y',     # 5-Year Treasury Yield
                '^IRX': 'treasury_3m'      # 13-Week Treasury Bill
            }

            treasury_data = {}
            for symbol, column in treasury_symbols.items():
                try:
                    logger.info(f"Fetching {symbol} ({column}) from yfinance...")
                    data = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)

                    if not data.empty:
                        # Handle MultiIndex columns
                        if isinstance(data.columns, pd.MultiIndex):
                            data.columns = data.columns.get_level_values(0)

                        # Use Close price as yield value
                        treasury_data[column] = data['Close']
                        logger.info(f"✓ Fetched {symbol}: {len(data)} records")
                    else:
                        logger.warning(f"✗ No data returned for {symbol}")

                    time.sleep(0.5)
                except Exception as e:
                    logger.warning(f"✗ Could not fetch {symbol}: {str(e)}")

            if not treasury_data:
                logger.error("No Treasury data collected")
                return pd.DataFrame()

            treasury_df = pd.DataFrame(treasury_data)
            treasury_df.index.name = 'Date'
            treasury_df = treasury_df.reset_index()

            # Add 2-year approximation (interpolate between 3m and 5y)
            if 'treasury_5y' in treasury_df.columns and 'treasury_3m' in treasury_df.columns:
                # 2-year is closer to 5-year, so weight it 70% 5y, 30% 3m
                treasury_df['treasury_2y'] = (0.7 * treasury_df['treasury_5y'] + 0.3 * treasury_df['treasury_3m'])

            # Calculate yield curve spreads
            if 'treasury_10y' in treasury_df.columns and 'treasury_2y' in treasury_df.columns:
                treasury_df['yield_curve_10y_2y'] = treasury_df['treasury_10y'] - treasury_df['treasury_2y']

            if 'treasury_10y' in treasury_df.columns and 'treasury_3m' in treasury_df.columns:
                treasury_df['yield_curve_10y_3m'] = treasury_df['treasury_10y'] - treasury_df['treasury_3m']

            logger.info(f"Collected {len(treasury_df)} rows of Treasury data")
            return treasury_df

        except Exception as e:
            logger.error(f"Error collecting Treasury data: {str(e)}")
            return pd.DataFrame()

    def collect_market_data(self) -> pd.DataFrame:
        """Collect market indices and commodities from yfinance"""
        try:
            logger.info("Collecting market data from yfinance")

            market_symbols = {
                'DX-Y.NYB': 'dollar_index',
                'GC=F': 'gold_price',
                'CL=F': 'oil_price',
                'HG=F': 'copper_price',
                '^GSPC': 'sp500_close',
                '^IXIC': 'nasdaq_close',
                '^DJI': 'dow_jones_close',
                '^RUT': 'russell_2000_close'
            }

            market_data = {}
            sp500_volume = None

            for symbol, column in market_symbols.items():
                try:
                    data = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)
                    if not data.empty:
                        # Handle MultiIndex columns
                        if isinstance(data.columns, pd.MultiIndex):
                            data.columns = data.columns.get_level_values(0)

                        market_data[column] = data['Close'] if 'Close' in data.columns else data.iloc[:, 0]

                        # Capture SP500 volume separately
                        if symbol == '^GSPC' and 'Volume' in data.columns:
                            sp500_volume = data['Volume']

                    logger.info(f"Fetched {symbol}: {len(data)} records")
                    time.sleep(0.5)  # Rate limiting
                except Exception as e:
                    logger.warning(f"Could not fetch {symbol}: {str(e)}")

            if not market_data:
                return pd.DataFrame()

            market_df = pd.DataFrame(market_data)

            # Add SP500 volume if available
            if sp500_volume is not None:
                market_df['sp500_volume'] = sp500_volume

            market_df.index.name = 'Date'
            market_df = market_df.reset_index()

            # Calculate dollar index change
            if 'dollar_index' in market_df.columns:
                market_df['dollar_index_change'] = market_df['dollar_index'].pct_change(fill_method=None)

            logger.info(f"Collected {len(market_df)} rows of market data")
            return market_df

        except Exception as e:
            logger.error(f"Error collecting market data: {str(e)}")
            return pd.DataFrame()

    def collect_fred_indicators(self) -> pd.DataFrame:
        """Collect economic indicators from FRED using direct JSON API (optional - won't block if fails)"""
        try:
            logger.info("Collecting economic indicators from FRED (direct JSON API) - Optional")

            fred_symbols = {
                'FEDFUNDS': 'fed_funds_rate',
                'CPIAUCSL': 'inflation_rate',
                'UNRATE': 'unemployment_rate',
                'GDP': 'gdp_growth',
                'UMCSENT': 'consumer_confidence'
            }

            fred_data = {}
            for symbol, column in fred_symbols.items():
                max_retries = 2  # Reduced retries to avoid long waits
                for attempt in range(max_retries):
                    try:
                        logger.info(f"Fetching {symbol} from FRED API...")

                        # Use direct FRED API with JSON format (avoids XML parsing issues)
                        url = "https://api.stlouisfed.org/fred/series/observations"
                        params = {
                            'series_id': symbol,
                            'api_key': self.fred_api_key,
                            'file_type': 'json',
                            'observation_start': self.start_date.strftime('%Y-%m-%d'),
                            'observation_end': self.end_date.strftime('%Y-%m-%d')
                        }

                        response = requests.get(url, params=params, timeout=45)  # Reduced timeout

                        if response.status_code != 200:
                            raise Exception(f"API returned status {response.status_code}")

                        data_json = response.json()

                        if 'observations' not in data_json:
                            raise Exception("No observations in response")

                        # Parse observations
                        observations = data_json['observations']
                        dates = []
                        values = []

                        for obs in observations:
                            if obs['value'] != '.':  # FRED uses '.' for missing values
                                try:
                                    dates.append(pd.to_datetime(obs['date']))
                                    values.append(float(obs['value']))
                                except (ValueError, KeyError):
                                    continue

                        if len(dates) > 0:
                            series = pd.Series(values, index=dates)
                            fred_data[column] = series
                            logger.info(f"✓ Fetched {symbol}: {len(series)} records")
                        else:
                            logger.warning(f"✗ No valid data for {symbol}")

                        time.sleep(0.6)  # FRED rate limit: 120 calls/min
                        break  # Success, exit retry loop

                    except requests.exceptions.Timeout:
                        if attempt < max_retries - 1:
                            logger.warning(f"✗ Timeout for {symbol} (attempt {attempt + 1}/{max_retries})")
                            logger.info(f"Retrying in 2 seconds...")
                            time.sleep(2)
                        else:
                            logger.warning(f"✗ Failed to fetch {symbol} after {max_retries} attempts due to timeout. Skipping (optional data).")
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"✗ Error for {symbol} (attempt {attempt + 1}/{max_retries}): {str(e)}")
                            logger.info(f"Retrying in 2 seconds...")
                            time.sleep(2)
                        else:
                            logger.warning(f"✗ Failed to fetch {symbol} after {max_retries} attempts. Skipping (optional data).")

            if not fred_data:
                logger.warning("No FRED indicators collected - continuing without macro economic data (optional)")
                return pd.DataFrame()

            fred_df = pd.DataFrame(fred_data)
            fred_df.index.name = 'Date'
            fred_df = fred_df.reset_index()

            logger.info(f"✓ Collected {len(fred_df)} rows of FRED indicators")
            return fred_df

        except Exception as e:
            logger.warning(f"FRED indicators collection failed (optional): {str(e)}")
            return pd.DataFrame()

    def combine_all_indicators(self) -> pd.DataFrame:
        """Combine all economic indicators into a single DataFrame"""
        logger.info("Combining all economic indicators")

        # Collect all data sources
        vix_df = self.collect_vix_data()
        treasury_df = self.collect_treasury_data()
        market_df = self.collect_market_data()
        fred_df = self.collect_fred_indicators()

        # Start with VIX data (daily frequency)
        if vix_df.empty:
            logger.error("No VIX data collected, cannot proceed")
            return pd.DataFrame()

        combined_df = vix_df.copy()
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])

        # Merge treasury data
        if not treasury_df.empty:
            treasury_df['Date'] = pd.to_datetime(treasury_df['Date'])
            combined_df = combined_df.merge(treasury_df, on='Date', how='left')

        # Merge market data
        if not market_df.empty:
            market_df['Date'] = pd.to_datetime(market_df['Date'])
            combined_df = combined_df.merge(market_df, on='Date', how='left')

        # Merge FRED indicators (these are often monthly, so forward fill)
        if not fred_df.empty:
            fred_df['Date'] = pd.to_datetime(fred_df['Date'])
            combined_df = combined_df.merge(fred_df, on='Date', how='left')

        # Forward fill FRED indicators (they update monthly/quarterly)
        fred_columns = ['fed_funds_rate', 'inflation_rate', 'unemployment_rate', 'gdp_growth', 'consumer_confidence']
        for col in fred_columns:
            if col in combined_df.columns:
                combined_df[col] = combined_df[col].ffill()

        # Rename Date to indicator_date
        combined_df = combined_df.rename(columns={'Date': 'indicator_date'})
        combined_df['indicator_date'] = combined_df['indicator_date'].dt.strftime('%Y-%m-%d')

        # Sort by date
        combined_df = combined_df.sort_values('indicator_date')

        logger.info(f"Combined data: {len(combined_df)} rows, {len(combined_df.columns)} columns")
        return combined_df

    def populate_economic_indicators_table(self, replace: bool = False) -> None:
        """
        Populate the economic_indicators table in the database

        Args:
            replace: If True, replace existing data. If False, append only new data.
        """
        logger.info("Starting economic_indicators table population")

        try:
            # Collect all economic indicators
            indicators_df = self.combine_all_indicators()

            if indicators_df.empty:
                logger.error("No economic indicator data collected. Aborting database insertion.")
                return

            # Connect to database
            conn = self._get_db_connection()
            cursor = conn.cursor()

            # Check existing data
            cursor.execute("SELECT COUNT(*) FROM economic_indicators")
            existing_count = cursor.fetchone()[0]

            if existing_count > 0 and not replace:
                logger.info(f"Found {existing_count} existing records. Appending new data only.")

                # Get existing dates
                existing_df = pd.read_sql("SELECT indicator_date FROM economic_indicators", conn)
                existing_dates = set(existing_df['indicator_date'])

                # Filter to only new records
                new_indicators_df = indicators_df[~indicators_df['indicator_date'].isin(existing_dates)]

                if len(new_indicators_df) > 0:
                    new_indicators_df.to_sql('economic_indicators', conn, if_exists='append', index=False)
                    logger.info(f"Inserted {len(new_indicators_df)} new records")
                else:
                    logger.info("No new records to insert")
            else:
                # Replace or insert all
                if_exists = 'replace' if replace else 'append'
                indicators_df.to_sql('economic_indicators', conn, if_exists=if_exists, index=False)
                logger.info(f"{'Replaced with' if replace else 'Inserted'} {len(indicators_df)} records")

            # Verify insertion and show summary
            cursor.execute("SELECT COUNT(*) FROM economic_indicators")
            final_count = cursor.fetchone()[0]
            logger.info(f"economic_indicators table now contains {final_count} total records")

            # Show summary statistics
            summary_df = pd.read_sql("""
                SELECT
                    COUNT(*) as total_days,
                    MIN(indicator_date) as start_date,
                    MAX(indicator_date) as end_date,
                    ROUND(AVG(vix), 2) as avg_vix,
                    ROUND(AVG(treasury_10y), 2) as avg_10y_yield,
                    ROUND(AVG(sp500_close), 2) as avg_sp500
                FROM economic_indicators
            """, conn)

            logger.info(f"\nOverall Statistics:")
            logger.info(f"\n{summary_df.to_string(index=False)}")

            # Recent data sample
            recent_df = pd.read_sql("""
                SELECT
                    indicator_date,
                    ROUND(vix, 2) as vix,
                    ROUND(treasury_10y, 2) as t10y,
                    ROUND(yield_curve_10y_2y, 2) as yield_curve,
                    ROUND(sp500_close, 2) as sp500,
                    ROUND(fed_funds_rate, 2) as fed_rate
                FROM economic_indicators
                ORDER BY indicator_date DESC
                LIMIT 10
            """, conn)

            logger.info(f"\nRecent Economic Indicators (last 10 days):")
            logger.info(f"\n{recent_df.to_string(index=False)}")

            # Data completeness
            completeness_df = pd.read_sql("""
                SELECT
                    COUNT(*) as total_records,
                    SUM(CASE WHEN vix IS NOT NULL THEN 1 ELSE 0 END) as has_vix,
                    SUM(CASE WHEN treasury_10y IS NOT NULL THEN 1 ELSE 0 END) as has_treasury,
                    SUM(CASE WHEN sp500_close IS NOT NULL THEN 1 ELSE 0 END) as has_sp500,
                    SUM(CASE WHEN fed_funds_rate IS NOT NULL THEN 1 ELSE 0 END) as has_fed_rate,
                    SUM(CASE WHEN unemployment_rate IS NOT NULL THEN 1 ELSE 0 END) as has_unemployment
                FROM economic_indicators
            """, conn)

            logger.info(f"\nData Completeness:")
            logger.info(f"\n{completeness_df.to_string(index=False)}")

            conn.close()
            logger.info("Successfully populated economic_indicators table")

        except Exception as e:
            logger.error(f"Error populating economic_indicators table: {str(e)}")
            raise


if __name__ == "__main__":
    # Initialize collector
    collector = EconomicIndicatorCollector()

    print(f"\n{'='*60}")
    print(f"Economic Indicators Collection Script")
    print(f"{'='*60}")
    print(f"Database: {collector.db_path}")
    print(f"Date Range: {collector.start_date.date()} to {collector.end_date.date()}")
    print(f"Data Sources: FRED API + yfinance")

    # Populate database
    print(f"\n{'='*60}")
    print("Collecting economic indicators...")
    print(f"{'='*60}\n")

    collector.populate_economic_indicators_table(replace=True)

    print(f"\n{'='*60}")
    print("Economic indicators collection complete!")
    print(f"{'='*60}\n")