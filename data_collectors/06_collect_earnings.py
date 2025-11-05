import sqlite3
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EarningsDataCollector:
    """
    Collects earnings data for all stocks

    Database: /Volumes/Vault/85_assets_prediction.db
    Table: earnings_data
    Data Sources:
        - yfinance: Historical earnings data
        - Finnhub: Earnings calendar and estimates
    Time Period: Historical earnings + upcoming earnings
    """

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db") -> None:
        """Initialize the collector with database path and API keys"""
        self.db_path = db_path

        # Finnhub API key
        self.finnhub_api_key = "d3nuofhr01qmj82v1vdgd3nuofhr01qmj82v1ve0"
        self.finnhub_base_url = "https://finnhub.io/api/v1"

        logger.info(f"Initialized EarningsDataCollector")

    def _get_db_connection(self) -> sqlite3.Connection:
        """Create and return database connection"""
        return sqlite3.connect(self.db_path)

    def _get_stock_tickers(self) -> List[str]:
        """Get all stock tickers (exclude ETFs) from the assets table"""
        try:
            conn = self._get_db_connection()
            query = """
                SELECT symbol_ticker
                FROM assets
                WHERE asset_type = 'Stock'
                ORDER BY symbol_ticker
            """
            df = pd.read_sql(query, conn)
            conn.close()
            tickers = df['symbol_ticker'].tolist()
            logger.info(f"Retrieved {len(tickers)} stock tickers from assets table")
            return tickers
        except Exception as e:
            logger.error(f"Error retrieving stock tickers: {str(e)}")
            raise

    def collect_yfinance_earnings(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Collect historical earnings data from yfinance

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of earnings records
        """
        try:
            logger.info(f"Collecting yfinance earnings for {ticker}")

            stock = yf.Ticker(ticker)

            # Get quarterly earnings
            earnings_history = stock.quarterly_earnings

            if earnings_history is None or earnings_history.empty:
                logger.warning(f"No earnings history found for {ticker}")
                return []

            earnings_records = []

            for date, row in earnings_history.iterrows():
                # Parse the data
                record = {
                    'symbol_ticker': ticker,
                    'earnings_date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                    'is_confirmed': 1,  # Historical data is confirmed
                    'reported_eps': row.get('Earnings') if pd.notna(row.get('Earnings')) else None,
                    'estimated_eps': None,  # yfinance doesn't provide estimates
                    'eps_surprise': None,
                    'eps_surprise_percent': None,
                    'reported_revenue': row.get('Revenue') if pd.notna(row.get('Revenue')) else None,
                    'estimated_revenue': None,
                    'revenue_surprise': None,
                    'revenue_surprise_percent': None,
                    'guidance_raised': None,
                    'guidance_lowered': None,
                    'guidance_met': None,
                    'beat_count_last_4q': None,
                    'avg_price_move_1d_historical': None
                }

                earnings_records.append(record)

            logger.info(f"Collected {len(earnings_records)} earnings records for {ticker}")
            return earnings_records

        except Exception as e:
            logger.error(f"Error collecting yfinance earnings for {ticker}: {str(e)}")
            return []

    def collect_finnhub_earnings(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Collect earnings estimates and surprises from Finnhub

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of earnings records
        """
        try:
            logger.info(f"Collecting Finnhub earnings for {ticker}")

            # Get earnings calendar
            url = f"{self.finnhub_base_url}/stock/earnings"
            params = {
                'symbol': ticker,
                'token': self.finnhub_api_key
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                logger.warning(f"Finnhub API returned status {response.status_code} for {ticker}")
                return []

            data = response.json()

            if not data:
                logger.warning(f"No Finnhub earnings data for {ticker}")
                return []

            earnings_records = []

            for earnings in data:
                # Calculate surprises
                eps_surprise = None
                eps_surprise_percent = None
                if earnings.get('actual') is not None and earnings.get('estimate') is not None:
                    eps_surprise = earnings['actual'] - earnings['estimate']
                    if earnings['estimate'] != 0:
                        eps_surprise_percent = (eps_surprise / abs(earnings['estimate'])) * 100

                record = {
                    'symbol_ticker': ticker,
                    'earnings_date': earnings.get('period'),
                    'is_confirmed': 1 if earnings.get('actual') is not None else 0,
                    'reported_eps': earnings.get('actual'),
                    'estimated_eps': earnings.get('estimate'),
                    'eps_surprise': eps_surprise,
                    'eps_surprise_percent': eps_surprise_percent,
                    'reported_revenue': None,  # Not in basic earnings endpoint
                    'estimated_revenue': None,
                    'revenue_surprise': None,
                    'revenue_surprise_percent': None,
                    'guidance_raised': None,
                    'guidance_lowered': None,
                    'guidance_met': 1 if eps_surprise is not None and abs(eps_surprise) < 0.01 else None,
                    'beat_count_last_4q': None,  # Will calculate later
                    'avg_price_move_1d_historical': None
                }

                earnings_records.append(record)

            logger.info(f"Collected {len(earnings_records)} Finnhub earnings records for {ticker}")
            return earnings_records

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error collecting Finnhub earnings for {ticker}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error collecting Finnhub earnings for {ticker}: {str(e)}")
            return []

    def merge_earnings_data(self, yf_earnings: List[Dict], finnhub_earnings: List[Dict]) -> pd.DataFrame:
        """
        Merge earnings data from yfinance and Finnhub

        Prioritize Finnhub for estimates/surprises, yfinance for revenue
        """
        all_earnings = []

        # Create dictionaries for faster lookup
        yf_dict = {e['earnings_date']: e for e in yf_earnings}
        finnhub_dict = {e['earnings_date']: e for e in finnhub_earnings}

        # Get all unique dates
        all_dates = set(yf_dict.keys()) | set(finnhub_dict.keys())

        for date in all_dates:
            yf_record = yf_dict.get(date, {})
            finnhub_record = finnhub_dict.get(date, {})

            # Merge records, prioritizing Finnhub for estimates
            merged = {
                'symbol_ticker': yf_record.get('symbol_ticker') or finnhub_record.get('symbol_ticker'),
                'earnings_date': date,
                'is_confirmed': finnhub_record.get('is_confirmed', yf_record.get('is_confirmed', 0)),
                'reported_eps': finnhub_record.get('reported_eps') or yf_record.get('reported_eps'),
                'estimated_eps': finnhub_record.get('estimated_eps'),
                'eps_surprise': finnhub_record.get('eps_surprise'),
                'eps_surprise_percent': finnhub_record.get('eps_surprise_percent'),
                'reported_revenue': yf_record.get('reported_revenue'),
                'estimated_revenue': yf_record.get('estimated_revenue'),
                'revenue_surprise': yf_record.get('revenue_surprise'),
                'revenue_surprise_percent': yf_record.get('revenue_surprise_percent'),
                'guidance_raised': finnhub_record.get('guidance_raised'),
                'guidance_lowered': finnhub_record.get('guidance_lowered'),
                'guidance_met': finnhub_record.get('guidance_met'),
                'beat_count_last_4q': None,
                'avg_price_move_1d_historical': None
            }

            all_earnings.append(merged)

        return pd.DataFrame(all_earnings)

    def collect_all_earnings(self) -> pd.DataFrame:
        """
        Collect earnings data for all stock tickers
        """
        logger.info("Starting earnings collection for all stocks")

        # Get all stock tickers
        tickers = self._get_stock_tickers()

        all_earnings = []
        success_count = 0
        failed_tickers = []

        for idx, ticker in enumerate(tickers, 1):
            try:
                # Collect from both sources
                yf_earnings = self.collect_yfinance_earnings(ticker)
                time.sleep(0.5)  # Rate limiting

                finnhub_earnings = self.collect_finnhub_earnings(ticker)
                time.sleep(1)  # Finnhub rate limiting

                # Merge earnings data
                if yf_earnings or finnhub_earnings:
                    merged_earnings = self.merge_earnings_data(yf_earnings, finnhub_earnings)
                    if not merged_earnings.empty:
                        all_earnings.append(merged_earnings)
                        success_count += 1
                else:
                    failed_tickers.append(ticker)

                # Log progress every 10 tickers
                if idx % 10 == 0:
                    logger.info(f"Progress: {idx}/{len(tickers)} stocks processed ({success_count} successful)")

            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                failed_tickers.append(ticker)

        # Combine all earnings data
        if all_earnings:
            earnings_df = pd.concat(all_earnings, axis=0, ignore_index=True)
            earnings_df = earnings_df.sort_values(['symbol_ticker', 'earnings_date'])

            logger.info(f"\n{'='*60}")
            logger.info(f"Collection Summary:")
            logger.info(f"  Total stocks: {len(tickers)}")
            logger.info(f"  Successful: {success_count}")
            logger.info(f"  Failed: {len(failed_tickers)}")
            if failed_tickers:
                logger.info(f"  Failed tickers: {', '.join(failed_tickers[:10])}")
            logger.info(f"  Total earnings records: {len(earnings_df)}")
            logger.info(f"{'='*60}\n")

            return earnings_df
        else:
            logger.warning("No earnings data collected")
            return pd.DataFrame()

    def populate_earnings_table(self, replace: bool = False) -> None:
        """
        Populate the earnings_data table in the database

        Args:
            replace: If True, replace existing data. If False, append only new data.
        """
        logger.info("Starting earnings_data table population")

        try:
            # Collect all earnings data
            earnings_df = self.collect_all_earnings()

            if earnings_df.empty:
                logger.error("No earnings data collected. Aborting database insertion.")
                return

            # Connect to database
            conn = self._get_db_connection()
            cursor = conn.cursor()

            # Check existing data
            cursor.execute("SELECT COUNT(*) FROM earnings_data")
            existing_count = cursor.fetchone()[0]

            if existing_count > 0 and not replace:
                logger.info(f"Found {existing_count} existing records. Appending new data only.")

                # Get existing ticker-date combinations
                existing_df = pd.read_sql(
                    "SELECT symbol_ticker, earnings_date FROM earnings_data",
                    conn
                )
                existing_keys = set(
                    zip(existing_df['symbol_ticker'], existing_df['earnings_date'])
                )

                # Filter to only new records
                new_records_mask = ~earnings_df.apply(
                    lambda row: (row['symbol_ticker'], row['earnings_date']) in existing_keys,
                    axis=1
                )
                new_earnings_df = earnings_df[new_records_mask]

                if len(new_earnings_df) > 0:
                    new_earnings_df.to_sql('earnings_data', conn, if_exists='append', index=False)
                    logger.info(f"Inserted {len(new_earnings_df)} new records")
                else:
                    logger.info("No new records to insert")
            else:
                # Replace or insert all
                if_exists = 'replace' if replace else 'append'
                earnings_df.to_sql('earnings_data', conn, if_exists=if_exists, index=False)
                logger.info(f"{'Replaced with' if replace else 'Inserted'} {len(earnings_df)} records")

            # Verify insertion and show summary
            cursor.execute("SELECT COUNT(*) FROM earnings_data")
            final_count = cursor.fetchone()[0]
            logger.info(f"earnings_data table now contains {final_count} total records")

            # Show summary statistics
            summary_df = pd.read_sql("""
                SELECT
                    symbol_ticker,
                    COUNT(*) as earnings_count,
                    MIN(earnings_date) as earliest_earnings,
                    MAX(earnings_date) as latest_earnings,
                    AVG(CASE WHEN eps_surprise_percent IS NOT NULL THEN eps_surprise_percent END) as avg_surprise_pct
                FROM earnings_data
                WHERE is_confirmed = 1
                GROUP BY symbol_ticker
                ORDER BY symbol_ticker
                LIMIT 15
            """, conn)

            logger.info(f"\nEarnings Summary (first 15 stocks):")
            logger.info(f"\n{summary_df.to_string(index=False)}")

            # Recent earnings
            recent_df = pd.read_sql("""
                SELECT
                    symbol_ticker,
                    earnings_date,
                    ROUND(reported_eps, 2) as eps,
                    ROUND(estimated_eps, 2) as est_eps,
                    ROUND(eps_surprise_percent, 2) as surprise_pct
                FROM earnings_data
                WHERE is_confirmed = 1 AND reported_eps IS NOT NULL
                ORDER BY earnings_date DESC
                LIMIT 15
            """, conn)

            logger.info(f"\nRecent Earnings (last 15 confirmed):")
            logger.info(f"\n{recent_df.to_string(index=False)}")

            conn.close()
            logger.info("Successfully populated earnings_data table")

        except Exception as e:
            logger.error(f"Error populating earnings_data table: {str(e)}")
            raise


if __name__ == "__main__":
    # Initialize collector
    collector = EarningsDataCollector()

    print(f"\n{'='*60}")
    print(f"Earnings Data Collection Script")
    print(f"{'='*60}")
    print(f"Database: {collector.db_path}")
    print(f"Data Sources: yfinance + Finnhub")

    # Populate database
    print(f"\n{'='*60}")
    print("Collecting earnings data for all stocks...")
    print(f"{'='*60}\n")

    collector.populate_earnings_table(replace=True)

    print(f"\n{'='*60}")
    print("Earnings data collection complete!")
    print(f"{'='*60}\n")
