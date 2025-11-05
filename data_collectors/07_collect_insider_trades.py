import sqlite3
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InsiderTradingCollector:
    """
    Collects insider trading data for all stocks

    Database: /Volumes/Vault/85_assets_prediction.db
    Table: insider_trading
    Data Source: Finnhub API
    Time Period: Last 3 months of insider transactions
    """

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db") -> None:
        """Initialize the collector with database path and API keys"""
        self.db_path = db_path

        # Finnhub API key
        self.finnhub_api_key = "d3nuofhr01qmj82v1vdgd3nuofhr01qmj82v1ve0"
        self.finnhub_base_url = "https://finnhub.io/api/v1"

        # Date range for insider trades (last 3 months)
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=90)

        logger.info(f"Initialized InsiderTradingCollector")
        logger.info(f"Date range: {self.start_date.date()} to {self.end_date.date()}")

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

    def collect_insider_transactions(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Collect insider trading transactions from Finnhub

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of insider trading records
        """
        try:
            logger.info(f"Collecting insider trades for {ticker}")

            # Finnhub insider transactions endpoint
            url = f"{self.finnhub_base_url}/stock/insider-transactions"
            params = {
                'symbol': ticker,
                'from': self.start_date.strftime('%Y-%m-%d'),
                'to': self.end_date.strftime('%Y-%m-%d'),
                'token': self.finnhub_api_key
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                logger.warning(f"Finnhub API returned status {response.status_code} for {ticker}")
                return []

            data = response.json()

            if not data or 'data' not in data:
                logger.warning(f"No insider trading data for {ticker}")
                return []

            insider_records = []

            for transaction in data['data']:
                # Determine transaction type
                change = transaction.get('change', 0)
                transaction_type = 'Buy' if change > 0 else ('Sell' if change < 0 else 'Other')

                # Calculate ownership change percentage
                shares_owned_after = transaction.get('share', 0)
                ownership_change_percent = None
                if shares_owned_after > 0 and change != 0:
                    shares_owned_before = shares_owned_after - change
                    if shares_owned_before > 0:
                        ownership_change_percent = (change / shares_owned_before) * 100

                # Parse insider information
                insider_name = transaction.get('name', '')

                # Determine if director or officer based on transaction code or other fields
                # Finnhub doesn't always provide explicit role, so we use heuristics
                is_director = 0
                is_officer = 0

                # Check if name contains common officer/director titles
                name_lower = insider_name.lower()
                if any(title in name_lower for title in ['ceo', 'cfo', 'coo', 'president', 'officer']):
                    is_officer = 1
                elif any(title in name_lower for title in ['director', 'board']):
                    is_director = 1

                record = {
                    'symbol_ticker': ticker,
                    'filing_date': transaction.get('filingDate'),
                    'transaction_date': transaction.get('transactionDate'),
                    'insider_name': insider_name,
                    'insider_title': None,  # Not directly available in Finnhub basic data
                    'is_director': is_director,
                    'is_officer': is_officer,
                    'transaction_type': transaction_type,
                    'shares_traded': abs(change) if change else None,
                    'price_per_share': transaction.get('transactionPrice'),
                    'total_value': abs(change * transaction.get('transactionPrice', 0)) if change and transaction.get('transactionPrice') else None,
                    'shares_owned_after': shares_owned_after,
                    'ownership_change_percent': ownership_change_percent
                }

                insider_records.append(record)

            logger.info(f"Collected {len(insider_records)} insider transactions for {ticker}")
            return insider_records

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error collecting insider trades for {ticker}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error collecting insider trades for {ticker}: {str(e)}")
            return []

    def collect_all_insider_trades(self) -> pd.DataFrame:
        """
        Collect insider trading data for all stock tickers
        """
        logger.info("Starting insider trading collection for all stocks")

        # Get all stock tickers
        tickers = self._get_stock_tickers()

        all_insider_trades = []
        success_count = 0
        failed_tickers = []

        for idx, ticker in enumerate(tickers, 1):
            try:
                # Collect insider transactions
                insider_trades = self.collect_insider_transactions(ticker)

                if insider_trades:
                    all_insider_trades.extend(insider_trades)
                    success_count += 1

                # Log progress every 10 tickers
                if idx % 10 == 0:
                    logger.info(f"Progress: {idx}/{len(tickers)} stocks processed ({success_count} with data)")

                # Rate limiting for Finnhub API
                time.sleep(1)

            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                failed_tickers.append(ticker)

        # Convert to DataFrame
        if all_insider_trades:
            insider_df = pd.DataFrame(all_insider_trades)
            insider_df = insider_df.sort_values(['symbol_ticker', 'transaction_date'])

            logger.info(f"\n{'='*60}")
            logger.info(f"Collection Summary:")
            logger.info(f"  Total stocks: {len(tickers)}")
            logger.info(f"  Stocks with insider trades: {success_count}")
            logger.info(f"  Failed: {len(failed_tickers)}")
            if failed_tickers:
                logger.info(f"  Failed tickers: {', '.join(failed_tickers[:10])}")
            logger.info(f"  Total insider transactions: {len(insider_df)}")
            logger.info(f"{'='*60}\n")

            return insider_df
        else:
            logger.warning("No insider trading data collected")
            return pd.DataFrame()

    def populate_insider_trading_table(self, replace: bool = False) -> None:
        """
        Populate the insider_trading table in the database

        Args:
            replace: If True, replace existing data. If False, append only new data.
        """
        logger.info("Starting insider_trading table population")

        try:
            # Collect all insider trading data
            insider_df = self.collect_all_insider_trades()

            if insider_df.empty:
                logger.error("No insider trading data collected. Aborting database insertion.")
                return

            # Connect to database
            conn = self._get_db_connection()
            cursor = conn.cursor()

            # Check existing data
            cursor.execute("SELECT COUNT(*) FROM insider_trading")
            existing_count = cursor.fetchone()[0]

            if existing_count > 0 and not replace:
                logger.info(f"Found {existing_count} existing records. Appending new data only.")

                # Get existing ticker-date combinations
                existing_df = pd.read_sql(
                    "SELECT symbol_ticker, transaction_date, insider_name FROM insider_trading",
                    conn
                )
                existing_keys = set(
                    zip(existing_df['symbol_ticker'], existing_df['transaction_date'], existing_df['insider_name'])
                )

                # Filter to only new records
                new_records_mask = ~insider_df.apply(
                    lambda row: (row['symbol_ticker'], row['transaction_date'], row['insider_name']) in existing_keys,
                    axis=1
                )
                new_insider_df = insider_df[new_records_mask]

                if len(new_insider_df) > 0:
                    new_insider_df.to_sql('insider_trading', conn, if_exists='append', index=False)
                    logger.info(f"Inserted {len(new_insider_df)} new records")
                else:
                    logger.info("No new records to insert")
            else:
                # Replace or insert all
                if_exists = 'replace' if replace else 'append'
                insider_df.to_sql('insider_trading', conn, if_exists=if_exists, index=False)
                logger.info(f"{'Replaced with' if replace else 'Inserted'} {len(insider_df)} records")

            # Verify insertion and show summary
            cursor.execute("SELECT COUNT(*) FROM insider_trading")
            final_count = cursor.fetchone()[0]
            logger.info(f"insider_trading table now contains {final_count} total records")

            # Show summary statistics
            summary_df = pd.read_sql("""
                SELECT
                    symbol_ticker,
                    COUNT(*) as transaction_count,
                    SUM(CASE WHEN transaction_type = 'Buy' THEN 1 ELSE 0 END) as buys,
                    SUM(CASE WHEN transaction_type = 'Sell' THEN 1 ELSE 0 END) as sells,
                    SUM(CASE WHEN transaction_type = 'Buy' THEN shares_traded ELSE 0 END) as total_shares_bought,
                    SUM(CASE WHEN transaction_type = 'Sell' THEN shares_traded ELSE 0 END) as total_shares_sold
                FROM insider_trading
                GROUP BY symbol_ticker
                ORDER BY transaction_count DESC
                LIMIT 15
            """, conn)

            logger.info(f"\nInsider Trading Summary (top 15 stocks by activity):")
            logger.info(f"\n{summary_df.to_string(index=False)}")

            # Recent transactions
            recent_df = pd.read_sql("""
                SELECT
                    symbol_ticker,
                    transaction_date,
                    insider_name,
                    transaction_type,
                    CAST(shares_traded AS INTEGER) as shares,
                    ROUND(price_per_share, 2) as price,
                    CAST(total_value AS INTEGER) as value
                FROM insider_trading
                ORDER BY transaction_date DESC
                LIMIT 15
            """, conn)

            logger.info(f"\nRecent Insider Transactions (last 15):")
            logger.info(f"\n{recent_df.to_string(index=False)}")

            # Net insider activity
            net_activity_df = pd.read_sql("""
                SELECT
                    symbol_ticker,
                    SUM(CASE WHEN transaction_type = 'Buy' THEN shares_traded ELSE -shares_traded END) as net_shares,
                    SUM(CASE WHEN transaction_type = 'Buy' THEN total_value ELSE -total_value END) as net_value
                FROM insider_trading
                WHERE transaction_type IN ('Buy', 'Sell')
                GROUP BY symbol_ticker
                HAVING ABS(net_shares) > 0
                ORDER BY net_value DESC
                LIMIT 10
            """, conn)

            logger.info(f"\nNet Insider Activity (top 10 by value):")
            logger.info(f"\n{net_activity_df.to_string(index=False)}")

            conn.close()
            logger.info("Successfully populated insider_trading table")

        except Exception as e:
            logger.error(f"Error populating insider_trading table: {str(e)}")
            raise


if __name__ == "__main__":
    # Initialize collector
    collector = InsiderTradingCollector()

    print(f"\n{'='*60}")
    print(f"Insider Trading Data Collection Script")
    print(f"{'='*60}")
    print(f"Database: {collector.db_path}")
    print(f"Date Range: {collector.start_date.date()} to {collector.end_date.date()}")
    print(f"Data Source: Finnhub API")

    # Populate database
    print(f"\n{'='*60}")
    print("Collecting insider trading data for all stocks...")
    print(f"{'='*60}\n")

    collector.populate_insider_trading_table(replace=True)

    print(f"\n{'='*60}")
    print("Insider trading data collection complete!")
    print(f"{'='*60}\n")
