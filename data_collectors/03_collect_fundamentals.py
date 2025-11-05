import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FundamentalDataCollector:
    """
    Collects fundamental data for all stocks (excludes ETFs)

    Database: /Volumes/Vault/85_assets_prediction.db
    Table: fundamental_data
    Data Source: yfinance
    Note: ETFs are excluded as they don't have traditional fundamentals
    """

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db") -> None:
        """Initialize the collector with database path"""
        self.db_path = db_path
        self.collection_date = datetime.now().strftime('%Y-%m-%d')

        logger.info(f"Initialized FundamentalDataCollector")
        logger.info(f"Collection date: {self.collection_date}")

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

    def collect_fundamentals(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Collect fundamental data for a single ticker using yfinance

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with fundamental data or None if failed
        """
        try:
            logger.info(f"Collecting fundamentals for {ticker}")

            # Get stock info from yfinance
            stock = yf.Ticker(ticker)
            info = stock.info

            # Extract fundamental metrics matching database schema
            fundamentals = {
                'symbol_ticker': ticker,
                'fundamental_date': self.collection_date,

                # Valuation metrics
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'price_to_sales': info.get('priceToSalesTrailing12Months'),
                'ev_to_ebitda': info.get('enterpriseToEbitda'),
                'ev_to_revenue': info.get('enterpriseToRevenue'),

                # Profitability metrics
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'return_on_equity': info.get('returnOnEquity'),
                'return_on_assets': info.get('returnOnAssets'),
                'return_on_invested_capital': None,  # Not directly available

                # Financial metrics
                'revenue': info.get('totalRevenue'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'earnings_per_share': info.get('trailingEps'),

                # Balance sheet metrics
                'total_debt': info.get('totalDebt'),
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                'free_cash_flow': info.get('freeCashflow'),

                # Risk metrics
                'beta': info.get('beta'),

                # Dividend metrics
                'dividend_yield': info.get('dividendYield'),
                'payout_ratio': info.get('payoutRatio'),

                # Short interest metrics
                'short_interest': info.get('sharesShort'),
                'short_ratio': info.get('shortRatio'),
                'short_percent_float': info.get('shortPercentOfFloat'),
                'days_to_cover': None,  # Not directly available

                # Share metrics
                'shares_outstanding': info.get('sharesOutstanding'),
                'float_shares': info.get('floatShares'),
                'institutional_ownership': info.get('heldPercentInstitutions')
            }

            logger.info(f"Collected fundamentals for {ticker}")
            return fundamentals

        except Exception as e:
            logger.error(f"Error collecting fundamentals for {ticker}: {str(e)}")
            return None

    def collect_all_stocks(self, delay: int = 1) -> pd.DataFrame:
        """
        Collect fundamental data for all stock tickers

        Args:
            delay: Delay in seconds between API calls to avoid rate limiting

        Returns:
            DataFrame with all fundamental data
        """
        logger.info("Starting fundamental data collection for all stocks")

        # Get all stock tickers from database
        tickers = self._get_stock_tickers()

        all_data = []
        success_count = 0
        failed_tickers = []

        for idx, ticker in enumerate(tickers, 1):
            # Collect fundamentals for this ticker
            ticker_fundamentals = self.collect_fundamentals(ticker)

            if ticker_fundamentals is not None:
                all_data.append(ticker_fundamentals)
                success_count += 1
            else:
                failed_tickers.append(ticker)

            # Log progress every 10 tickers
            if idx % 10 == 0:
                logger.info(f"Progress: {idx}/{len(tickers)} stocks processed ({success_count} successful)")

            # Small delay to avoid rate limiting
            time.sleep(delay)

        # Convert to DataFrame
        if not all_data:
            logger.warning("No fundamental data collected")
            return pd.DataFrame()

        fundamentals_df = pd.DataFrame(all_data)

        logger.info(f"\n{'='*60}")
        logger.info(f"Collection Summary:")
        logger.info(f"  Total stocks: {len(tickers)}")
        logger.info(f"  Successful: {success_count}")
        logger.info(f"  Failed: {len(failed_tickers)}")
        if failed_tickers:
            logger.info(f"  Failed tickers: {', '.join(failed_tickers)}")
        logger.info(f"  Total records collected: {len(fundamentals_df)}")
        logger.info(f"{'='*60}\n")

        return fundamentals_df

    def populate_fundamentals_table(self, replace: bool = False) -> None:
        """
        Populate the fundamental_data table in the database

        Args:
            replace: If True, replace existing data. If False, append only new data.
        """
        logger.info("Starting fundamental_data table population")

        try:
            # Collect all fundamental data
            fundamentals_df = self.collect_all_stocks()

            if fundamentals_df.empty:
                logger.error("No fundamental data collected. Aborting database insertion.")
                return

            # Connect to database
            conn = self._get_db_connection()
            cursor = conn.cursor()

            # Check existing data
            cursor.execute("SELECT COUNT(*) FROM fundamental_data")
            existing_count = cursor.fetchone()[0]

            if existing_count > 0 and not replace:
                logger.info(f"Found {existing_count} existing records. Appending new data only.")

                # Get existing ticker-date combinations
                existing_df = pd.read_sql(
                    "SELECT symbol_ticker, fundamental_date FROM fundamental_data",
                    conn
                )
                existing_keys = set(
                    zip(existing_df['symbol_ticker'], existing_df['fundamental_date'])
                )

                # Filter to only new records
                new_records_mask = ~fundamentals_df.apply(
                    lambda row: (row['symbol_ticker'], row['fundamental_date']) in existing_keys,
                    axis=1
                )
                new_fundamentals_df = fundamentals_df[new_records_mask]

                if len(new_fundamentals_df) > 0:
                    new_fundamentals_df.to_sql('fundamental_data', conn, if_exists='append', index=False)
                    logger.info(f"Inserted {len(new_fundamentals_df)} new records")
                else:
                    logger.info("No new records to insert")
            else:
                # Replace or insert all
                if_exists = 'replace' if replace else 'append'
                fundamentals_df.to_sql('fundamental_data', conn, if_exists=if_exists, index=False)
                logger.info(f"{'Replaced with' if replace else 'Inserted'} {len(fundamentals_df)} records")

            # Verify insertion and show summary
            cursor.execute("SELECT COUNT(*) FROM fundamental_data")
            final_count = cursor.fetchone()[0]
            logger.info(f"fundamental_data table now contains {final_count} total records")

            # Show summary statistics
            summary_df = pd.read_sql("""
                SELECT
                    symbol_ticker,
                    fundamental_date,
                    ROUND(market_cap / 1000000000.0, 2) as market_cap_b,
                    ROUND(pe_ratio, 2) as pe_ratio,
                    ROUND(price_to_book, 2) as pb_ratio,
                    ROUND(profit_margin * 100, 2) as profit_margin_pct,
                    ROUND(revenue_growth * 100, 2) as revenue_growth_pct,
                    ROUND(beta, 2) as beta
                FROM fundamental_data
                ORDER BY market_cap DESC
                LIMIT 15
            """, conn)

            logger.info(f"\nTop 15 Stocks by Market Cap:")
            logger.info(f"\n{summary_df.to_string(index=False)}")

            # Data completeness check
            completeness_df = pd.read_sql("""
                SELECT
                    COUNT(*) as total_records,
                    SUM(CASE WHEN market_cap IS NOT NULL THEN 1 ELSE 0 END) as has_market_cap,
                    SUM(CASE WHEN pe_ratio IS NOT NULL THEN 1 ELSE 0 END) as has_pe_ratio,
                    SUM(CASE WHEN revenue IS NOT NULL THEN 1 ELSE 0 END) as has_revenue,
                    SUM(CASE WHEN beta IS NOT NULL THEN 1 ELSE 0 END) as has_beta,
                    SUM(CASE WHEN dividend_yield IS NOT NULL THEN 1 ELSE 0 END) as has_dividend
                FROM fundamental_data
            """, conn)

            logger.info(f"\nData Completeness:")
            logger.info(f"\n{completeness_df.to_string(index=False)}")

            conn.close()
            logger.info("Successfully populated fundamental_data table")

        except Exception as e:
            logger.error(f"Error populating fundamental_data table: {str(e)}")
            raise

    def get_fundamentals_summary(self) -> pd.DataFrame:
        """Get summary of fundamental data in the database"""
        try:
            conn = self._get_db_connection()
            query = """
                SELECT
                    symbol_ticker,
                    fundamental_date,
                    market_cap,
                    pe_ratio,
                    beta
                FROM fundamental_data
                ORDER BY symbol_ticker
            """
            df = pd.read_sql(query, conn)
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Error getting fundamentals summary: {str(e)}")
            raise


if __name__ == "__main__":
    # Initialize collector
    collector = FundamentalDataCollector()

    print(f"\n{'='*60}")
    print(f"Fundamental Data Collection Script")
    print(f"{'='*60}")
    print(f"Database: {collector.db_path}")
    print(f"Collection Date: {collector.collection_date}")
    print(f"Data Source: yfinance")
    print(f"Note: Only collecting for stocks (ETFs excluded)")

    # Populate database
    print(f"\n{'='*60}")
    print("Collecting fundamental data for all stocks...")
    print(f"{'='*60}\n")

    collector.populate_fundamentals_table(replace=True)

    print(f"\n{'='*60}")
    print("Fundamental data collection complete!")
    print(f"{'='*60}\n")