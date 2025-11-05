import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuarterlyFundamentalsBackfiller:
    """
    Backfills quarterly fundamental data for all stocks using historical financials

    Database: /Volumes/Vault/85_assets_prediction.db
    Table: fundamental_data
    Data Source: yfinance (quarterly financials)
    Timeframe: Last 8 quarters (2 years)
    """

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db") -> None:
        """Initialize the backfiller with database path"""
        self.db_path = db_path
        logger.info(f"Initialized QuarterlyFundamentalsBackfiller")

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

    def _get_quarterly_dates(self, financials_df: pd.DataFrame) -> List[str]:
        """
        Extract quarterly dates from financials dataframe

        Args:
            financials_df: DataFrame with quarterly financials

        Returns:
            List of quarter-end dates (YYYY-MM-DD format)
        """
        if financials_df.empty:
            return []

        # Column names are the quarter-end dates
        dates = [col.strftime('%Y-%m-%d') for col in financials_df.columns if isinstance(col, pd.Timestamp)]
        return sorted(dates, reverse=True)[:8]  # Last 8 quarters

    def collect_quarterly_fundamentals(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Collect quarterly fundamental data for a single ticker

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of dictionaries with quarterly fundamental data
        """
        try:
            logger.info(f"Collecting quarterly fundamentals for {ticker}")

            stock = yf.Ticker(ticker)

            # Get current info for latest valuation metrics
            info = stock.info

            # Get quarterly financials
            quarterly_financials = stock.quarterly_financials
            quarterly_balance_sheet = stock.quarterly_balance_sheet
            quarterly_cashflow = stock.quarterly_cashflow

            if quarterly_financials.empty:
                logger.warning(f"No quarterly financials available for {ticker}")
                return []

            # Get quarter dates
            quarter_dates = self._get_quarterly_dates(quarterly_financials)

            if not quarter_dates:
                logger.warning(f"No quarter dates found for {ticker}")
                return []

            fundamentals_list = []

            for quarter_date in quarter_dates:
                try:
                    # Convert date string to timestamp for indexing
                    date_ts = pd.Timestamp(quarter_date)

                    # Extract data matching the actual database schema
                    quarter_data = {
                        'symbol_ticker': ticker,
                        'fundamental_date': quarter_date,
                    }

                    # Get net income and revenue for calculations
                    net_income = None
                    total_revenue = None
                    operating_income = None
                    total_assets = None
                    total_equity = None

                    # Income Statement metrics
                    if date_ts in quarterly_financials.columns:
                        if 'Total Revenue' in quarterly_financials.index:
                            total_revenue = float(quarterly_financials.loc['Total Revenue', date_ts])
                        if 'Net Income' in quarterly_financials.index:
                            net_income = float(quarterly_financials.loc['Net Income', date_ts])
                        if 'Operating Income' in quarterly_financials.index:
                            operating_income = float(quarterly_financials.loc['Operating Income', date_ts])

                    # Balance Sheet metrics
                    if date_ts in quarterly_balance_sheet.columns:
                        if 'Total Assets' in quarterly_balance_sheet.index:
                            total_assets = float(quarterly_balance_sheet.loc['Total Assets', date_ts])
                        if 'Stockholders Equity' in quarterly_balance_sheet.index:
                            total_equity = float(quarterly_balance_sheet.loc['Stockholders Equity', date_ts])
                        if 'Total Debt' in quarterly_balance_sheet.index:
                            quarter_data['total_debt'] = float(quarterly_balance_sheet.loc['Total Debt', date_ts])
                        if 'Current Assets' in quarterly_balance_sheet.index and 'Current Liabilities' in quarterly_balance_sheet.index:
                            current_assets = float(quarterly_balance_sheet.loc['Current Assets', date_ts])
                            current_liabilities = float(quarterly_balance_sheet.loc['Current Liabilities', date_ts])
                            quarter_data['current_ratio'] = current_assets / current_liabilities if current_liabilities != 0 else None

                            # Quick ratio (if we have inventory data)
                            if 'Inventory' in quarterly_balance_sheet.index:
                                inventory = float(quarterly_balance_sheet.loc['Inventory', date_ts])
                                quarter_data['quick_ratio'] = (current_assets - inventory) / current_liabilities if current_liabilities != 0 else None

                    # Cash Flow metrics
                    if date_ts in quarterly_cashflow.columns:
                        if 'Free Cash Flow' in quarterly_cashflow.index:
                            quarter_data['free_cash_flow'] = float(quarterly_cashflow.loc['Free Cash Flow', date_ts])

                    # Calculate derived metrics matching schema
                    quarter_data['revenue'] = total_revenue
                    quarter_data['earnings_per_share'] = info.get('trailingEps')

                    if total_revenue and net_income:
                        quarter_data['profit_margin'] = net_income / total_revenue

                    if total_revenue and operating_income:
                        quarter_data['operating_margin'] = operating_income / total_revenue

                    if total_equity and net_income:
                        quarter_data['return_on_equity'] = net_income / total_equity

                    if total_assets and net_income:
                        quarter_data['return_on_assets'] = net_income / total_assets

                    if quarter_data.get('total_debt') and total_equity:
                        quarter_data['debt_to_equity'] = quarter_data['total_debt'] / total_equity

                    # Valuation metrics (from current info)
                    quarter_data['market_cap'] = info.get('marketCap')
                    quarter_data['enterprise_value'] = info.get('enterpriseValue')
                    quarter_data['pe_ratio'] = info.get('trailingPE')
                    quarter_data['forward_pe'] = info.get('forwardPE')
                    quarter_data['peg_ratio'] = info.get('pegRatio')
                    quarter_data['price_to_book'] = info.get('priceToBook')
                    quarter_data['price_to_sales'] = info.get('priceToSalesTrailing12Months')
                    quarter_data['ev_to_ebitda'] = info.get('enterpriseToEbitda')
                    quarter_data['ev_to_revenue'] = info.get('enterpriseToRevenue')

                    # Growth metrics
                    quarter_data['revenue_growth'] = info.get('revenueGrowth')
                    quarter_data['earnings_growth'] = info.get('earningsGrowth')

                    # Shares and ownership
                    quarter_data['shares_outstanding'] = info.get('sharesOutstanding')
                    quarter_data['float_shares'] = info.get('floatShares')
                    quarter_data['institutional_ownership'] = info.get('heldPercentInstitutions')

                    # Dividend metrics
                    quarter_data['dividend_yield'] = info.get('dividendYield')
                    quarter_data['payout_ratio'] = info.get('payoutRatio')

                    # Short interest
                    quarter_data['short_interest'] = info.get('sharesShort')
                    quarter_data['short_ratio'] = info.get('shortRatio')
                    quarter_data['short_percent_float'] = info.get('shortPercentOfFloat')

                    # Beta
                    quarter_data['beta'] = info.get('beta')

                    # ROIC and other fields
                    quarter_data['return_on_invested_capital'] = None  # Not directly available
                    quarter_data['days_to_cover'] = None  # Not directly available

                    fundamentals_list.append(quarter_data)

                except Exception as e:
                    logger.warning(f"Error processing quarter {quarter_date} for {ticker}: {str(e)}")
                    continue

            logger.info(f"✓ Collected {len(fundamentals_list)} quarters for {ticker}")
            return fundamentals_list

        except Exception as e:
            logger.error(f"Error collecting fundamentals for {ticker}: {str(e)}")
            return []

    def backfill_all_fundamentals(self) -> pd.DataFrame:
        """
        Backfill quarterly fundamentals for all stocks

        Returns:
            DataFrame with all quarterly fundamental data
        """
        logger.info("Starting quarterly fundamentals backfill for all stocks")

        tickers = self._get_stock_tickers()
        all_fundamentals = []
        success_count = 0

        for idx, ticker in enumerate(tickers, 1):
            try:
                quarterly_data = self.collect_quarterly_fundamentals(ticker)

                if quarterly_data:
                    all_fundamentals.extend(quarterly_data)
                    success_count += 1

                # Log progress every 10 tickers
                if idx % 10 == 0:
                    logger.info(f"Progress: {idx}/{len(tickers)} tickers processed ({success_count} successful)")

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")

        if all_fundamentals:
            df = pd.DataFrame(all_fundamentals)
            logger.info(f"Collected {len(df)} quarterly records across {success_count} stocks")
            return df
        else:
            logger.warning("No fundamental data collected")
            return pd.DataFrame()

    def populate_fundamentals_table(self, replace: bool = False) -> None:
        """
        Populate the fundamental_data table with quarterly historical data

        Args:
            replace: If True, replace existing data. If False, append only new data.
        """
        logger.info("Starting fundamental_data table backfill")

        try:
            # Collect all quarterly fundamentals
            fundamentals_df = self.backfill_all_fundamentals()

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

            # Show summary by ticker
            summary_df = pd.read_sql("""
                SELECT
                    symbol_ticker,
                    COUNT(*) as quarter_count,
                    MIN(fundamental_date) as earliest_quarter,
                    MAX(fundamental_date) as latest_quarter
                FROM fundamental_data
                GROUP BY symbol_ticker
                ORDER BY symbol_ticker
                LIMIT 10
            """, conn)

            logger.info(f"\nFundamental Data Summary (first 10 tickers):")
            logger.info(f"\n{summary_df.to_string(index=False)}")

            # Show sample data for one ticker
            sample_df = pd.read_sql("""
                SELECT
                    fundamental_date,
                    revenue,
                    ROUND(profit_margin, 4) as margin,
                    ROUND(return_on_equity, 4) as roe,
                    pe_ratio
                FROM fundamental_data
                WHERE symbol_ticker = 'AAPL'
                ORDER BY fundamental_date DESC
                LIMIT 5
            """, conn)

            logger.info(f"\nRecent Fundamentals for AAPL (sample):")
            logger.info(f"\n{sample_df.to_string(index=False)}")

            conn.close()
            logger.info("Successfully backfilled fundamental_data table")

        except Exception as e:
            logger.error(f"Error backfilling fundamental_data table: {str(e)}")
            raise


if __name__ == "__main__":
    # Initialize backfiller
    backfiller = QuarterlyFundamentalsBackfiller()

    print(f"\n{'='*60}")
    print(f"Quarterly Fundamentals Backfill Script")
    print(f"{'='*60}")
    print(f"Database: {backfiller.db_path}")
    print(f"Target: Last 8 quarters (2 years) for all stocks")

    # Populate database
    print(f"\n{'='*60}")
    print("Backfilling quarterly fundamental data...")
    print(f"{'='*60}\n")

    backfiller.populate_fundamentals_table(replace=False)

    print(f"\n{'='*60}")
    print("Quarterly fundamentals backfill complete!")
    print(f"{'='*60}\n")