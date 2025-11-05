import sqlite3
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalystRatingsCollector:
    """
    Collects analyst ratings and price targets for all stocks

    Database: /Volumes/Vault/85_assets_prediction.db
    Table: analyst_ratings
    Data Source: Finnhub API
    Time Period: Last 6 months of analyst ratings
    """

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db") -> None:
        """Initialize the collector with database path and API keys"""
        self.db_path = db_path

        # Finnhub API key
        self.finnhub_api_key = "d3nuofhr01qmj82v1vdgd3nuofhr01qmj82v1ve0"
        self.finnhub_base_url = "https://finnhub.io/api/v1"

        # Date range for analyst ratings (last 6 months)
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=180)

        # Rating to numeric mapping
        self.rating_to_numeric = {
            'strong buy': 5,
            'buy': 4,
            'hold': 3,
            'sell': 2,
            'strong sell': 1
        }

        logger.info(f"Initialized AnalystRatingsCollector")
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

    def _convert_rating_to_numeric(self, rating: str) -> Optional[float]:
        """Convert text rating to numeric value"""
        if not rating:
            return None

        rating_lower = rating.lower().strip()
        return self.rating_to_numeric.get(rating_lower)

    def _determine_rating_change(self, current_rating: str, previous_rating: str) -> Optional[str]:
        """Determine rating change direction"""
        if not current_rating or not previous_rating:
            return None

        current_numeric = self._convert_rating_to_numeric(current_rating)
        previous_numeric = self._convert_rating_to_numeric(previous_rating)

        if current_numeric is None or previous_numeric is None:
            return None

        if current_numeric > previous_numeric:
            return 'Upgrade'
        elif current_numeric < previous_numeric:
            return 'Downgrade'
        else:
            return 'Maintained'

    def collect_analyst_recommendations(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Collect analyst recommendations from Finnhub

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of analyst rating records
        """
        try:
            logger.info(f"Collecting analyst ratings for {ticker}")

            # Finnhub recommendation trends endpoint
            url = f"{self.finnhub_base_url}/stock/recommendation"
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
                logger.warning(f"No analyst rating data for {ticker}")
                return []

            rating_records = []

            # Filter data to last 6 months and process
            for idx, recommendation in enumerate(data):
                rating_date = recommendation.get('period')

                if not rating_date:
                    continue

                # Check if within date range
                try:
                    rating_date_obj = datetime.strptime(rating_date, '%Y-%m-%d')
                    if rating_date_obj < self.start_date:
                        continue
                except:
                    continue

                # Determine consensus rating based on counts
                buy_count = recommendation.get('buy', 0) + recommendation.get('strongBuy', 0)
                hold_count = recommendation.get('hold', 0)
                sell_count = recommendation.get('sell', 0) + recommendation.get('strongSell', 0)

                total_ratings = buy_count + hold_count + sell_count

                if total_ratings == 0:
                    consensus_rating = 'Hold'
                elif buy_count > hold_count and buy_count > sell_count:
                    consensus_rating = 'Buy'
                elif sell_count > hold_count and sell_count > buy_count:
                    consensus_rating = 'Sell'
                else:
                    consensus_rating = 'Hold'

                # Get previous rating for comparison
                previous_rating = None
                if idx < len(data) - 1:
                    prev_rec = data[idx + 1]
                    prev_buy = prev_rec.get('buy', 0) + prev_rec.get('strongBuy', 0)
                    prev_hold = prev_rec.get('hold', 0)
                    prev_sell = prev_rec.get('sell', 0) + prev_rec.get('strongSell', 0)
                    prev_total = prev_buy + prev_hold + prev_sell

                    if prev_total > 0:
                        if prev_buy > prev_hold and prev_buy > prev_sell:
                            previous_rating = 'Buy'
                        elif prev_sell > prev_hold and prev_sell > prev_buy:
                            previous_rating = 'Sell'
                        else:
                            previous_rating = 'Hold'

                record = {
                    'symbol_ticker': ticker,
                    'rating_date': rating_date,
                    'firm_name': None,  # Aggregated data, no specific firm
                    'analyst_name': None,  # Aggregated data
                    'rating': consensus_rating,
                    'rating_numeric': self._convert_rating_to_numeric(consensus_rating),
                    'rating_change': self._determine_rating_change(consensus_rating, previous_rating),
                    'previous_rating': previous_rating,
                    'price_target': None,  # Not in recommendation trends endpoint
                    'previous_price_target': None,
                    'price_target_change_percent': None,
                    'upside_to_target': None
                }

                rating_records.append(record)

            logger.info(f"Collected {len(rating_records)} analyst rating records for {ticker}")
            return rating_records

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error collecting analyst ratings for {ticker}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error collecting analyst ratings for {ticker}: {str(e)}")
            return []

    def collect_price_targets(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Collect analyst price targets from yfinance

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of price target records
        """
        try:
            logger.info(f"Collecting price targets for {ticker}")

            # Get stock info from yfinance
            stock = yf.Ticker(ticker)
            info = stock.info

            # Get current price for upside calculation
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')

            # Get analyst price targets
            target_mean = info.get('targetMeanPrice')
            target_high = info.get('targetHighPrice')
            target_low = info.get('targetLowPrice')
            target_median = info.get('targetMedianPrice')

            if not target_mean:
                logger.warning(f"No price target data for {ticker}")
                return []

            # Calculate upside to target
            upside_to_target = None
            if current_price and target_mean:
                upside_to_target = ((target_mean - current_price) / current_price) * 100

            # Create a single record with current consensus
            record = {
                'symbol_ticker': ticker,
                'rating_date': datetime.now().strftime('%Y-%m-%d'),
                'firm_name': None,  # Consensus data
                'analyst_name': None,
                'rating': None,  # Price target only, no rating
                'rating_numeric': None,
                'rating_change': None,
                'previous_rating': None,
                'price_target': target_mean,
                'previous_price_target': None,
                'price_target_change_percent': None,
                'upside_to_target': upside_to_target
            }

            logger.info(f"Collected price target for {ticker}: ${target_mean:.2f} (upside: {upside_to_target:.2f}%)" if target_mean and upside_to_target else f"Collected price target for {ticker}")
            return [record]

        except Exception as e:
            logger.error(f"Error collecting price targets for {ticker}: {str(e)}")
            return []

    def merge_ratings_and_targets(self, ratings: List[Dict], targets: List[Dict]) -> pd.DataFrame:
        """
        Merge analyst ratings and price targets
        """
        all_records = []

        # Add all rating records
        all_records.extend(ratings)

        # Add price target if not already covered by ratings
        if targets and ratings:
            target_date = targets[0]['rating_date']
            # Check if we already have a record for this date
            existing_dates = {r['rating_date'] for r in ratings}
            if target_date not in existing_dates:
                all_records.extend(targets)
        elif targets and not ratings:
            all_records.extend(targets)

        if not all_records:
            return pd.DataFrame()

        return pd.DataFrame(all_records)

    def collect_all_analyst_ratings(self) -> pd.DataFrame:
        """
        Collect analyst ratings for all stock tickers
        """
        logger.info("Starting analyst ratings collection for all stocks")

        # Get all stock tickers
        tickers = self._get_stock_tickers()

        all_ratings = []
        success_count = 0
        failed_tickers = []

        for idx, ticker in enumerate(tickers, 1):
            try:
                # Collect analyst ratings
                ratings = self.collect_analyst_recommendations(ticker)
                time.sleep(0.5)  # Rate limiting

                # Collect price targets
                targets = self.collect_price_targets(ticker)
                time.sleep(0.5)  # Rate limiting

                # Merge data
                if ratings or targets:
                    merged_data = self.merge_ratings_and_targets(ratings, targets)
                    if not merged_data.empty:
                        all_ratings.append(merged_data)
                        success_count += 1

                # Log progress every 10 tickers
                if idx % 10 == 0:
                    logger.info(f"Progress: {idx}/{len(tickers)} stocks processed ({success_count} with data)")

            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                failed_tickers.append(ticker)

        # Combine all ratings data
        if all_ratings:
            ratings_df = pd.concat(all_ratings, axis=0, ignore_index=True)
            ratings_df = ratings_df.sort_values(['symbol_ticker', 'rating_date'])

            logger.info(f"\n{'='*60}")
            logger.info(f"Collection Summary:")
            logger.info(f"  Total stocks: {len(tickers)}")
            logger.info(f"  Successful: {success_count}")
            logger.info(f"  Failed: {len(failed_tickers)}")
            if failed_tickers:
                logger.info(f"  Failed tickers: {', '.join(failed_tickers[:10])}")
            logger.info(f"  Total analyst rating records: {len(ratings_df)}")
            logger.info(f"{'='*60}\n")

            return ratings_df
        else:
            logger.warning("No analyst ratings data collected")
            return pd.DataFrame()

    def populate_analyst_ratings_table(self, replace: bool = False) -> None:
        """
        Populate the analyst_ratings table in the database

        Args:
            replace: If True, replace existing data. If False, append only new data.
        """
        logger.info("Starting analyst_ratings table population")

        try:
            # Collect all analyst ratings data
            ratings_df = self.collect_all_analyst_ratings()

            if ratings_df.empty:
                logger.error("No analyst ratings data collected. Aborting database insertion.")
                return

            # Connect to database
            conn = self._get_db_connection()
            cursor = conn.cursor()

            # Check existing data
            cursor.execute("SELECT COUNT(*) FROM analyst_ratings")
            existing_count = cursor.fetchone()[0]

            if existing_count > 0 and not replace:
                logger.info(f"Found {existing_count} existing records. Appending new data only.")

                # Get existing ticker-date combinations
                existing_df = pd.read_sql(
                    "SELECT symbol_ticker, rating_date FROM analyst_ratings",
                    conn
                )
                existing_keys = set(
                    zip(existing_df['symbol_ticker'], existing_df['rating_date'])
                )

                # Filter to only new records
                new_records_mask = ~ratings_df.apply(
                    lambda row: (row['symbol_ticker'], row['rating_date']) in existing_keys,
                    axis=1
                )
                new_ratings_df = ratings_df[new_records_mask]

                if len(new_ratings_df) > 0:
                    new_ratings_df.to_sql('analyst_ratings', conn, if_exists='append', index=False)
                    logger.info(f"Inserted {len(new_ratings_df)} new records")
                else:
                    logger.info("No new records to insert")
            else:
                # Replace or insert all
                if_exists = 'replace' if replace else 'append'
                ratings_df.to_sql('analyst_ratings', conn, if_exists=if_exists, index=False)
                logger.info(f"{'Replaced with' if replace else 'Inserted'} {len(ratings_df)} records")

            # Verify insertion and show summary
            cursor.execute("SELECT COUNT(*) FROM analyst_ratings")
            final_count = cursor.fetchone()[0]
            logger.info(f"analyst_ratings table now contains {final_count} total records")

            # Show summary statistics
            summary_df = pd.read_sql("""
                SELECT
                    symbol_ticker,
                    COUNT(*) as rating_count,
                    AVG(rating_numeric) as avg_rating,
                    MIN(rating_date) as earliest_rating,
                    MAX(rating_date) as latest_rating
                FROM analyst_ratings
                WHERE rating_numeric IS NOT NULL
                GROUP BY symbol_ticker
                ORDER BY symbol_ticker
                LIMIT 15
            """, conn)

            logger.info(f"\nAnalyst Ratings Summary (first 15 stocks):")
            logger.info(f"\n{summary_df.to_string(index=False)}")

            # Recent ratings
            recent_df = pd.read_sql("""
                SELECT
                    symbol_ticker,
                    rating_date,
                    rating,
                    rating_change,
                    ROUND(price_target, 2) as target
                FROM analyst_ratings
                ORDER BY rating_date DESC
                LIMIT 15
            """, conn)

            logger.info(f"\nRecent Analyst Ratings (last 15):")
            logger.info(f"\n{recent_df.to_string(index=False)}")

            # Rating distribution
            distribution_df = pd.read_sql("""
                SELECT
                    rating,
                    COUNT(*) as count
                FROM analyst_ratings
                WHERE rating IS NOT NULL
                GROUP BY rating
                ORDER BY rating
            """, conn)

            logger.info(f"\nRating Distribution:")
            logger.info(f"\n{distribution_df.to_string(index=False)}")

            conn.close()
            logger.info("Successfully populated analyst_ratings table")

        except Exception as e:
            logger.error(f"Error populating analyst_ratings table: {str(e)}")
            raise


if __name__ == "__main__":
    # Initialize collector
    collector = AnalystRatingsCollector()

    print(f"\n{'='*60}")
    print(f"Analyst Ratings Data Collection Script")
    print(f"{'='*60}")
    print(f"Database: {collector.db_path}")
    print(f"Date Range: {collector.start_date.date()} to {collector.end_date.date()}")
    print(f"Data Source: Finnhub API")

    # Populate database
    print(f"\n{'='*60}")
    print("Collecting analyst ratings for all stocks...")
    print(f"{'='*60}\n")

    collector.populate_analyst_ratings_table(replace=True)

    print(f"\n{'='*60}")
    print("Analyst ratings data collection complete!")
    print(f"{'='*60}\n")
