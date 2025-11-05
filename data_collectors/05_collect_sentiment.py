import sqlite3
import pandas as pd
import praw
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
import time
from newsapi import NewsApiClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentDataCollector:
    """
    Collects sentiment data from news sources and Reddit

    Database: /Volumes/Vault/85_assets_prediction.db
    Table: sentiment_data
    Data Sources:
        - NewsAPI: News headlines and sentiment
        - Reddit (PRAW): Reddit posts and comments
    Time Period: Last 30 days for sentiment (rolling window)
    Focus: Top 10 sentiment-tracked assets
    """

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db") -> None:
        """Initialize the collector with database path and API keys"""
        self.db_path = db_path

        # API Keys
        self.news_api_key = "e78eec9cf81b49c89400158d0975ae3e"
        self.reddit_client_id = "izdVpTnfSo46e5N-Cz_XjA"
        self.reddit_client_secret = "GnYwXq5_krmjwoizveSUv9-hjnJ-dw" 
        self.reddit_user_agent = "TradingBot/1.0"

        # Initialize APIs
        self.newsapi = NewsApiClient(api_key=self.news_api_key)

        # Reddit API
        self.reddit = None
        try:
            self.reddit = praw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent=self.reddit_user_agent
            )
            logger.info("Reddit API initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize Reddit API: {str(e)}")

        # Date range for sentiment (6-12 months historical)
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=180)  # 6 months

        # Get all 85 tickers from database
        self.sentiment_tickers = self._get_all_tickers()

        logger.info(f"Initialized SentimentDataCollector")
        logger.info(f"Date range: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"Tracking sentiment for: {', '.join(self.sentiment_tickers)}")

    def _get_db_connection(self) -> sqlite3.Connection:
        """Create and return database connection"""
        return sqlite3.connect(self.db_path)

    def _get_all_tickers(self) -> List[str]:
        """Get all 85 tickers from the assets table"""
        try:
            conn = self._get_db_connection()
            query = "SELECT symbol_ticker FROM assets ORDER BY symbol_ticker"
            df = pd.read_sql(query, conn)
            conn.close()
            tickers = df['symbol_ticker'].tolist()
            logger.info(f"Retrieved {len(tickers)} tickers from assets table")
            return tickers
        except Exception as e:
            logger.warning(f"Could not retrieve tickers from database: {str(e)}")
            logger.warning("Falling back to default sentiment tickers")
            # Fallback to original list if database is not populated yet
            return [
                'TSLA', 'AAPL', 'GME', 'AMC', 'NVDA',
                'AMD', 'SPY', 'PLTR', 'META', 'SOFI'
            ]

    def _calculate_sentiment_score(self, text: str) -> Dict[str, Any]:
        """
        Simple sentiment analysis (placeholder for more sophisticated analysis)

        In production, you would use:
        - VADER sentiment
        - TextBlob
        - Transformers-based models (FinBERT, etc.)
        """
        # Simple keyword-based sentiment (placeholder)
        positive_keywords = ['buy', 'bull', 'bullish', 'moon', 'rocket', 'growth', 'up', 'gains', 'profit', 'winning']
        negative_keywords = ['sell', 'bear', 'bearish', 'crash', 'down', 'loss', 'losing', 'drop', 'fall', 'dump']

        text_lower = text.lower()

        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)

        total = positive_count + negative_count
        if total == 0:
            sentiment_score = 0.0
            sentiment_direction = 'neutral'
        else:
            sentiment_score = (positive_count - negative_count) / total
            if sentiment_score > 0.2:
                sentiment_direction = 'positive'
            elif sentiment_score < -0.2:
                sentiment_direction = 'negative'
            else:
                sentiment_direction = 'neutral'

        return {
            'sentiment_score': sentiment_score,
            'sentiment_direction': sentiment_direction,
            'positive_mentions': positive_count,
            'negative_mentions': negative_count,
            'neutral_mentions': total - positive_count - negative_count if total > 0 else 1
        }

    def collect_news_sentiment(self, ticker: str, days_back: int = 30) -> List[Dict[str, Any]]:
        """
        Collect news sentiment for a ticker using NewsAPI

        Args:
            ticker: Stock ticker symbol
            days_back: Number of days to look back

        Returns:
            List of sentiment records
        """
        try:
            logger.info(f"Collecting news sentiment for {ticker}")

            # Calculate date range (NewsAPI free tier only allows last 30 days)
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')

            # Search query
            query = f"{ticker} stock OR ${ticker}"

            # Get articles from NewsAPI
            articles = self.newsapi.get_everything(
                q=query,
                from_param=from_date,
                to=to_date,
                language='en',
                sort_by='relevancy',
                page_size=100
            )

            sentiment_records = []

            if articles['totalResults'] > 0:
                for article in articles['articles']:
                    # Combine title and description for sentiment analysis
                    text = f"{article.get('title', '')} {article.get('description', '')}"

                    # Calculate sentiment
                    sentiment = self._calculate_sentiment_score(text)

                    # Create sentiment record
                    record = {
                        'symbol_ticker': ticker,
                        'sentiment_date': article['publishedAt'][:10],  # Extract date only
                        'source': 'NewsAPI',
                        'sentiment_score': sentiment['sentiment_score'],
                        'sentiment_magnitude': abs(sentiment['sentiment_score']),
                        'sentiment_direction': sentiment['sentiment_direction'],
                        'mention_count': 1,
                        'positive_mentions': sentiment['positive_mentions'],
                        'negative_mentions': sentiment['negative_mentions'],
                        'neutral_mentions': sentiment['neutral_mentions'],
                        'engagement_score': None,  # Not available from NewsAPI
                        'virality_score': None,
                        'sentiment_momentum_3d': None,  # Will be calculated later
                        'sentiment_acceleration': None,
                        'price_sentiment_divergence': None,
                        'sample_headlines': article.get('title', '')[:200]
                    }

                    sentiment_records.append(record)

                logger.info(f"Collected {len(sentiment_records)} news articles for {ticker}")
            else:
                logger.warning(f"No news articles found for {ticker}")

            return sentiment_records

        except Exception as e:
            logger.error(f"Error collecting news sentiment for {ticker}: {str(e)}")
            return []

    def collect_reddit_sentiment(self, ticker: str, days_back: int = 7) -> List[Dict[str, Any]]:
        """
        Collect Reddit sentiment for a ticker using PRAW

        Note: Requires Reddit API secret which was not provided
        """
        if self.reddit is None:
            logger.warning(f"Reddit API not initialized, skipping Reddit sentiment for {ticker}")
            return []

        try:
            logger.info(f"Collecting Reddit sentiment for {ticker}")

            sentiment_records = []

            # Search in popular trading subreddits
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'StockMarket']

            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)

                    # Search for ticker mentions
                    for submission in subreddit.search(ticker, time_filter='week', limit=50):
                        # Combine title and selftext
                        text = f"{submission.title} {submission.selftext}"

                        # Calculate sentiment
                        sentiment = self._calculate_sentiment_score(text)

                        # Create sentiment record
                        record = {
                            'symbol_ticker': ticker,
                            'sentiment_date': datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d'),
                            'source': f'Reddit_{subreddit_name}',
                            'sentiment_score': sentiment['sentiment_score'],
                            'sentiment_magnitude': abs(sentiment['sentiment_score']),
                            'sentiment_direction': sentiment['sentiment_direction'],
                            'mention_count': 1,
                            'positive_mentions': sentiment['positive_mentions'],
                            'negative_mentions': sentiment['negative_mentions'],
                            'neutral_mentions': sentiment['neutral_mentions'],
                            'engagement_score': submission.score,
                            'virality_score': submission.num_comments,
                            'sentiment_momentum_3d': None,
                            'sentiment_acceleration': None,
                            'price_sentiment_divergence': None,
                            'sample_headlines': submission.title[:200]
                        }

                        sentiment_records.append(record)

                    time.sleep(1)  # Rate limiting

                except Exception as e:
                    logger.warning(f"Error accessing r/{subreddit_name}: {str(e)}")

            logger.info(f"Collected {len(sentiment_records)} Reddit posts for {ticker}")
            return sentiment_records

        except Exception as e:
            logger.error(f"Error collecting Reddit sentiment for {ticker}: {str(e)}")
            return []

    def aggregate_daily_sentiment(self, sentiment_records: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Aggregate sentiment records by date and calculate daily metrics
        """
        if not sentiment_records:
            return pd.DataFrame()

        df = pd.DataFrame(sentiment_records)

        # Group by ticker, date, and source
        agg_df = df.groupby(['symbol_ticker', 'sentiment_date', 'source']).agg({
            'sentiment_score': 'mean',
            'sentiment_magnitude': 'mean',
            'mention_count': 'sum',
            'positive_mentions': 'sum',
            'negative_mentions': 'sum',
            'neutral_mentions': 'sum',
            'engagement_score': 'sum',
            'virality_score': 'sum'
        }).reset_index()

        # Determine overall sentiment direction
        agg_df['sentiment_direction'] = agg_df['sentiment_score'].apply(
            lambda x: 'positive' if x > 0.2 else ('negative' if x < -0.2 else 'neutral')
        )

        # Add placeholder columns
        agg_df['sentiment_momentum_3d'] = None
        agg_df['sentiment_acceleration'] = None
        agg_df['price_sentiment_divergence'] = None
        agg_df['sample_headlines'] = df.groupby(['symbol_ticker', 'sentiment_date', 'source'])['sample_headlines'].first().values

        return agg_df

    def collect_all_sentiment(self) -> pd.DataFrame:
        """
        Collect sentiment data for all tracked tickers
        """
        logger.info(f"Starting sentiment collection for {len(self.sentiment_tickers)} tickers")

        all_sentiment = []

        for ticker in self.sentiment_tickers:
            # Collect news sentiment (30 days - NewsAPI free tier limit)
            news_sentiment = self.collect_news_sentiment(ticker, days_back=30)
            all_sentiment.extend(news_sentiment)

            # Collect Reddit sentiment (if available)
            reddit_sentiment = self.collect_reddit_sentiment(ticker, days_back=30)
            all_sentiment.extend(reddit_sentiment)

            time.sleep(2)  # Rate limiting for larger dataset

        # Aggregate sentiment
        if all_sentiment:
            sentiment_df = self.aggregate_daily_sentiment(all_sentiment)
            logger.info(f"Collected {len(sentiment_df)} sentiment records")
            return sentiment_df
        else:
            logger.warning("No sentiment data collected")
            return pd.DataFrame()

    def populate_sentiment_table(self, replace: bool = False) -> None:
        """
        Populate the sentiment_data table in the database

        Args:
            replace: If True, replace existing data. If False, append only new data.
        """
        logger.info("Starting sentiment_data table population")

        try:
            # Collect all sentiment data
            sentiment_df = self.collect_all_sentiment()

            if sentiment_df.empty:
                logger.error("No sentiment data collected. Aborting database insertion.")
                return

            # Connect to database
            conn = self._get_db_connection()
            cursor = conn.cursor()

            # Check existing data
            cursor.execute("SELECT COUNT(*) FROM sentiment_data")
            existing_count = cursor.fetchone()[0]

            if existing_count > 0 and not replace:
                logger.info(f"Found {existing_count} existing records. Appending new data only.")

                # Get existing ticker-date-source combinations
                existing_df = pd.read_sql(
                    "SELECT symbol_ticker, sentiment_date, source FROM sentiment_data",
                    conn
                )
                existing_keys = set(
                    zip(existing_df['symbol_ticker'], existing_df['sentiment_date'], existing_df['source'])
                )

                # Filter to only new records
                new_records_mask = ~sentiment_df.apply(
                    lambda row: (row['symbol_ticker'], row['sentiment_date'], row['source']) in existing_keys,
                    axis=1
                )
                new_sentiment_df = sentiment_df[new_records_mask]

                if len(new_sentiment_df) > 0:
                    new_sentiment_df.to_sql('sentiment_data', conn, if_exists='append', index=False)
                    logger.info(f"Inserted {len(new_sentiment_df)} new records")
                else:
                    logger.info("No new records to insert")
            else:
                # Replace or insert all
                if_exists = 'replace' if replace else 'append'
                sentiment_df.to_sql('sentiment_data', conn, if_exists=if_exists, index=False)
                logger.info(f"{'Replaced with' if replace else 'Inserted'} {len(sentiment_df)} records")

            # Verify insertion and show summary
            cursor.execute("SELECT COUNT(*) FROM sentiment_data")
            final_count = cursor.fetchone()[0]
            logger.info(f"sentiment_data table now contains {final_count} total records")

            # Show summary statistics
            summary_df = pd.read_sql("""
                SELECT
                    symbol_ticker,
                    source,
                    COUNT(*) as record_count,
                    ROUND(AVG(sentiment_score), 3) as avg_sentiment,
                    SUM(mention_count) as total_mentions
                FROM sentiment_data
                GROUP BY symbol_ticker, source
                ORDER BY symbol_ticker, source
            """, conn)

            logger.info(f"\nSentiment Summary by Ticker and Source:")
            logger.info(f"\n{summary_df.to_string(index=False)}")

            # Recent sentiment
            recent_df = pd.read_sql("""
                SELECT
                    symbol_ticker,
                    sentiment_date,
                    source,
                    ROUND(sentiment_score, 3) as sentiment,
                    sentiment_direction,
                    mention_count
                FROM sentiment_data
                ORDER BY sentiment_date DESC
                LIMIT 15
            """, conn)

            logger.info(f"\nRecent Sentiment (last 15 records):")
            logger.info(f"\n{recent_df.to_string(index=False)}")

            conn.close()
            logger.info("Successfully populated sentiment_data table")

        except Exception as e:
            logger.error(f"Error populating sentiment_data table: {str(e)}")
            raise


if __name__ == "__main__":
    # Initialize collector
    collector = SentimentDataCollector()

    print(f"\n{'='*60}")
    print(f"Sentiment Data Collection Script")
    print(f"{'='*60}")
    print(f"Database: {collector.db_path}")
    print(f"Date Range: {collector.start_date.date()} to {collector.end_date.date()}")
    print(f"Data Sources: NewsAPI + Reddit (if configured)")
    print(f"Tracked Tickers: {', '.join(collector.sentiment_tickers)}")

    # Populate database
    print(f"\n{'='*60}")
    print("Collecting sentiment data...")
    print(f"{'='*60}\n")

    collector.populate_sentiment_table(replace=True)

    print(f"\n{'='*60}")
    print("Sentiment data collection complete!")
    print(f"{'='*60}\n")
