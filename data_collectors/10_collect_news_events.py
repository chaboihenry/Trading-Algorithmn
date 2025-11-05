import sqlite3
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List, Dict
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpandedNewsEventsCollector:
    """
    Expanded news events collector with broader search and historical coverage

    Database: /Volumes/Vault/85_assets_prediction.db
    Table: news_events
    Data Source: NewsAPI with everything endpoint for historical data
    Improvements:
        - Uses 'everything' endpoint instead of 'top-headlines'
        - Searches multiple sources
        - Collects up to 30 days of history per ticker
        - More flexible keyword matching
    """

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db") -> None:
        """Initialize the collector with database path and API keys"""
        self.db_path = db_path

        # NewsAPI key
        self.newsapi_key = "e78eec9cf81b49c89400158d0975ae3e"
        self.newsapi_everything_url = "https://newsapi.org/v2/everything"

        # Event keywords for categorization
        self.event_categories = {
            'Earnings': ['earnings', 'revenue', 'profit', 'eps', 'quarterly', 'results', 'guidance', 'beat', 'miss'],
            'M&A': ['merger', 'acquisition', 'buyout', 'takeover', 'deal', 'acquire'],
            'Product': ['launch', 'product', 'release', 'unveil', 'announce', 'new'],
            'Regulatory': ['fda', 'sec', 'regulation', 'lawsuit', 'fine', 'investigation', 'approval'],
            'Executive': ['ceo', 'cfo', 'executive', 'management', 'resignation', 'hire', 'appoint'],
            'Partnership': ['partnership', 'collaboration', 'joint venture', 'agreement', 'partner'],
            'Financial': ['dividend', 'buyback', 'debt', 'financing', 'ipo', 'offering', 'stock split']
        }

        logger.info(f"Initialized ExpandedNewsEventsCollector")

    def _get_db_connection(self) -> sqlite3.Connection:
        """Create and return database connection"""
        return sqlite3.connect(self.db_path)

    def _get_all_tickers(self) -> List[str]:
        """Get all 85 tickers from assets table"""
        try:
            conn = self._get_db_connection()
            query = "SELECT symbol_ticker FROM assets ORDER BY symbol_ticker"
            df = pd.read_sql(query, conn)
            conn.close()
            tickers = df['symbol_ticker'].tolist()
            logger.info(f"Retrieved {len(tickers)} tickers from assets table")
            return tickers
        except Exception as e:
            logger.error(f"Error retrieving tickers: {str(e)}")
            raise

    def _categorize_event(self, title: str, description: str) -> tuple:
        """Categorize news event based on keywords"""
        text = (title + " " + description).lower()

        matched_category = None
        for category, keywords in self.event_categories.items():
            if any(keyword in text for keyword in keywords):
                matched_category = category
                break

        event_type = "News"
        magnitude = 5.0

        # High impact keywords increase magnitude
        if any(word in text for word in ['major', 'significant', 'massive', 'record', 'historic', 'breakthrough']):
            magnitude = 8.0
        elif any(word in text for word in ['strong', 'beats', 'exceeds', 'surges', 'jumps', 'soars']):
            magnitude = 7.0
        elif any(word in text for word in ['weak', 'misses', 'falls', 'declines', 'cuts', 'plunges']):
            magnitude = 6.0

        return (matched_category or 'General', event_type, magnitude)

    def _calculate_sentiment(self, title: str, description: str) -> float:
        """Simple sentiment scoring based on keywords (-1 to 1)"""
        text = (title + " " + description).lower()

        positive_words = ['beats', 'exceeds', 'rises', 'gains', 'strong', 'growth', 'success',
                         'positive', 'profit', 'surge', 'rally', 'upgrade', 'outperform', 'breakthrough']
        negative_words = ['misses', 'falls', 'weak', 'loss', 'decline', 'cut', 'negative',
                         'lawsuit', 'investigation', 'downgrade', 'warning', 'concern', 'struggle']

        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)

        total = positive_count + negative_count
        if total == 0:
            return 0.0

        sentiment = (positive_count - negative_count) / total
        return round(sentiment, 2)

    def collect_news_for_ticker(self, ticker: str, days_back: int = 30) -> List[Dict]:
        """
        Collect news events for a single ticker using everything endpoint

        Args:
            ticker: Stock ticker symbol
            days_back: Number of days to look back (max 30 for free tier)

        Returns:
            List of news event dictionaries
        """
        try:
            logger.info(f"Fetching news for {ticker}...")

            # Calculate date range (NewsAPI free tier: last 30 days)
            to_date = datetime.now()
            from_date = to_date - timedelta(days=min(days_back, 30))

            # Build search query (more comprehensive)
            query = f'"{ticker}" OR "${ticker}" OR "{ticker} stock"'

            params = {
                'q': query,
                'from': from_date.strftime('%Y-%m-%d'),
                'to': to_date.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'relevancy',
                'pageSize': 100,  # Max allowed
                'apiKey': self.newsapi_key
            }

            response = requests.get(self.newsapi_everything_url, params=params, timeout=30)

            if response.status_code != 200:
                logger.warning(f"NewsAPI returned status {response.status_code} for {ticker}")
                return []

            data = response.json()

            if data.get('status') != 'ok' or not data.get('articles'):
                logger.warning(f"No articles found for {ticker}")
                return []

            articles = data['articles']
            events = []

            for article in articles:
                title = article.get('title', '')
                description = article.get('description', '') or ''
                published_at = article.get('publishedAt', '')
                source = article.get('source', {}).get('name', 'Unknown')

                if not title or not published_at:
                    continue

                # Filter out removed/deleted articles
                if '[Removed]' in title or '[Deleted]' in title:
                    continue

                # Parse date
                try:
                    event_date = datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d')
                except:
                    try:
                        event_date = datetime.fromisoformat(published_at.replace('Z', '+00:00')).strftime('%Y-%m-%d')
                    except:
                        continue

                # Categorize and score
                category, event_type, magnitude = self._categorize_event(title, description)
                sentiment = self._calculate_sentiment(title, description)

                # Determine expected impact
                if magnitude >= 7.5:
                    expected_impact = 'High'
                elif magnitude >= 5.5:
                    expected_impact = 'Medium'
                else:
                    expected_impact = 'Low'

                event = {
                    'symbol_ticker': ticker,
                    'event_date': event_date,
                    'event_type': event_type,
                    'event_category': category,
                    'event_magnitude': magnitude,
                    'event_description': title[:200],
                    'event_source': source,
                    'sentiment_score': sentiment,
                    'expected_impact': expected_impact,
                    'similar_historical_events': None,
                    'avg_price_reaction_historical': None
                }

                events.append(event)

            logger.info(f"✓ Collected {len(events)} news events for {ticker}")
            return events

        except Exception as e:
            logger.error(f"Error collecting news for {ticker}: {str(e)}")
            return []

    def collect_all_news_events(self, days_back: int = 30) -> pd.DataFrame:
        """
        Collect news events for all tickers

        Args:
            days_back: Days of history to collect

        Returns:
            DataFrame with all news events
        """
        logger.info(f"Starting expanded news collection (last {days_back} days)")

        tickers = self._get_all_tickers()
        all_events = []
        success_count = 0

        for idx, ticker in enumerate(tickers, 1):
            try:
                events = self.collect_news_for_ticker(ticker, days_back)

                if events:
                    all_events.extend(events)
                    success_count += 1

                # Log progress every 10 tickers
                if idx % 10 == 0:
                    logger.info(f"Progress: {idx}/{len(tickers)} tickers processed ({len(all_events)} events collected)")

                # Rate limiting (free tier: 100 requests/day, so ~1 per ticker max)
                time.sleep(2)

            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")

        if all_events:
            events_df = pd.DataFrame(all_events)
            # Remove duplicates based on ticker, date, and description
            events_df = events_df.drop_duplicates(subset=['symbol_ticker', 'event_date', 'event_description'])
            events_df = events_df.sort_values(['symbol_ticker', 'event_date'])

            logger.info(f"\n{'='*60}")
            logger.info(f"Collection Summary:")
            logger.info(f"  Total tickers: {len(tickers)}")
            logger.info(f"  Tickers with events: {success_count}")
            logger.info(f"  Total events collected: {len(events_df)}")
            logger.info(f"{'='*60}\n")

            return events_df
        else:
            logger.warning("No news events collected")
            return pd.DataFrame()

    def populate_news_events_table(self, days_back: int = 30, replace: bool = False) -> None:
        """
        Populate the news_events table with expanded historical data

        Args:
            days_back: Days of history to collect
            replace: If True, replace existing data. If False, append only new events.
        """
        logger.info("Starting expanded news_events table population")

        try:
            # Collect all news events
            events_df = self.collect_all_news_events(days_back)

            if events_df.empty:
                logger.error("No news events collected. Aborting database insertion.")
                return

            # Connect to database
            conn = self._get_db_connection()
            cursor = conn.cursor()

            # Check existing data
            cursor.execute("SELECT COUNT(*) FROM news_events")
            existing_count = cursor.fetchone()[0]

            if existing_count > 0 and not replace:
                logger.info(f"Found {existing_count} existing events. Appending new events only.")

                # Get existing event keys (ticker + date + description)
                existing_df = pd.read_sql(
                    "SELECT symbol_ticker, event_date, event_description FROM news_events",
                    conn
                )
                existing_keys = set(
                    zip(existing_df['symbol_ticker'],
                        existing_df['event_date'],
                        existing_df['event_description'])
                )

                # Filter to only new events
                new_events_mask = ~events_df.apply(
                    lambda row: (row['symbol_ticker'], row['event_date'], row['event_description']) in existing_keys,
                    axis=1
                )
                new_events_df = events_df[new_events_mask]

                if len(new_events_df) > 0:
                    new_events_df.to_sql('news_events', conn, if_exists='append', index=False)
                    logger.info(f"Inserted {len(new_events_df)} new events")
                else:
                    logger.info("No new events to insert")
            else:
                # Replace or insert all
                if_exists = 'replace' if replace else 'append'
                events_df.to_sql('news_events', conn, if_exists=if_exists, index=False)
                logger.info(f"{'Replaced with' if replace else 'Inserted'} {len(events_df)} events")

            # Verify insertion and show summary
            cursor.execute("SELECT COUNT(*) FROM news_events")
            final_count = cursor.fetchone()[0]
            logger.info(f"news_events table now contains {final_count} total events")

            # Show summary by ticker
            ticker_summary_df = pd.read_sql("""
                SELECT
                    symbol_ticker,
                    COUNT(*) as event_count,
                    MIN(event_date) as earliest_event,
                    MAX(event_date) as latest_event
                FROM news_events
                GROUP BY symbol_ticker
                ORDER BY event_count DESC
                LIMIT 15
            """, conn)

            logger.info(f"\nTop Tickers by Event Count:")
            logger.info(f"\n{ticker_summary_df.to_string(index=False)}")

            # Show summary by category
            category_df = pd.read_sql("""
                SELECT
                    event_category,
                    expected_impact,
                    COUNT(*) as count,
                    ROUND(AVG(event_magnitude), 2) as avg_magnitude,
                    ROUND(AVG(sentiment_score), 2) as avg_sentiment
                FROM news_events
                GROUP BY event_category, expected_impact
                ORDER BY count DESC
                LIMIT 15
            """, conn)

            logger.info(f"\nEvents by Category and Impact:")
            logger.info(f"\n{category_df.to_string(index=False)}")

            conn.close()
            logger.info("Successfully populated news_events table")

        except Exception as e:
            logger.error(f"Error populating news_events table: {str(e)}")
            raise


if __name__ == "__main__":
    # Initialize collector
    collector = ExpandedNewsEventsCollector()

    print(f"\n{'='*60}")
    print(f"Expanded News Events Collection Script")
    print(f"{'='*60}")
    print(f"Database: {collector.db_path}")
    print(f"Data Source: NewsAPI (everything endpoint)")
    print(f"Coverage: Last 30 days for all 85 assets")

    # Populate database
    print(f"\n{'='*60}")
    print("Collecting expanded news events...")
    print(f"{'='*60}\n")

    collector.populate_news_events_table(days_back=30, replace=False)

    print(f"\n{'='*60}")
    print("Expanded news events collection complete!")
    print(f"{'='*60}\n")
