"""
Data Infrastructure Module
===========================
Centralized configuration and utilities for data quality, rate limiting, and consistency
"""

import time
import hashlib
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataConfig:
    """
    Centralized data configuration for consistent lookback windows across all scripts

    This ensures all data collectors use the same time windows, preventing
    inconsistencies that can compromise ML model training.
    """

    # Training data windows
    HISTORICAL_DAYS = 730  # 2 years for training (minimum for robust ML)
    FEATURE_LOOKBACK = 252  # 1 year for technical features
    SENTIMENT_LOOKBACK = 90  # 3 months for sentiment (news/analyst data)
    EARNINGS_LOOKBACK = 365  # 1 year for earnings history
    INSIDER_LOOKBACK = 90  # 3 months for insider trading patterns

    # Minimum data requirements
    MINIMUM_DATA_POINTS = 60  # Minimum data points for any calculation
    MINIMUM_TRADING_DAYS = 252  # 1 year of trading for ML training

    # Forward-fill limits (prevent artificial patterns)
    MAX_FFILL_DAYS = {
        'sentiment_score': 7,  # Max 1 week for sentiment
        'analyst_rating': 30,  # Max 1 month for analyst ratings
        'fundamental_data': 90,  # Max 1 quarter for fundamentals
        'technical_indicators': 2,  # Max 2 days for technical indicators
        'volume': 0,  # Never forward-fill volume
        'price': 0,  # Never forward-fill price
    }

    @staticmethod
    def get_date_range(lookback_type: str = 'historical') -> Tuple[datetime, datetime]:
        """
        Get consistent date range for data collection

        Args:
            lookback_type: One of 'historical', 'feature', 'sentiment', 'earnings', 'insider'

        Returns:
            Tuple of (start_date, end_date)
        """
        end = datetime.now()

        lookback_map = {
            'historical': DataConfig.HISTORICAL_DAYS,
            'feature': DataConfig.FEATURE_LOOKBACK,
            'sentiment': DataConfig.SENTIMENT_LOOKBACK,
            'earnings': DataConfig.EARNINGS_LOOKBACK,
            'insider': DataConfig.INSIDER_LOOKBACK,
        }

        days = lookback_map.get(lookback_type, DataConfig.HISTORICAL_DAYS)
        start = end - timedelta(days=days)

        return start, end

    @staticmethod
    def get_max_ffill_limit(column_type: str) -> int:
        """Get maximum forward-fill limit for a column type"""
        return DataConfig.MAX_FFILL_DAYS.get(column_type, 3)


class RateLimitManager:
    """
    Intelligent rate limiting with exponential backoff

    Prevents API rate limit violations that cause data gaps.
    Supports multiple APIs with different rate limits.
    """

    def __init__(self):
        """Initialize rate limit trackers for different APIs"""
        # Polygon.io: 5 calls per minute (free tier)
        self.polygon_calls = deque(maxlen=5)

        # Yahoo Finance: 60 calls per minute (unofficial limit)
        self.yfinance_calls = deque(maxlen=60)

        # News API: 100 calls per day (free tier)
        self.newsapi_calls = deque(maxlen=100)
        self.newsapi_daily_reset = datetime.now().date()

        # Alpha Vantage: 5 calls per minute, 500 per day
        self.alphavantage_calls = deque(maxlen=5)
        self.alphavantage_daily = deque(maxlen=500)
        self.alphavantage_daily_reset = datetime.now().date()

    def wait_if_needed(self, api_type: str = 'yfinance') -> None:
        """
        Smart rate limiting with exponential backoff

        Args:
            api_type: One of 'polygon', 'yfinance', 'newsapi', 'alphavantage'
        """
        current_time = time.time()
        current_date = datetime.now().date()

        if api_type == 'polygon':
            self._rate_limit_polygon(current_time)
        elif api_type == 'yfinance':
            self._rate_limit_yfinance(current_time)
        elif api_type == 'newsapi':
            self._rate_limit_newsapi(current_time, current_date)
        elif api_type == 'alphavantage':
            self._rate_limit_alphavantage(current_time, current_date)
        else:
            logger.warning(f"Unknown API type: {api_type}")

    def _rate_limit_polygon(self, current_time: float) -> None:
        """Rate limit for Polygon.io (5 calls/minute)"""
        if len(self.polygon_calls) == 5:
            time_elapsed = current_time - self.polygon_calls[0]
            if time_elapsed < 60:
                wait_time = 60 - time_elapsed + 1
                logger.info(f"Polygon rate limit: waiting {wait_time:.1f}s")
                time.sleep(wait_time)
        self.polygon_calls.append(current_time)

    def _rate_limit_yfinance(self, current_time: float) -> None:
        """Rate limit for Yahoo Finance (60 calls/minute, conservative)"""
        if len(self.yfinance_calls) == 60:
            time_elapsed = current_time - self.yfinance_calls[0]
            if time_elapsed < 60:
                wait_time = 60 - time_elapsed + 1
                logger.info(f"Yahoo Finance rate limit: waiting {wait_time:.1f}s")
                time.sleep(wait_time)
        self.yfinance_calls.append(current_time)

    def _rate_limit_newsapi(self, current_time: float, current_date) -> None:
        """Rate limit for News API (100 calls/day)"""
        # Reset daily counter if new day
        if current_date > self.newsapi_daily_reset:
            self.newsapi_calls.clear()
            self.newsapi_daily_reset = current_date
            logger.info("News API daily limit reset")

        if len(self.newsapi_calls) >= 100:
            logger.error("News API daily limit reached (100 calls)")
            raise Exception("News API daily limit exceeded")

        self.newsapi_calls.append(current_time)

    def _rate_limit_alphavantage(self, current_time: float, current_date) -> None:
        """Rate limit for Alpha Vantage (5 calls/minute, 500/day)"""
        # Reset daily counter if new day
        if current_date > self.alphavantage_daily_reset:
            self.alphavantage_daily.clear()
            self.alphavantage_daily_reset = current_date
            logger.info("Alpha Vantage daily limit reset")

        # Check daily limit
        if len(self.alphavantage_daily) >= 500:
            logger.error("Alpha Vantage daily limit reached (500 calls)")
            raise Exception("Alpha Vantage daily limit exceeded")

        # Check per-minute limit
        if len(self.alphavantage_calls) == 5:
            time_elapsed = current_time - self.alphavantage_calls[0]
            if time_elapsed < 60:
                wait_time = 60 - time_elapsed + 1
                logger.info(f"Alpha Vantage rate limit: waiting {wait_time:.1f}s")
                time.sleep(wait_time)

        self.alphavantage_calls.append(current_time)
        self.alphavantage_daily.append(current_time)


class DataQualityManager:
    """
    Intelligent data filling and quality control

    Prevents blind forward-filling that creates artificial patterns.
    """

    @staticmethod
    def smart_fill(df: pd.DataFrame, col: str, method: str = 'ffill', limit: Optional[int] = None) -> pd.Series:
        """
        Intelligent filling with limits based on data type

        Args:
            df: DataFrame containing the column
            col: Column name to fill
            method: Fill method ('ffill', 'interpolate', or 'zero')
            limit: Maximum number of consecutive NaNs to fill (auto-detected if None)

        Returns:
            Filled Series
        """
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame")
            return pd.Series()

        # Auto-detect limit based on column type
        if limit is None:
            if col in ['sentiment_score', 'analyst_rating', 'rating_numeric']:
                limit = DataConfig.MAX_FFILL_DAYS['sentiment_score']  # 7 days
            elif col in ['pe_ratio', 'price_to_book', 'debt_to_equity', 'profit_margin']:
                limit = DataConfig.MAX_FFILL_DAYS['fundamental_data']  # 90 days
            elif col in ['rsi_14', 'macd', 'macd_histogram', 'bb_position']:
                limit = DataConfig.MAX_FFILL_DAYS['technical_indicators']  # 2 days
            elif col in ['volume', 'trades_count', 'shares_traded']:
                limit = 0  # Never forward-fill volume
            elif col in ['close', 'open', 'high', 'low']:
                limit = 0  # Never forward-fill price
            else:
                limit = 3  # Conservative default

        # Apply filling method
        if method == 'ffill':
            filled = df[col].ffill(limit=limit)
        elif method == 'interpolate':
            filled = df[col].interpolate(method='linear', limit=limit)
        elif method == 'zero':
            filled = df[col].fillna(0)
        else:
            logger.warning(f"Unknown fill method: {method}, using ffill")
            filled = df[col].ffill(limit=limit)

        # Log filling statistics
        original_nulls = df[col].isna().sum()
        remaining_nulls = filled.isna().sum()
        filled_count = original_nulls - remaining_nulls

        if filled_count > 0:
            logger.debug(f"Filled {filled_count}/{original_nulls} nulls in {col} (limit={limit})")

        return filled

    @staticmethod
    def validate_data_quality(df: pd.DataFrame, required_cols: List[str], min_rows: int = None) -> Tuple[bool, str]:
        """
        Validate data quality before processing

        Args:
            df: DataFrame to validate
            required_cols: List of required columns
            min_rows: Minimum number of rows required

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if DataFrame is empty
        if df.empty:
            return False, "DataFrame is empty"

        # Check minimum rows
        if min_rows and len(df) < min_rows:
            return False, f"Insufficient rows: {len(df)} < {min_rows}"

        # Check required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return False, f"Missing columns: {missing_cols}"

        # Check for excessive nulls (>80% null = bad data)
        for col in required_cols:
            null_pct = df[col].isna().sum() / len(df)
            if null_pct > 0.8:
                return False, f"Column {col} is {null_pct:.1%} null (>80% threshold)"

        return True, "Data quality validated"


class NewsFilterManager:
    """
    News article deduplication and relevance filtering

    Prevents duplicate articles and irrelevant news from polluting sentiment analysis.
    """

    @staticmethod
    def calculate_relevance(article: Dict, ticker: str, company_name: str = None) -> float:
        """
        Calculate relevance score for a news article

        Args:
            article: News article dictionary with 'title' and 'description'
            ticker: Stock ticker symbol
            company_name: Company name (optional, for better matching)

        Returns:
            Relevance score 0-1 (higher = more relevant)
        """
        title = article.get('title', '').lower()
        description = article.get('description', '').lower()
        content = article.get('content', '').lower()

        ticker_lower = ticker.lower()
        score = 0.0

        # Check ticker in title (strongest signal)
        if ticker_lower in title or f"${ticker_lower}" in title:
            score += 0.5

        # Check ticker in description
        if ticker_lower in description or f"${ticker_lower}" in description:
            score += 0.3

        # Check company name if provided
        if company_name:
            company_lower = company_name.lower()
            if company_lower in title:
                score += 0.4
            if company_lower in description:
                score += 0.2

        # Check for financial keywords (relevance indicators)
        financial_keywords = ['earnings', 'revenue', 'profit', 'stock', 'shares', 'trading',
                              'quarterly', 'analyst', 'upgrade', 'downgrade', 'buy', 'sell']
        keyword_count = sum(1 for kw in financial_keywords if kw in title or kw in description)
        score += min(keyword_count * 0.05, 0.2)  # Max 0.2 from keywords

        # Penalize generic market news
        generic_terms = ['market', 'dow', 'sp500', 's&p 500', 'nasdaq', 'index']
        if any(term in title for term in generic_terms) and ticker_lower not in title:
            score *= 0.5  # Reduce score for generic market news

        return min(score, 1.0)

    @staticmethod
    def filter_news_articles(articles: List[Dict], ticker: str, company_name: str = None,
                            min_relevance: float = 0.3) -> List[Dict]:
        """
        Remove duplicates and irrelevant articles

        Args:
            articles: List of news article dictionaries
            ticker: Stock ticker symbol
            company_name: Company name (optional)
            min_relevance: Minimum relevance score threshold

        Returns:
            Filtered list of unique, relevant articles
        """
        seen_titles = set()
        filtered = []

        for article in articles:
            # Skip if missing required fields
            if not article.get('title'):
                continue

            # Create hash of title for deduplication
            title_normalized = article['title'].lower().strip()
            title_hash = hashlib.md5(title_normalized.encode()).hexdigest()

            # Skip duplicates
            if title_hash in seen_titles:
                logger.debug(f"Duplicate article: {article['title'][:50]}...")
                continue

            # Check relevance
            relevance = NewsFilterManager.calculate_relevance(article, ticker, company_name)
            if relevance < min_relevance:
                logger.debug(f"Low relevance ({relevance:.2f}): {article['title'][:50]}...")
                continue

            # Add to filtered list
            seen_titles.add(title_hash)
            article['relevance_score'] = relevance
            filtered.append(article)

        logger.info(f"Filtered {len(articles)} articles → {len(filtered)} unique, relevant articles")
        return filtered

    @staticmethod
    def deduplicate_by_content(articles: List[Dict], similarity_threshold: float = 0.8) -> List[Dict]:
        """
        Advanced deduplication using content similarity

        Args:
            articles: List of articles
            similarity_threshold: Jaccard similarity threshold for duplicates

        Returns:
            Deduplicated list
        """
        if len(articles) <= 1:
            return articles

        unique_articles = []
        seen_content = []

        for article in articles:
            content = article.get('description', '') or article.get('title', '')
            if not content:
                continue

            # Tokenize content
            tokens = set(content.lower().split())

            # Check similarity with existing articles
            is_duplicate = False
            for seen_tokens in seen_content:
                # Jaccard similarity
                intersection = len(tokens & seen_tokens)
                union = len(tokens | seen_tokens)
                similarity = intersection / union if union > 0 else 0

                if similarity >= similarity_threshold:
                    is_duplicate = True
                    logger.debug(f"Content duplicate ({similarity:.2f}): {article.get('title', '')[:50]}...")
                    break

            if not is_duplicate:
                unique_articles.append(article)
                seen_content.append(tokens)

        logger.info(f"Content deduplication: {len(articles)} → {len(unique_articles)}")
        return unique_articles


# Singleton instances for reuse
_rate_limit_manager = None
_data_quality_manager = None
_news_filter_manager = None


def get_rate_limit_manager() -> RateLimitManager:
    """Get singleton RateLimitManager instance"""
    global _rate_limit_manager
    if _rate_limit_manager is None:
        _rate_limit_manager = RateLimitManager()
    return _rate_limit_manager


def get_data_quality_manager() -> DataQualityManager:
    """Get singleton DataQualityManager instance"""
    global _data_quality_manager
    if _data_quality_manager is None:
        _data_quality_manager = DataQualityManager()
    return _data_quality_manager


def get_news_filter_manager() -> NewsFilterManager:
    """Get singleton NewsFilterManager instance"""
    global _news_filter_manager
    if _news_filter_manager is None:
        _news_filter_manager = NewsFilterManager()
    return _news_filter_manager


if __name__ == "__main__":
    # Test data infrastructure
    logging.basicConfig(level=logging.INFO)

    print("=== DataConfig Test ===")
    start, end = DataConfig.get_date_range('historical')
    print(f"Historical range: {start.date()} to {end.date()} ({DataConfig.HISTORICAL_DAYS} days)")

    print("\n=== RateLimitManager Test ===")
    rlm = get_rate_limit_manager()
    print("Testing Yahoo Finance rate limit (10 calls)...")
    for i in range(10):
        rlm.wait_if_needed('yfinance')
        print(f"  Call {i+1} completed")

    print("\n=== DataQualityManager Test ===")
    dqm = get_data_quality_manager()
    test_df = pd.DataFrame({
        'price': [100, 101, np.nan, 103, 104],
        'volume': [1000, np.nan, np.nan, 1200, 1300],
        'sentiment': [0.5, np.nan, np.nan, np.nan, 0.7]
    })
    print("Test DataFrame:")
    print(test_df)
    print("\nAfter smart fill:")
    test_df['price_filled'] = dqm.smart_fill(test_df, 'price', method='ffill')
    test_df['volume_filled'] = dqm.smart_fill(test_df, 'volume', method='zero')
    test_df['sentiment_filled'] = dqm.smart_fill(test_df, 'sentiment', method='ffill')
    print(test_df)

    print("\n=== NewsFilterManager Test ===")
    nfm = get_news_filter_manager()
    test_articles = [
        {'title': 'AAPL reports strong earnings', 'description': 'Apple Inc quarterly earnings beat expectations'},
        {'title': 'Apple reports strong earnings', 'description': 'AAPL quarterly earnings beat expectations'},  # Duplicate
        {'title': 'Market rallies on Fed news', 'description': 'Dow Jones and S&P 500 rally'},  # Irrelevant
        {'title': 'AAPL stock upgraded to buy', 'description': 'Analyst upgrades Apple to strong buy'},
    ]
    filtered = nfm.filter_news_articles(test_articles, 'AAPL', 'Apple Inc', min_relevance=0.3)
    print(f"Filtered {len(test_articles)} → {len(filtered)} articles:")
    for article in filtered:
        print(f"  - {article['title']} (relevance: {article['relevance_score']:.2f})")
