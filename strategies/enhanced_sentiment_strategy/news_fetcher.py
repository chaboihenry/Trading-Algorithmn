"""
News Data Fetcher for Trading Bot
=================================

This module fetches financial news from multiple sources:
1. Alpaca News API (built-in with trading account)
2. Finnhub (free tier available)
3. Alpha Vantage (free tier available)

The fetcher aggregates news from multiple sources to get better coverage.
"""

import os
import requests
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class NewsArticle:
    """
    Represents a single news article.
    
    Attributes:
    -----------
    headline : str
        The article headline (what we use for sentiment analysis)
    source : str
        Where the news came from (e.g., "Reuters", "Bloomberg")
    published_at : datetime
        When the article was published
    url : str
        Link to the full article
    symbols : List[str]
        Tickers mentioned in the article (e.g., ["AAPL", "MSFT"])
    summary : Optional[str]
        Brief summary if available
    """
    headline: str
    source: str
    published_at: datetime
    url: str
    symbols: List[str]
    summary: Optional[str] = None


class NewsProvider(ABC):
    """
    Abstract base class for news providers.
    
    What is ABC (Abstract Base Class)?
    ----------------------------------
    It's a way to define a "template" that all news providers must follow.
    - You CANNOT create an instance of NewsProvider directly
    - Any class that inherits from NewsProvider MUST implement get_news()
    - This ensures all providers work the same way
    
    Why use this pattern?
    ---------------------
    - Consistency: All providers have the same interface
    - Flexibility: Easy to add new providers
    - Testability: Easy to mock for testing
    """
    
    @abstractmethod
    def get_news(self, symbol: str, start_date: datetime, end_date: datetime) -> List[NewsArticle]:
        """
        Fetch news articles for a symbol within a date range.
        
        Parameters:
        -----------
        symbol : str
            Stock ticker (e.g., "SPY", "AAPL")
        start_date : datetime
            Start of date range
        end_date : datetime
            End of date range
            
        Returns:
        --------
        List[NewsArticle]
            List of news articles
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is configured and available."""
        pass


class AlpacaNewsProvider(NewsProvider):
    """
    Fetches news from Alpaca's News API.
    
    Alpaca provides news as part of their trading API - no extra cost!
    This is the same provider used in the original trading bot.
    
    Setup:
    ------
    1. Create account at https://alpaca.markets/
    2. Get API keys from the dashboard (Paper trading is free)
    3. Set environment variables or pass directly
    
    Attributes:
    -----------
    api_key : PKPRNMTBPHVHTQA4ULVUETELQJ
    api_secret : 4JMSDuHfCuvDGi7v48SEJdtU6VRWwXFgcbNS6SPP8skK
    base_url : https://paper-api.alpaca.markets/v2
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        paper: bool = True
    ):
        """
        Initialize Alpaca news provider.
        
        Parameters:
        -----------
        api_key : str, optional
            Alpaca API key. If not provided, reads from ALPACA_API_KEY env var.
        api_secret : str, optional
            Alpaca API secret. If not provided, reads from ALPACA_API_SECRET env var.
        paper : bool, default=True
            Use paper trading URL (recommended for testing).
        """
        # Try to get credentials from environment if not provided
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.api_secret = api_secret or os.environ.get("ALPACA_API_SECRET")
        
        # Set base URL based on paper/live trading
        if paper:
            self.base_url = "https://paper-api.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"
        
        # News endpoint is always the same
        self.news_url = "https://data.alpaca.markets/v1beta1/news"
        
    def is_available(self) -> bool:
        """Check if Alpaca credentials are configured."""
        return bool(self.api_key and self.api_secret)
    
    def get_news(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        limit: int = 50
    ) -> List[NewsArticle]:
        """
        Fetch news from Alpaca.
        
        Parameters:
        -----------
        symbol : str
            Stock ticker (e.g., "SPY")
        start_date : datetime
            Start of date range
        end_date : datetime
            End of date range
        limit : int, default=50
            Maximum number of articles to fetch
            
        Returns:
        --------
        List[NewsArticle]
            List of news articles
        """
        if not self.is_available():
            print("⚠️ Alpaca credentials not configured")
            return []
        
        # Format dates for API (ISO 8601)
        start_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Set up request headers
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret
        }
        
        # Set up query parameters
        params = {
            "symbols": symbol,
            "start": start_str,
            "end": end_str,
            "limit": limit,
            "sort": "desc"  # Most recent first
        }
        
        try:
            response = requests.get(self.news_url, headers=headers, params=params)
            response.raise_for_status()  # Raise exception for bad status codes
            
            data = response.json()
            articles = []
            
            # Parse each news item
            for item in data.get("news", []):
                article = NewsArticle(
                    headline=item.get("headline", ""),
                    source=item.get("source", "Unknown"),
                    published_at=datetime.fromisoformat(
                        item.get("created_at", "").replace("Z", "+00:00")
                    ),
                    url=item.get("url", ""),
                    symbols=item.get("symbols", []),
                    summary=item.get("summary")
                )
                articles.append(article)
            
            return articles
            
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Alpaca API error: {e}")
            return []


class FinnhubNewsProvider(NewsProvider):
    """
    Fetches news from Finnhub API.
    
    Finnhub offers a generous free tier:
    - 60 API calls/minute
    - Real-time US stock news
    - Company news by symbol
    
    Setup:
    ------
    1. Create free account at https://finnhub.io/
    2. Get API key from dashboard
    3. Set FINNHUB_API_KEY environment variable
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Finnhub news provider.
        
        Parameters:
        -----------
        api_key : str, optional
            Finnhub API key. If not provided, reads from FINNHUB_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("FINNHUB_API_KEY")
        self.base_url = "https://finnhub.io/api/v1"
        
    def is_available(self) -> bool:
        """Check if Finnhub API key is configured."""
        return bool(self.api_key)
    
    def get_news(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[NewsArticle]:
        """
        Fetch news from Finnhub.
        
        Parameters:
        -----------
        symbol : str
            Stock ticker (e.g., "AAPL")
        start_date : datetime
            Start of date range
        end_date : datetime
            End of date range
            
        Returns:
        --------
        List[NewsArticle]
            List of news articles
        """
        if not self.is_available():
            print("⚠️ Finnhub API key not configured")
            return []
        
        # Format dates for Finnhub (YYYY-MM-DD)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Build request URL
        url = f"{self.base_url}/company-news"
        params = {
            "symbol": symbol,
            "from": start_str,
            "to": end_str,
            "token": self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            articles = []
            
            for item in data:
                article = NewsArticle(
                    headline=item.get("headline", ""),
                    source=item.get("source", "Unknown"),
                    published_at=datetime.fromtimestamp(item.get("datetime", 0)),
                    url=item.get("url", ""),
                    symbols=[symbol],  # Finnhub returns news for the requested symbol
                    summary=item.get("summary")
                )
                articles.append(article)
            
            return articles
            
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Finnhub API error: {e}")
            return []


class AlphaVantageNewsProvider(NewsProvider):
    """
    Fetches news from Alpha Vantage API.
    
    Alpha Vantage offers:
    - Free tier: 25 API calls/day
    - News sentiment endpoint with AI-powered sentiment scores
    - Global market coverage
    
    Setup:
    ------
    1. Get free API key at https://www.alphavantage.co/support/#api-key
    2. Set ALPHA_VANTAGE_API_KEY environment variable
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Alpha Vantage news provider.
        
        Parameters:
        -----------
        api_key : str, optional
            Alpha Vantage API key. If not provided, reads from ALPHA_VANTAGE_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY")
        self.base_url = "https://www.alphavantage.co/query"
        
    def is_available(self) -> bool:
        """Check if Alpha Vantage API key is configured."""
        return bool(self.api_key)
    
    def get_news(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[NewsArticle]:
        """
        Fetch news from Alpha Vantage.
        
        Note: Alpha Vantage's free tier has limited calls, so use sparingly.
        
        Parameters:
        -----------
        symbol : str
            Stock ticker (e.g., "AAPL")
        start_date : datetime
            Start of date range (may not be exact, API returns recent news)
        end_date : datetime
            End of date range
            
        Returns:
        --------
        List[NewsArticle]
            List of news articles
        """
        if not self.is_available():
            print("⚠️ Alpha Vantage API key not configured")
            return []
        
        # Build request URL
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "apikey": self.api_key,
            "limit": 50
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            articles = []
            
            # Check for error response
            if "feed" not in data:
                print(f"⚠️ Alpha Vantage response: {data.get('Note', 'No feed found')}")
                return []
            
            for item in data.get("feed", []):
                # Parse the timestamp
                time_str = item.get("time_published", "")
                try:
                    # Format: 20231215T120000
                    pub_time = datetime.strptime(time_str[:15], "%Y%m%dT%H%M%S")
                except ValueError:
                    pub_time = datetime.now()
                
                # Filter by date range
                if start_date <= pub_time <= end_date:
                    article = NewsArticle(
                        headline=item.get("title", ""),
                        source=item.get("source", "Unknown"),
                        published_at=pub_time,
                        url=item.get("url", ""),
                        symbols=[t.get("ticker", "") for t in item.get("ticker_sentiment", [])],
                        summary=item.get("summary")
                    )
                    articles.append(article)
            
            return articles
            
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Alpha Vantage API error: {e}")
            return []


class MultiSourceNewsFetcher:
    """
    Aggregates news from multiple providers.
    
    This class manages multiple news providers and combines their results
    to give you better coverage and reliability.
    
    Why use multiple sources?
    -------------------------
    1. Redundancy: If one source fails, others still work
    2. Coverage: Different sources may have different news
    3. Speed: Some sources are faster than others
    
    Example Usage:
    --------------
    >>> fetcher = MultiSourceNewsFetcher()
    >>> fetcher.add_provider(AlpacaNewsProvider())
    >>> fetcher.add_provider(FinnhubNewsProvider())
    >>> 
    >>> headlines = fetcher.get_headlines("AAPL", days_back=3)
    >>> print(headlines)
    ['Apple announces new iPhone...', 'AAPL stock rises...']
    """
    
    def __init__(self):
        """Initialize with empty list of providers."""
        self.providers: List[NewsProvider] = []
        
    def add_provider(self, provider: NewsProvider) -> "MultiSourceNewsFetcher":
        """
        Add a news provider.
        
        Parameters:
        -----------
        provider : NewsProvider
            A news provider instance
            
        Returns:
        --------
        MultiSourceNewsFetcher
            Returns self for method chaining:
            fetcher.add_provider(X).add_provider(Y)
        """
        if provider.is_available():
            self.providers.append(provider)
            print(f"✓ Added provider: {provider.__class__.__name__}")
        else:
            print(f"⚠️ Provider not available: {provider.__class__.__name__}")
        return self
    
    def get_news(
        self,
        symbol: str,
        days_back: int = 3
    ) -> List[NewsArticle]:
        """
        Get news from all providers.
        
        Parameters:
        -----------
        symbol : str
            Stock ticker (e.g., "SPY")
        days_back : int, default=3
            How many days of news to fetch
            
        Returns:
        --------
        List[NewsArticle]
            Combined list of articles from all providers, sorted by date
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        all_articles = []
        
        for provider in self.providers:
            articles = provider.get_news(symbol, start_date, end_date)
            print(f"   {provider.__class__.__name__}: {len(articles)} articles")
            all_articles.extend(articles)
        
        # Remove duplicates based on headline
        seen_headlines = set()
        unique_articles = []
        for article in all_articles:
            # Normalize headline for comparison
            normalized = article.headline.lower().strip()
            if normalized not in seen_headlines:
                seen_headlines.add(normalized)
                unique_articles.append(article)
        
        # Sort by date (most recent first)
        unique_articles.sort(key=lambda x: x.published_at, reverse=True)
        
        return unique_articles
    
    def get_headlines(
        self,
        symbol: str,
        days_back: int = 3
    ) -> List[str]:
        """
        Get just the headlines (for sentiment analysis).
        
        This is a convenience method that returns only the headline text,
        which is what you need for the FinBERT sentiment analyzer.
        
        Parameters:
        -----------
        symbol : str
            Stock ticker (e.g., "SPY")
        days_back : int, default=3
            How many days of news to fetch
            
        Returns:
        --------
        List[str]
            List of headline strings
        """
        articles = self.get_news(symbol, days_back)
        return [article.headline for article in articles if article.headline]


def create_default_fetcher(
    alpaca_key: Optional[str] = None,
    alpaca_secret: Optional[str] = None,
    finnhub_key: Optional[str] = None,
    alpha_vantage_key: Optional[str] = None
) -> MultiSourceNewsFetcher:
    """
    Create a news fetcher with all available providers.
    
    This is a convenience function that sets up all providers
    based on available API keys (from parameters or environment variables).
    
    Parameters:
    -----------
    alpaca_key : str, optional
        Alpaca API key
    alpaca_secret : str, optional
        Alpaca API secret
    finnhub_key : str, optional
        Finnhub API key
    alpha_vantage_key : str, optional
        Alpha Vantage API key
        
    Returns:
    --------
    MultiSourceNewsFetcher
        Configured fetcher with all available providers
    """
    fetcher = MultiSourceNewsFetcher()
    
    # Add Alpaca (primary provider)
    fetcher.add_provider(AlpacaNewsProvider(
        api_key=alpaca_key,
        api_secret=alpaca_secret,
        paper=True
    ))
    
    # Add Finnhub (good free tier)
    fetcher.add_provider(FinnhubNewsProvider(api_key=finnhub_key))
    
    # Add Alpha Vantage (limited free tier, but good data)
    fetcher.add_provider(AlphaVantageNewsProvider(api_key=alpha_vantage_key))
    
    return fetcher


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING NEWS FETCHERS")
    print("="*60 + "\n")
    
    # Create fetcher (will use available providers based on env vars)
    print("Setting up news fetcher...")
    fetcher = create_default_fetcher()
    
    if len(fetcher.providers) == 0:
        print("\n⚠️ No providers available!")
        print("Please set at least one of these environment variables:")
        print("  - ALPACA_API_KEY and ALPACA_API_SECRET")
        print("  - FINNHUB_API_KEY")
        print("  - ALPHA_VANTAGE_API_KEY")
    else:
        print(f"\n✓ {len(fetcher.providers)} provider(s) available")
        
        # Test fetching news
        print("\nFetching news for SPY (last 3 days)...")
        headlines = fetcher.get_headlines("SPY", days_back=3)
        
        print(f"\nFound {len(headlines)} headlines:")
        for i, headline in enumerate(headlines[:5], 1):
            print(f"  {i}. {headline[:80]}...")