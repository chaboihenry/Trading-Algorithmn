"""
Alpaca Tick Data Client

This module fetches historical tick data from Alpaca's API with:
- Automatic pagination (handles 10,000 trade limit per request)
- Rate limiting (stays under 200 requests/minute)
- IEX/SIP feed selection (controlled by config)
- Extended hours support (4 AM - 8 PM ET)

OOP Concepts:
- Class encapsulation: All Alpaca API logic in one place
- Composition: Uses Alpaca's StockHistoricalDataClient class
- Error handling: Gracefully handles API errors and rate limits

Why use a class?
- Manages API client state (credentials, rate limiting)
- Provides clean interface for fetching ticks
- Handles pagination complexity internally
"""

import time
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from functools import wraps
import pytz

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockTradesRequest
from alpaca.data.timeframe import TimeFrame

from config.tick_config import (
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    DATA_FEED,
    FEED_NAME,
    RATE_LIMIT_DELAY,
    COLLECTION_START_HOUR,
    COLLECTION_END_HOUR
)
from config.tick_config import ensure_alpaca_credentials

logger = logging.getLogger(__name__)


def retry_with_backoff(max_retries=3, base_delay=1.0, exceptions=(Exception,)):
    """
    Decorator that retries a function with exponential backoff.

    This handles transient failures (network issues, rate limits, temporary API errors)
    by automatically retrying the function with increasing delays between attempts.

    Backoff Strategy:
    - Attempt 1: immediate
    - Attempt 2: wait base_delay seconds (default 1s)
    - Attempt 3: wait base_delay * 2 seconds (default 2s)
    - Attempt 4: wait base_delay * 4 seconds (default 4s)
    - etc.

    Args:
        max_retries: Maximum number of retry attempts (default 3)
        base_delay: Base delay in seconds for exponential backoff (default 1.0)
        exceptions: Tuple of exception types to catch and retry (default all exceptions)

    Returns:
        Decorated function that retries on failure

    Example:
        @retry_with_backoff(max_retries=5, base_delay=2.0)
        def fetch_data():
            return api.get_data()  # Will retry up to 5 times on failure
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        # Final attempt failed - raise the exception
                        logger.error(f"{func.__name__} failed after {max_retries} retries: {e}")
                        raise

                    # Calculate exponential backoff delay
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)

            # Should never reach here, but just in case
            return func(*args, **kwargs)

        return wrapper
    return decorator


class AlpacaTickClient:
    """
    Fetches historical tick data from Alpaca API.

    This class handles all the complexity of:
    1. API authentication
    2. Pagination (Alpaca returns max 10,000 trades per request)
    3. Rate limiting (stay under 200 requests/minute)
    4. Time zone handling (ET for market hours)
    5. IEX vs SIP feed selection

    Attributes:
        client (StockHistoricalDataClient): Alpaca API client
        feed: Data feed (IEX or SIP)
        rate_limit_delay: Seconds to wait between requests

    Example:
        >>> client = AlpacaTickClient()
        >>> ticks = client.fetch_day_ticks('SPY', datetime(2024, 1, 5))
        >>> print(f"Fetched {len(ticks)} ticks for SPY on 2024-01-05")
    """

    def __init__(self):
        """
        Initialize Alpaca tick client with credentials from config.

        This creates a StockHistoricalDataClient instance that handles
        all communication with Alpaca's API. The client is configured
        to use either IEX (free) or SIP (paid) data based on config.
        """
        ensure_alpaca_credentials()
        if DATA_FEED is None:
            raise ImportError(
                "alpaca-py is not installed. Install requirements.txt before backfill."
            )
        logger.info(f"Initializing Alpaca Tick Client with {FEED_NAME}")

        # Create Alpaca API client
        # StockHistoricalDataClient is a class from alpaca-py SDK
        self.client = StockHistoricalDataClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_API_SECRET
        )

        self.feed = DATA_FEED
        self.rate_limit_delay = RATE_LIMIT_DELAY

        logger.info(f"✓ Client initialized (rate limit delay: {self.rate_limit_delay}s)")

    @retry_with_backoff(max_retries=3, base_delay=2.0, exceptions=(Exception,))
    def fetch_day_ticks(
        self,
        symbol: str,
        date: datetime,
        extended_hours: bool = True
    ) -> List[Dict]:
        """
        Fetch all ticks for a single trading day with automatic pagination.

        Alpaca returns a maximum of 10,000 trades per API call. For highly
        liquid symbols like SPY, you might have 500,000+ trades per day.
        This method handles pagination automatically by making multiple
        requests until all data is fetched.

        Args:
            symbol: Stock ticker (e.g., "SPY", "QQQ")
            date: Trading date (datetime object)
            extended_hours: Include pre-market (4 AM) and after-hours (8 PM)

        Returns:
            List of tick dictionaries:
            [
                {
                    'timestamp': datetime,
                    'price': float,
                    'size': int,
                    'exchange': str,
                    'trade_id': int
                },
                ...
            ]

        Example:
            >>> client = AlpacaTickClient()
            >>> from datetime import datetime
            >>> ticks = client.fetch_day_ticks('SPY', datetime(2024, 1, 5))
            >>> print(f"Fetched {len(ticks):,} ticks")
        """
        # Set up time range for the trading day
        # Extended hours: 4 AM - 8 PM ET
        # Regular hours: 9:30 AM - 4 PM ET
        if extended_hours:
            start_hour = COLLECTION_START_HOUR  # 4 AM
            end_hour = COLLECTION_END_HOUR      # 8 PM
        else:
            start_hour = 9
            end_hour = 16

        # Create timezone-aware datetime objects (Eastern Time)
        et_tz = pytz.timezone('America/New_York')

        # Ensure date is naive (remove any existing timezone info)
        # This handles cases where the caller passes timezone-aware datetimes
        naive_date = date.replace(tzinfo=None) if date.tzinfo else date

        # Start of trading day (4 AM or 9:30 AM ET)
        start = et_tz.localize(naive_date.replace(
            hour=start_hour,
            minute=30 if start_hour == 9 else 0,
            second=0,
            microsecond=0
        ))

        # End of trading day (8 PM or 4 PM ET)
        end = et_tz.localize(naive_date.replace(
            hour=end_hour,
            minute=0,
            second=0,
            microsecond=0
        ))

        logger.info(f"Fetching ticks for {symbol} on {date.date()} ({start.strftime('%H:%M')} - {end.strftime('%H:%M')} ET)")

        all_ticks = []
        page_token = None
        page_num = 1

        while True:
            # Create API request
            # StockTradesRequest is a class from alpaca-py that specifies what data to fetch
            request = StockTradesRequest(
                symbol_or_symbols=symbol,
                start=start,
                end=end,
                feed=self.feed,  # IEX or SIP
                limit=10000,     # Max trades per request
                page_token=page_token  # For pagination
            )

            try:
                # Fetch trades from API
                # Returns a TradesSet object containing a dict of symbol -> list of Trade objects
                logger.debug(f"  Page {page_num}: Requesting up to 10,000 trades...")

                trades_response = self.client.get_stock_trades(request)

                # Extract trades for this symbol
                # trades_response.data is a dict: {symbol: [Trade, Trade, ...]}
                trades_dict = trades_response.data if hasattr(trades_response, 'data') else {}
                trades = trades_dict.get(symbol, [])

                if not trades:
                    logger.debug(f"  Page {page_num}: No more trades")
                    break

                # Convert Trade objects to dictionaries
                for trade in trades:
                    all_ticks.append({
                        'timestamp': trade.timestamp.isoformat(),  # Convert to ISO string
                        'price': float(trade.price),
                        'size': int(trade.size),
                        'exchange': str(trade.exchange) if hasattr(trade, 'exchange') else None,
                        'trade_id': str(trade.id) if hasattr(trade, 'id') else None
                    })

                logger.debug(f"  Page {page_num}: Got {len(trades)} trades (total: {len(all_ticks)})")

                # Check if there's more data (pagination)
                # Alpaca provides a page_token if there's another page
                page_token = getattr(trades_response, 'next_page_token', None)

                if not page_token:
                    # No more pages - we've got all the data
                    logger.debug(f"  No more pages - fetch complete")
                    break

                page_num += 1

                # Rate limiting: Wait before next request
                # This prevents hitting Alpaca's 200 requests/minute limit
                time.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.error(f"Error fetching ticks for {symbol} on page {page_num}: {e}")
                # If we got some data, return it. Otherwise raise the error.
                if all_ticks:
                    logger.warning(f"Returning partial data ({len(all_ticks)} ticks)")
                    break
                else:
                    raise

        # Pagination verification and enhanced logging
        tick_count = len(all_ticks)

        # Log warning for zero ticks (possible data issue)
        if tick_count == 0:
            logger.warning(
                f"⚠️  Zero ticks fetched for {symbol} on {date.date()}. "
                f"This could indicate: (1) market holiday, (2) symbol not traded that day, "
                f"or (3) API data unavailable."
            )
        # Log info for high-volume days
        elif tick_count > 1_000_000:
            logger.info(
                f"📊 High-volume day for {symbol} on {date.date()}: "
                f"{tick_count:,} ticks across {page_num} pages"
            )

        # Pagination verification
        if page_num > 1:
            logger.debug(
                f"Pagination complete: {page_num} pages fetched, "
                f"avg {tick_count // page_num:,} ticks/page"
            )

        logger.info(f"✓ Fetched {tick_count:,} ticks for {symbol} on {date.date()}")
        return all_ticks

    def fetch_ticks_range(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        extended_hours: bool = True
    ) -> List[Dict]:
        """
        Fetch ticks across multiple trading days.

        This is a convenience method that calls fetch_day_ticks() for each
        trading day in the range. It automatically skips weekends.

        Args:
            symbol: Stock ticker
            start_date: First day to fetch
            end_date: Last day to fetch (inclusive)
            extended_hours: Include pre/post market hours

        Returns:
            List of all ticks across the date range

        Example:
            >>> client = AlpacaTickClient()
            >>> from datetime import datetime
            >>> # Fetch 5 days of ticks
            >>> ticks = client.fetch_ticks_range(
            ...     'SPY',
            ...     datetime(2024, 1, 2),
            ...     datetime(2024, 1, 8)
            ... )
            >>> print(f"Fetched {len(ticks):,} ticks across the week")
        """
        logger.info(
            f"Fetching ticks for {symbol} from {start_date.date()} to {end_date.date()}"
        )

        all_ticks = []
        current_date = start_date

        while current_date <= end_date:
            # Skip weekends (Monday=0, Sunday=6)
            if current_date.weekday() < 5:  # Monday to Friday
                try:
                    day_ticks = self.fetch_day_ticks(
                        symbol,
                        current_date,
                        extended_hours
                    )
                    all_ticks.extend(day_ticks)

                except Exception as e:
                    logger.error(f"Failed to fetch {symbol} for {current_date.date()}: {e}")
                    # Continue with next day instead of failing completely

            # Move to next day
            current_date += timedelta(days=1)

        logger.info(
            f"✓ Fetched {len(all_ticks):,} total ticks for {symbol} "
            f"across {(end_date - start_date).days + 1} days"
        )

        return all_ticks

    @retry_with_backoff(max_retries=3, base_delay=1.0, exceptions=(Exception,))
    def test_connection(self) -> bool:
        """
        Test API connection and feed access.

        This verifies:
        1. API credentials are valid
        2. You have access to the selected feed (IEX or SIP)

        Returns:
            bool: True if connection successful, False otherwise

        Example:
            >>> client = AlpacaTickClient()
            >>> if client.test_connection():
            ...     print("✓ Ready to fetch tick data!")
            ... else:
            ...     print("❌ Connection failed - check credentials")
        """
        logger.info(f"Testing connection to Alpaca ({FEED_NAME})...")

        try:
            # Try to fetch a small amount of recent data
            # Use SPY because it's always liquid and available
            today = datetime.now()

            # Go back a few days to ensure we're asking for a valid trading day
            test_date = today - timedelta(days=3)

            request = StockTradesRequest(
                symbol_or_symbols='SPY',
                start=test_date,
                end=test_date + timedelta(days=1),
                feed=self.feed,
                limit=1  # Just fetch 1 trade to test
            )

            response = self.client.get_stock_trades(request)

            trades_dict = response.data if hasattr(response, 'data') else {}
            spy_trades = trades_dict.get('SPY', [])

            if spy_trades:
                logger.info(f"✓ Connection successful! Using {FEED_NAME}")
                logger.info(f"  Test trade: {spy_trades[0]}")
                return True
            else:
                logger.warning("Connection successful but no data returned (may be weekend/holiday)")
                return True  # API works, just no data for that period

        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            if "subscription" in str(e).lower():
                logger.error(
                    f"  It looks like you don't have access to {FEED_NAME}. "
                    f"For SIP data, upgrade to Alpaca Algo Trader Plus."
                )
            return False


if __name__ == "__main__":
    """
    Test the client when run directly.

    Run with: python data/alpaca_tick_client.py
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 80)
    print("ALPACA TICK CLIENT TEST")
    print("=" * 80)

    # Create client
    client = AlpacaTickClient()

    # Test connection
    if not client.test_connection():
        print("\n❌ Connection test failed!")
        print("Check your API credentials and subscription level")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("✓ Alpaca Tick Client is ready!")
    print("=" * 80)
