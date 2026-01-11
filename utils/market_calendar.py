"""
Market Calendar Utility

Provides accurate trading day information using Alpaca's Calendar API.
No hardcoded dates - always fetches current exchange calendar.

Why use Alpaca's API instead of hardcoding?
1. Always accurate - NYSE updates their calendar, Alpaca reflects it
2. Handles edge cases - early closes, special holidays
3. No maintenance - works for any year without code changes

OOP Concepts:
- Singleton pattern: One calendar instance shared across the app
- Caching: Store results to avoid repeated API calls
- Lazy loading: Only fetch when needed
"""

import logging
from datetime import datetime, date, timedelta
from typing import List, Optional, Set
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# US Eastern timezone - the market's timezone
MARKET_TZ = ZoneInfo("America/New_York")

# Market hours in Eastern Time
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# Extended hours (if using IEX with extended data)
EXTENDED_OPEN_HOUR = 4
EXTENDED_CLOSE_HOUR = 20


class MarketCalendar:
    """
    Provides market calendar information using Alpaca's API.

    This class answers questions like:
    - Is the market open right now?
    - Is tomorrow a trading day?
    - What are the trading days between two dates?

    Uses caching to avoid hitting the API repeatedly.

    Attributes:
        _trading_days_cache: Set of dates that are trading days
        _cache_start: Start of cached date range
        _cache_end: End of cached date range
    """

    # Class-level cache (shared across all instances - singleton pattern)
    _trading_days_cache: Set[date] = set()
    _cache_start: Optional[date] = None
    _cache_end: Optional[date] = None
    _api_client = None

    def __init__(self):
        """Initialize the calendar (lazy - doesn't fetch until needed)."""
        self._init_api_client()

    def _init_api_client(self):
        """Initialize Alpaca API client if not already done."""
        if MarketCalendar._api_client is not None:
            return

        from config.tick_config import ALPACA_API_KEY, ALPACA_API_SECRET
        from alpaca.trading.client import TradingClient

        MarketCalendar._api_client = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_API_SECRET,
            paper=True  # Doesn't matter for calendar
        )
        logger.debug("Alpaca trading client initialized for calendar")

    def _ensure_cache(self, start_date: date, end_date: date):
        """
        Ensure the cache covers the requested date range.

        This method fetches trading days from Alpaca if needed.
        The cache expands to cover any requested range.

        Args:
            start_date: Start of range to ensure
            end_date: End of range to ensure
        """
        # Check if cache already covers this range
        if (MarketCalendar._cache_start is not None and
            MarketCalendar._cache_end is not None and
            start_date >= MarketCalendar._cache_start and
            end_date <= MarketCalendar._cache_end):
            return  # Cache hit - no fetch needed

        # Expand range to cache more (avoid repeated API calls)
        # Cache 1 year before and after the requested range
        fetch_start = min(
            start_date - timedelta(days=365),
            MarketCalendar._cache_start or start_date
        )
        fetch_end = max(
            end_date + timedelta(days=365),
            MarketCalendar._cache_end or end_date
        )

        logger.info(f"Fetching market calendar: {fetch_start} to {fetch_end}")

        # Fetch from Alpaca
        from alpaca.trading.requests import GetCalendarRequest

        request = GetCalendarRequest(
            start=fetch_start,
            end=fetch_end
        )

        calendar_days = MarketCalendar._api_client.get_calendar(request)

        # Update cache
        for day in calendar_days:
            # day.date is already a date object
            MarketCalendar._trading_days_cache.add(day.date)

        MarketCalendar._cache_start = fetch_start
        MarketCalendar._cache_end = fetch_end

        logger.info(f"Cached {len(calendar_days)} trading days")

    def is_trading_day(self, check_date: date) -> bool:
        """
        Check if a specific date is a trading day.

        Args:
            check_date: The date to check (date object or datetime)

        Returns:
            True if markets are open on this date

        Example:
            >>> cal = MarketCalendar()
            >>> cal.is_trading_day(date(2024, 12, 25))  # Christmas
            False
            >>> cal.is_trading_day(date(2024, 12, 26))  # Regular Thursday
            True
        """
        if isinstance(check_date, datetime):
            check_date = check_date.date()

        # Quick check: weekends are never trading days
        if check_date.weekday() >= 5:
            return False

        # Ensure cache covers this date
        self._ensure_cache(check_date, check_date)

        return check_date in MarketCalendar._trading_days_cache

    def get_trading_days(
        self,
        start_date: date,
        end_date: date
    ) -> List[date]:
        """
        Get all trading days in a date range.

        Args:
            start_date: Start of range (inclusive)
            end_date: End of range (inclusive)

        Returns:
            List of trading day dates, sorted chronologically

        Example:
            >>> cal = MarketCalendar()
            >>> days = cal.get_trading_days(date(2024, 12, 23), date(2024, 12, 27))
            >>> # Returns [Dec 23, Dec 24, Dec 26, Dec 27] - skips Christmas
        """
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()

        self._ensure_cache(start_date, end_date)

        trading_days = [
            d for d in MarketCalendar._trading_days_cache
            if start_date <= d <= end_date
        ]

        return sorted(trading_days)

    def is_market_open(self) -> bool:
        """
        Check if the market is currently open.

        Returns:
            True if market is open right now
        """
        now_et = datetime.now(MARKET_TZ)

        # Check if today is a trading day
        if not self.is_trading_day(now_et.date()):
            return False

        # Check if within market hours
        market_open = now_et.replace(
            hour=MARKET_OPEN_HOUR,
            minute=MARKET_OPEN_MINUTE,
            second=0,
            microsecond=0
        )
        market_close = now_et.replace(
            hour=MARKET_CLOSE_HOUR,
            minute=MARKET_CLOSE_MINUTE,
            second=0,
            microsecond=0
        )

        return market_open <= now_et <= market_close

    def next_trading_day(self, from_date: Optional[date] = None) -> date:
        """
        Get the next trading day after a given date.

        Args:
            from_date: Starting date (default: today)

        Returns:
            The next trading day
        """
        if from_date is None:
            from_date = datetime.now(MARKET_TZ).date()
        elif isinstance(from_date, datetime):
            from_date = from_date.date()

        check_date = from_date + timedelta(days=1)

        # Look up to 10 days ahead (handles long weekends)
        for _ in range(10):
            if self.is_trading_day(check_date):
                return check_date
            check_date += timedelta(days=1)

        # Shouldn't happen, but fallback
        return check_date

    def time_until_market_open(self) -> Optional[timedelta]:
        """
        Get time remaining until market opens.

        Returns:
            timedelta until open, or None if market is already open
        """
        now_et = datetime.now(MARKET_TZ)

        if self.is_market_open():
            return None

        # Find next trading day
        if self.is_trading_day(now_et.date()):
            # Today is trading day - check if before open
            market_open = now_et.replace(
                hour=MARKET_OPEN_HOUR,
                minute=MARKET_OPEN_MINUTE,
                second=0,
                microsecond=0
            )
            if now_et < market_open:
                return market_open - now_et

        # Market closed today, find next trading day
        next_day = self.next_trading_day(now_et.date())
        next_open = datetime(
            next_day.year, next_day.month, next_day.day,
            MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE,
            tzinfo=MARKET_TZ
        )

        return next_open - now_et


# Singleton instance for easy import
market_calendar = MarketCalendar()


# Convenience functions (so you don't need to instantiate the class)
def is_trading_day(check_date: date) -> bool:
    """Check if a date is a trading day."""
    return market_calendar.is_trading_day(check_date)


def get_trading_days(start_date: date, end_date: date) -> List[date]:
    """Get all trading days in a range."""
    return market_calendar.get_trading_days(start_date, end_date)


def is_market_open() -> bool:
    """Check if market is currently open."""
    return market_calendar.is_market_open()


def now_et() -> datetime:
    """Get current time in Eastern timezone."""
    return datetime.now(MARKET_TZ)
