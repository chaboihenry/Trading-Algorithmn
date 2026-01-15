import sys
import os
from datetime import date, timedelta
import alpaca_trade_api as tradeapi

# --- PATH SETUP ---
# Ensure we can find config
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY

# --- GLOBAL CACHE ---
# We store known trading days here so we don't spam the API
_TRADING_DAYS_CACHE = set()

def get_api():
    """Returns an authenticated Alpaca API connection."""
    return tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url='https://paper-api.alpaca.markets')

def fetch_calendar(start_date, end_date):
    """
    Fetches trading days from Alpaca and updates the cache.
    """
    api = get_api()
    print(f"   (Calendar) Fetching market schedule from {start_date} to {end_date}...")
    
    try:
        # Get calendar (returns list of Calendar objects)
        calendar = api.get_calendar(start=start_date.isoformat(), end=end_date.isoformat())
        
        for day in calendar:
            # day.date is a Timestamp, convert to standard python date
            _TRADING_DAYS_CACHE.add(day.date.date())
            
    except Exception as e:
        print(f"   (Calendar) Error fetching schedule: {e}")

def is_trading_day(check_date):
    """
    Checks if a specific date is a valid trading day.
    Automatically fetches data if the cache is empty.
    """
    # 1. Simple check: Weekends are never trading days
    if check_date.weekday() >= 5: # 5=Sat, 6=Sun
        return False

    # 2. Check Cache
    if not _TRADING_DAYS_CACHE:
        # If cache is empty, fetch a wide range (e.g., 2 years back to 1 year forward)
        # This handles the most common use case (backfilling) in one go.
        start = date.today() - timedelta(days=730)
        end = date.today() + timedelta(days=365)
        fetch_calendar(start, end)

    # 3. Return status
    return check_date in _TRADING_DAYS_CACHE