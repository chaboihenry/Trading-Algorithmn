import sys
import os
from datetime import datetime, date, time, timedelta
import pytz
import alpaca_trade_api as tradeapi

# --- PATH SETUP ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY

# --- CONSTANTS ---
MARKET_TZ = pytz.timezone('America/New_York')
OPEN_TIME = time(9, 30)
CLOSE_TIME = time(16, 0)

# --- GLOBAL CACHE ---
_TRADING_DAYS_CACHE = set()

def get_api():
    return tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url='https://paper-api.alpaca.markets')

def fetch_calendar(start_date, end_date):
    api = get_api()
    print(f"   (Calendar) Fetching market schedule from {start_date} to {end_date}...")
    try:
        calendar = api.get_calendar(start=start_date.isoformat(), end=end_date.isoformat())
        for day in calendar:
            _TRADING_DAYS_CACHE.add(day.date.date())
    except Exception as e:
        print(f"   (Calendar) Error fetching schedule: {e}")

def is_trading_day(check_date):
    """Checks if a date is a valid trading day (skips weekends/holidays)."""
    if isinstance(check_date, datetime):
        check_date = check_date.date()
        
    if check_date.weekday() >= 5: 
        return False

    if not _TRADING_DAYS_CACHE:
        # Cache massive range on first run
        start = date.today() - timedelta(days=730)
        end = date.today() + timedelta(days=365)
        fetch_calendar(start, end)

    return check_date in _TRADING_DAYS_CACHE

def now_et():
    """Returns current time in Eastern Time."""
    return datetime.now(MARKET_TZ)

def is_market_open():
    """
    Checks if the market is currently open.
    Returns: True if today is a trading day AND time is between 9:30-16:00 ET.
    """
    now = now_et()
    
    # 1. Check Day
    if not is_trading_day(now.date()):
        return False
        
    # 2. Check Time
    current_time = now.time()
    return OPEN_TIME <= current_time <= CLOSE_TIME