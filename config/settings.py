"""
Centralized configuration for the Integrated Trading Agent.

All magic numbers, thresholds, symbol lists, and API configuration
live here. This makes it easy to tune the bot without digging through code.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Get the project root directory (parent of config/)
PROJECT_ROOT = Path(__file__).parent.parent

# Load environment variables from .env file at project root
# This ensures .env is found regardless of where scripts are run from
env_path = PROJECT_ROOT / '.env'
load_dotenv(dotenv_path=env_path)

# =============================================================================
# API CREDENTIALS
# =============================================================================

ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_API_SECRET = os.getenv('ALPACA_API_SECRET')
ALPACA_PAPER = True  # Always use paper trading for testing

# =============================================================================
# RISK MANAGEMENT THRESHOLDS
# =============================================================================

# Stop-loss and take-profit (applied to ALL positions)
STOP_LOSS_PCT = 0.05  # Exit at -5% loss
TAKE_PROFIT_PCT = 0.15  # Exit at +15% profit

# Position sizing limits
MAX_POSITION_PCT = 0.15  # No single position > 15% of portfolio
MAX_POSITION_VALUE = 500  # Maximum $ per position for new entries
MIN_CASH_RESERVE = 1000  # Always keep this much cash available

# Inverse ETF allocation
MAX_INVERSE_ALLOCATION = 0.20  # Max 20% of portfolio in hedges

# =============================================================================
# TRADING PARAMETERS
# =============================================================================

# How often to check positions and look for opportunities
SLEEP_INTERVAL = "1H"  # Check every 1 hour
RETRAIN_FREQUENCY_DAYS = 7  # Retrain meta-model weekly

# Signal generation
CONFIDENCE_THRESHOLD = 0.6  # Only trade if 60%+ confident
MIN_TRAINING_SAMPLES = 100  # Minimum samples needed to train model

# Market hours
ENABLE_EXTENDED_HOURS = True  # Trade 4 AM - 8 PM ET (not just 9:30 AM - 4 PM)

# =============================================================================
# SYMBOL LISTS
# =============================================================================

# Primary trading universe (quality large-caps)
TRADING_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "META", "NVDA",
    "AMZN", "TSLA", "AMD", "NFLX", "CRM",
    "ADBE", "INTC", "PYPL", "SQ"
]

# Market sentiment indicators (sample these to gauge overall market)
MARKET_SENTIMENT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]

# Inverse ETFs for hedging (profit when market drops)
INVERSE_ETFS = {
    'tech': 'SQQQ',    # 3x inverse NASDAQ
    'sp500': 'SPXS',   # 3x inverse S&P 500
    'general': 'SH',   # 1x inverse S&P 500 (conservative, default)
    'nasdaq': 'PSQ'    # 1x inverse NASDAQ (alternative to SQQQ)
}

# Default inverse ETF to use for hedging
DEFAULT_INVERSE_ETF = 'SH'

# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

# RSI thresholds
RSI_OVERSOLD = 30  # Buy signal
RSI_OVERBOUGHT = 70  # Sell signal / hedge trigger
RSI_EXTREME_OVERSOLD = 20  # Strong buy
RSI_EXTREME_OVERBOUGHT = 80  # Strong sell

# Volatility thresholds
MAX_VOLATILITY = 0.4  # Skip stocks with volatility > 40%

# Market sentiment thresholds
BEARISH_MARKET_THRESHOLD = 0.6  # If 60%+ of market overbought, hedge

# =============================================================================
# DATA SOURCES
# =============================================================================

# Database path
DB_PATH = '/Volumes/Vault/85_assets_prediction.db'

# Model storage
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
LOGS_DIR = PROJECT_ROOT / 'logs'

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# =============================================================================
# PERFORMANCE TRACKING (for real-money readiness evaluation)
# =============================================================================

# Minimum performance criteria before using real money
REAL_MONEY_CRITERIA = {
    'min_days': 90,  # Track for at least 90 days
    'min_sharpe': 1.0,  # Sharpe ratio must be > 1.0
    'max_drawdown': 0.10,  # Maximum drawdown < 10%
    'min_return': 0.0,  # Must be profitable (> 0% return)
    'stop_loss_compliance': 1.0  # 100% of positions must have stops
}

# Performance tracking database
PERFORMANCE_DB = PROJECT_ROOT / 'data' / 'performance.db'

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """
    Validate configuration on startup.
    Raises ValueError if critical settings are missing or invalid.
    """
    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        raise ValueError(
            "API credentials not found! "
            "Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables."
        )

    if STOP_LOSS_PCT >= TAKE_PROFIT_PCT:
        raise ValueError(
            f"Invalid risk/reward: stop-loss ({STOP_LOSS_PCT}) "
            f"must be < take-profit ({TAKE_PROFIT_PCT})"
        )

    if MAX_POSITION_PCT > 0.5:
        raise ValueError(
            f"MAX_POSITION_PCT ({MAX_POSITION_PCT}) too high! "
            "No position should exceed 50% of portfolio."
        )

    if not DB_PATH:
        raise ValueError("DB_PATH not set!")

    return True


# Auto-validate on import
validate_config()
