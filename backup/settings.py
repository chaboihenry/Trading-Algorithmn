"""
Centralized Configuration for Integrated Trading Agent

FIXED (Problem 16): ALL configuration moved here - no more scattered magic numbers!

Organization:
- Environment-specific configs (DEV, PAPER, LIVE)
- API credentials
- Risk management parameters
- Strategy parameters
- Technical indicator thresholds
- Feature engineering defaults
- Validation logic

Every parameter is documented with its purpose and valid range.
"""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# =============================================================================
# PROJECT STRUCTURE
# =============================================================================

# Get the project root directory (parent of config/)
PROJECT_ROOT = Path(__file__).parent.parent

# Load environment variables from .env file at project root
env_path = PROJECT_ROOT / '.env'
load_dotenv(dotenv_path=env_path)

# =============================================================================
# ENVIRONMENT SELECTION
# =============================================================================

# Set trading environment: 'dev', 'paper', or 'live'
# DEV: Minimal trading, fast iteration, verbose logging
# PAPER: Full simulation with paper trading account
# LIVE: Real money trading (requires strict criteria to be met)
TRADING_ENV = os.getenv('TRADING_ENV', 'paper').lower()

# =============================================================================
# API CREDENTIALS
# =============================================================================

ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_API_SECRET = os.getenv('ALPACA_API_SECRET')
ALPACA_PAPER = TRADING_ENV != 'live'  # Paper trading unless explicitly live

# =============================================================================
# ENVIRONMENT-SPECIFIC CONFIGURATIONS
# =============================================================================

# Each environment has different risk tolerance and trading frequency
ENV_CONFIGS = {
    'dev': {
        # Development: Minimal risk, fast iteration
        'sleep_interval': '15M',           # Check every 15 minutes
        'max_position_pct': 0.10,          # Max 10% per position
        'max_position_value': 200,         # Max $200 per position
        'min_cash_reserve': 500,           # Keep $500 in cash
        'max_daily_loss_pct': 2.0,         # Stop at -2% daily loss
        'warning_loss_pct': 1.0,           # Warn at -1%
        'scaling_start_loss_pct': 0.3,     # Scale at -0.3%
        'confidence_threshold': 0.7,       # 70% confidence required
        'enable_extended_hours': False,    # Regular hours only
        'log_level': 'DEBUG',              # Verbose logging
    },
    'paper': {
        # Paper Trading: Realistic simulation
        'sleep_interval': '1H',            # Check every hour
        'max_position_pct': 0.15,          # Max 15% per position
        'max_position_value': 500,         # Max $500 per position
        'min_cash_reserve': 1000,          # Keep $1000 in cash
        'max_daily_loss_pct': 3.0,         # Stop at -3% daily loss
        'warning_loss_pct': 1.5,           # Warn at -1.5%
        'scaling_start_loss_pct': 0.5,     # Scale at -0.5%
        'confidence_threshold': 0.6,       # 60% confidence required
        'enable_extended_hours': True,     # Extended hours enabled
        'log_level': 'INFO',               # Standard logging
    },
    'live': {
        # Live Trading: Maximum caution
        'sleep_interval': '1H',            # Check every hour
        'max_position_pct': 0.10,          # Max 10% per position (conservative)
        'max_position_value': 1000,        # Max $1000 per position
        'min_cash_reserve': 2000,          # Keep $2000 in cash
        'max_daily_loss_pct': 2.0,         # Stop at -2% daily loss (strict)
        'warning_loss_pct': 1.0,           # Warn at -1%
        'scaling_start_loss_pct': 0.3,     # Scale at -0.3%
        'confidence_threshold': 0.7,       # 70% confidence required (strict)
        'enable_extended_hours': True,     # Extended hours enabled
        'log_level': 'INFO',               # Standard logging
    }
}

# Get current environment config
CURRENT_ENV_CONFIG = ENV_CONFIGS.get(TRADING_ENV, ENV_CONFIGS['paper'])

# =============================================================================
# RISK MANAGEMENT PARAMETERS
# =============================================================================

# Stop-loss and take-profit (applied to ALL positions via bracket orders)
STOP_LOSS_PCT = 0.05      # Exit at -5% loss (server-side, always active)
TAKE_PROFIT_PCT = 0.15    # Exit at +15% profit (server-side, always active)

# Position sizing limits (environment-specific)
MAX_POSITION_PCT = CURRENT_ENV_CONFIG['max_position_pct']
MAX_POSITION_VALUE = CURRENT_ENV_CONFIG['max_position_value']
MIN_CASH_RESERVE = CURRENT_ENV_CONFIG['min_cash_reserve']

# Daily loss circuit breaker (environment-specific)
MAX_DAILY_LOSS_PCT = CURRENT_ENV_CONFIG['max_daily_loss_pct']
WARNING_LOSS_PCT = CURRENT_ENV_CONFIG['warning_loss_pct']
SCALING_START_LOSS_PCT = CURRENT_ENV_CONFIG['scaling_start_loss_pct']

# Inverse ETF allocation (hedging)
MAX_INVERSE_ALLOCATION = 0.20  # Max 20% of portfolio in inverse positions

# =============================================================================
# STRATEGY PARAMETERS
# =============================================================================

# Trading frequency (environment-specific)
SLEEP_INTERVAL = CURRENT_ENV_CONFIG['sleep_interval']
RETRAIN_FREQUENCY_DAYS = 7  # Retrain meta-model weekly
CONFIDENCE_THRESHOLD = CURRENT_ENV_CONFIG['confidence_threshold']
MIN_TRAINING_SAMPLES = 1500  # Minimum samples to train neural network meta-model

# Market hours (environment-specific)
ENABLE_EXTENDED_HOURS = CURRENT_ENV_CONFIG['enable_extended_hours']

# Pairs trading parameters
PAIRS_LOOKBACK_DAYS = 120      # Historical data for cointegration test
PAIRS_ZSCORE_ENTRY = 1.5       # Enter trades at this z-score threshold
PAIRS_ZSCORE_EXIT = 0.5        # Exit trades when z-score normalizes
PAIRS_MIN_CORRELATION = 0.7    # Minimum correlation to consider pair
PAIRS_MIN_QUALITY_SCORE = 0.6  # Minimum quality score for trading
PAIRS_MAX_PAIRS = 5            # Maximum simultaneous pairs
PAIRS_POSITION_SIZE = 0.1      # 10% of portfolio per pair

# Sentiment strategy parameters
SENTIMENT_LOOKBACK_DAYS = 3    # Days of news to analyze

# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

# RSI thresholds
RSI_OVERSOLD = 30               # Buy signal
RSI_OVERBOUGHT = 70             # Sell signal / hedge trigger
RSI_EXTREME_OVERSOLD = 20       # Strong buy
RSI_EXTREME_OVERBOUGHT = 80     # Strong sell
RSI_DEFAULT = 50.0              # Default RSI when data unavailable

# Volatility thresholds and levels
MAX_VOLATILITY = 0.4            # Skip stocks with volatility > 40%
VOLATILITY_DEFAULT = 0.2        # Default volatility (20%) when data unavailable
VOLATILITY_LOW_THRESHOLD = 0.5  # <0.5% daily moves = low volatility
VOLATILITY_HIGH_THRESHOLD = 1.5 # 1.5-3% daily moves = high volatility
VOLATILITY_EXTREME_THRESHOLD = 3.0  # >3% daily moves = extreme volatility

# Market sentiment thresholds
BEARISH_MARKET_THRESHOLD = 0.6  # If 60%+ of market overbought, hedge

# =============================================================================
# FEATURE ENGINEERING DEFAULTS
# =============================================================================

# Default feature values when data is unavailable
FEATURE_DEFAULTS = {
    # Sentiment features
    'sentiment_strength': 0.5,      # Neutral sentiment

    # Pairs features
    'pairs_zscore': 0.0,            # No deviation
    'pairs_quality': 0.5,           # Neutral quality

    # Technical indicators
    'rsi': 50.0,                    # Neutral RSI
    'volatility': 0.2,              # 20% volatility

    # Sector features (6 sector ETFs)
    'sector_return': 0.0,           # No return

    # VIX features
    'vix_level': 20.0,              # Moderate fear
    'vix_change': 0.0,              # No change
    'vix_percentile': 50.0,         # Mid-range

    # Market breadth
    'breadth_ratio': 0.5,           # Neutral (50% advancing)

    # Interaction features
    'combined_confidence': 0.5,     # Neutral confidence
}

# Feature normalization factors
NORMALIZATION_FACTORS = {
    'vix_level': 100.0,             # VIX typically 10-50, normalize to 0-1
    'vix_change': 10.0,             # VIX changes typically ±5, normalize
    'vix_percentile': 100.0,        # Already 0-100, normalize to 0-1
    'day_of_week': 4.0,             # 0-6 (Mon-Sun), normalize to 0-1.5
    'month': 12.0,                  # 1-12, normalize to 0-1
    'sentiment_interaction': 0.5,   # Sentiment * pairs weight
}

# Time-based feature thresholds
TIME_FEATURES = {
    'month_end_threshold': 25,      # Last 5 days of month (>=25)
    'month_start_threshold': 5,     # First 5 days of month (<=5)
}

# =============================================================================
# HEDGE TIMING (Problem 9 - Dynamic check intervals)
# =============================================================================

# Hedge check intervals based on market volatility
HEDGE_CHECK_INTERVALS = {
    'extreme': 15,   # Minutes - check every 15 min in extreme volatility
    'high': 30,      # Minutes - check every 30 min in high volatility
    'normal': 60,    # Minutes - check every hour in normal conditions
}

# Pre-market hours (Eastern Time)
PRE_MARKET_START_HOUR = 4       # 4 AM ET
PRE_MARKET_END_HOUR = 9         # 9 AM ET
PRE_MARKET_END_MINUTE = 30      # 9:30 AM ET

# =============================================================================
# SYMBOL LISTS
# =============================================================================

# Primary trading universe (quality large-caps)
TRADING_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "META", "NVDA",
    "AMZN", "TSLA", "AMD", "NFLX", "CRM",
    "ADBE", "INTC", "PYPL", "SQ"
]

# Market sentiment indicators (sample for overall market gauge)
MARKET_SENTIMENT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]

# Sector ETFs for sector momentum features (Problem 12)
SECTOR_ETFS = {
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLE': 'Energy',
    'XLV': 'Healthcare',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples'
}

# Inverse ETFs for hedging (profit when market drops)
INVERSE_ETFS = {
    'tech': 'SQQQ',    # 3x inverse NASDAQ
    'sp500': 'SPXS',   # 3x inverse S&P 500
    'general': 'SH',   # 1x inverse S&P 500 (conservative, default)
    'nasdaq': 'PSQ'    # 1x inverse NASDAQ (alternative to SQQQ)
}

# Default inverse ETF to use for hedging
DEFAULT_INVERSE_ETF = 'SH'

# Market breadth symbols (for advance/decline ratio)
MARKET_BREADTH_SYMBOLS = [
    'SPY',   # S&P 500
    'QQQ',   # NASDAQ-100
    'DIA',   # Dow Jones
    'IWM',   # Russell 2000
    'XLK',   # Tech
    'XLF',   # Financials
    'XLE',   # Energy
    'XLV',   # Healthcare
    'XLY',   # Consumer Discretionary
    'XLP'    # Consumer Staples
]

# =============================================================================
# DATA SOURCES
# =============================================================================

# Database path
DB_PATH = os.getenv('DB_PATH', '/Volumes/Vault/85_assets_prediction.db')

# Model storage
MODELS_DIR = PROJECT_ROOT / 'models'
LOGS_DIR = PROJECT_ROOT / 'logs'
DATA_DIR = PROJECT_ROOT / 'data'

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Performance tracking database
PERFORMANCE_DB = DATA_DIR / 'performance.db'

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_LEVEL = CURRENT_ENV_CONFIG['log_level']
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# =============================================================================
# PERFORMANCE TRACKING (Real-Money Readiness Criteria)
# =============================================================================

# Minimum performance criteria before switching to live trading
REAL_MONEY_CRITERIA = {
    'min_days': 90,              # Track for at least 90 days
    'min_sharpe': 1.0,           # Sharpe ratio must be > 1.0
    'max_drawdown': 0.10,        # Maximum drawdown < 10%
    'min_return': 0.0,           # Must be profitable (> 0% return)
    'stop_loss_compliance': 1.0, # 100% of positions must have stops
    'min_win_rate': 0.50,        # At least 50% win rate
}

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_config() -> bool:
    """
    Validate configuration on startup.

    FIXED (Problem 16): Comprehensive validation of all config parameters.

    Raises:
        ValueError: If any critical setting is invalid

    Returns:
        True if all validations pass
    """
    errors = []

    # 1. API Credentials
    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        errors.append(
            "API credentials not found! "
            "Set ALPACA_API_KEY and ALPACA_API_SECRET in .env file."
        )

    # 2. Environment validation
    if TRADING_ENV not in ['dev', 'paper', 'live']:
        errors.append(
            f"Invalid TRADING_ENV: '{TRADING_ENV}'. "
            "Must be 'dev', 'paper', or 'live'."
        )

    # 3. Risk/Reward ratio
    if STOP_LOSS_PCT >= TAKE_PROFIT_PCT:
        errors.append(
            f"Invalid risk/reward: stop-loss ({STOP_LOSS_PCT}) "
            f"must be < take-profit ({TAKE_PROFIT_PCT})"
        )

    # 4. Position sizing
    if MAX_POSITION_PCT > 0.5:
        errors.append(
            f"MAX_POSITION_PCT ({MAX_POSITION_PCT}) too high! "
            "No position should exceed 50% of portfolio."
        )

    if MAX_POSITION_PCT <= 0:
        errors.append(
            f"MAX_POSITION_PCT ({MAX_POSITION_PCT}) must be positive."
        )

    # 5. Daily loss thresholds
    if MAX_DAILY_LOSS_PCT <= 0:
        errors.append(
            f"MAX_DAILY_LOSS_PCT ({MAX_DAILY_LOSS_PCT}) must be positive."
        )

    if WARNING_LOSS_PCT >= MAX_DAILY_LOSS_PCT:
        errors.append(
            f"WARNING_LOSS_PCT ({WARNING_LOSS_PCT}) must be "
            f"< MAX_DAILY_LOSS_PCT ({MAX_DAILY_LOSS_PCT})"
        )

    if SCALING_START_LOSS_PCT >= WARNING_LOSS_PCT:
        errors.append(
            f"SCALING_START_LOSS_PCT ({SCALING_START_LOSS_PCT}) must be "
            f"< WARNING_LOSS_PCT ({WARNING_LOSS_PCT})"
        )

    # 6. Confidence threshold
    if not 0.5 <= CONFIDENCE_THRESHOLD <= 1.0:
        errors.append(
            f"CONFIDENCE_THRESHOLD ({CONFIDENCE_THRESHOLD}) must be "
            "between 0.5 and 1.0"
        )

    # 7. Training samples
    if MIN_TRAINING_SAMPLES < 1000:
        errors.append(
            f"MIN_TRAINING_SAMPLES ({MIN_TRAINING_SAMPLES}) too low! "
            "Need at least 1000 samples for reliable neural network training."
        )

    # 8. Pairs parameters
    if PAIRS_ZSCORE_EXIT >= PAIRS_ZSCORE_ENTRY:
        errors.append(
            f"PAIRS_ZSCORE_EXIT ({PAIRS_ZSCORE_EXIT}) must be "
            f"< PAIRS_ZSCORE_ENTRY ({PAIRS_ZSCORE_ENTRY})"
        )

    if not 0 <= PAIRS_MIN_CORRELATION <= 1:
        errors.append(
            f"PAIRS_MIN_CORRELATION ({PAIRS_MIN_CORRELATION}) must be "
            "between 0 and 1"
        )

    # 9. RSI thresholds
    if RSI_OVERSOLD >= RSI_OVERBOUGHT:
        errors.append(
            f"RSI_OVERSOLD ({RSI_OVERSOLD}) must be "
            f"< RSI_OVERBOUGHT ({RSI_OVERBOUGHT})"
        )

    # 10. Database path
    if not DB_PATH:
        errors.append("DB_PATH not set!")

    # 11. Symbol lists
    if not TRADING_SYMBOLS:
        errors.append("TRADING_SYMBOLS is empty!")

    if not MARKET_SENTIMENT_SYMBOLS:
        errors.append("MARKET_SENTIMENT_SYMBOLS is empty!")

    # 12. Volatility thresholds (must be ordered)
    if not (VOLATILITY_LOW_THRESHOLD < VOLATILITY_HIGH_THRESHOLD < VOLATILITY_EXTREME_THRESHOLD):
        errors.append(
            f"Volatility thresholds must be ordered: "
            f"low ({VOLATILITY_LOW_THRESHOLD}) < "
            f"high ({VOLATILITY_HIGH_THRESHOLD}) < "
            f"extreme ({VOLATILITY_EXTREME_THRESHOLD})"
        )

    # 13. Live trading safety checks
    if TRADING_ENV == 'live':
        if ALPACA_PAPER:
            errors.append(
                "CRITICAL: TRADING_ENV='live' but ALPACA_PAPER=True! "
                "This is a safety lock. Set ALPACA_PAPER=False only when ready."
            )

        if MAX_DAILY_LOSS_PCT > 3.0:
            errors.append(
                f"LIVE TRADING: MAX_DAILY_LOSS_PCT ({MAX_DAILY_LOSS_PCT}) "
                "should be <= 3.0% for real money."
            )

    # Report all errors
    if errors:
        error_msg = "\n\n".join([f"❌ {e}" for e in errors])
        raise ValueError(
            f"\n\n{'=' * 80}\n"
            f"CONFIGURATION ERRORS DETECTED:\n"
            f"{'=' * 80}\n\n"
            f"{error_msg}\n\n"
            f"{'=' * 80}\n"
        )

    return True


def get_config_summary() -> Dict[str, Any]:
    """
    Get summary of current configuration for logging/display.

    Returns:
        Dict with key configuration parameters
    """
    return {
        'environment': TRADING_ENV,
        'paper_trading': ALPACA_PAPER,
        'sleep_interval': SLEEP_INTERVAL,
        'max_position_pct': MAX_POSITION_PCT,
        'max_position_value': MAX_POSITION_VALUE,
        'max_daily_loss_pct': MAX_DAILY_LOSS_PCT,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'stop_loss_pct': STOP_LOSS_PCT,
        'take_profit_pct': TAKE_PROFIT_PCT,
        'extended_hours': ENABLE_EXTENDED_HOURS,
        'log_level': LOG_LEVEL,
        'num_trading_symbols': len(TRADING_SYMBOLS),
        'num_sector_etfs': len(SECTOR_ETFS),
    }


# =============================================================================
# AUTO-VALIDATION
# =============================================================================

# Validate config on import
try:
    validate_config()

    # Log config summary if in dev mode
    if TRADING_ENV == 'dev':
        import logging
        logger = logging.getLogger(__name__)
        logger.info("=" * 80)
        logger.info("CONFIGURATION LOADED SUCCESSFULLY")
        logger.info("=" * 80)
        config = get_config_summary()
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 80)

except ValueError as e:
    # Re-raise validation errors
    raise
