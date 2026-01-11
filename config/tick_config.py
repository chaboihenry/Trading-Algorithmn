"""
Tick Data Configuration

This module configures tick data collection and storage for the RiskLabAI trading bot.

KEY FEATURE: Single switch to toggle between IEX (free) and SIP (paid) data feeds.
- IEX: ~10% market coverage, free with basic Alpaca account
- SIP: 100% market coverage, requires Algo Trader Plus ($49/month)

When you upgrade to Algo Trader Plus, simply change USE_SIP = True and
restart the bot. Everything else stays the same.

OOP Concepts:
- This is a configuration module (not a class), but imports DataFeed enum
- DataFeed is an enum class from alpaca-py SDK representing feed types
- Using conditional logic to select the right feed based on USE_SIP flag
"""

import os
from pathlib import Path
from alpaca.data.enums import DataFeed
from dotenv import load_dotenv

# Load environment variables from .env file
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / '.env')

# =============================================================================
# FLIP THIS SWITCH WHEN YOU UPGRADE TO ALGO TRADER PLUS
# =============================================================================

USE_SIP = False  # Set True after purchasing Algo Trader Plus subscription

# =============================================================================
# DATA FEED SELECTION (Automatically chosen based on USE_SIP)
# =============================================================================

# DataFeed.SIP: Securities Information Processor - 100% market coverage
# DataFeed.IEX: Investors Exchange - ~10% market coverage (free)
DATA_FEED = DataFeed.SIP if USE_SIP else DataFeed.IEX

# Readable name for logging
FEED_NAME = "SIP (Algo Trader Plus)" if USE_SIP else "IEX (Free)"

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
# Get data path from environment variable, with default for Docker
DATA_BASE_PATH = Path(os.environ.get("DATA_PATH", "/app/data"))

# Create base directory if it doesn't exist
DATA_BASE_PATH.mkdir(parents=True, exist_ok=True)

# Database path
TICK_DB_PATH = DATA_BASE_PATH / "trading_data" / "tick-data-storage.db"

# Create parent directories for the database
TICK_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

print(f"✓ Tick configuration loaded successfully (Data path: {DATA_BASE_PATH})")

# =============================================================================
# ALPACA API CREDENTIALS
# =============================================================================

# Load from environment variables (same as main config)
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_API_SECRET = os.getenv('ALPACA_API_SECRET')

if not ALPACA_API_KEY or not ALPACA_API_SECRET:
    raise ValueError(
        "Alpaca API credentials not found! "
        "Set ALPACA_API_KEY and ALPACA_API_SECRET in your .env file."
    )

# Paper trading mode (True = paper, False = live)
# ALWAYS start with paper trading before going live!
ALPACA_PAPER = True

# =============================================================================
# TRADING SYMBOLS
# =============================================================================

# Top 99 S&P 500 symbols by market cap (plus SPY, QQQ, IWM)
# These symbols have been selected for high liquidity and market coverage
SYMBOLS = [
    # Market indices
    "SPY", "QQQ", "IWM",
    # Top 99 stocks
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "ADI", "AMAT", "AMD", "AMGN", "AMT",
    "AMZN", "AON", "APH", "AVGO", "AXP", "BA", "BAC", "BDX", "BKNG", "BLK",
    "BRK.B", "BSX", "C", "CAT", "CB", "CI", "CMCSA", "CME", "COP", "COST",
    "CRM", "CSCO", "CVX", "DE", "DHR", "DUK", "ELV", "ETN", "GD", "GE",
    "GILD", "GOOG", "GOOGL", "HD", "HON", "IBM", "INTC", "INTU", "ISRG", "ITW",
    "JNJ", "JPM", "KO", "LIN", "LLY", "LOW", "LRCX", "MA", "MCD", "MDLZ",
    "META", "MMC", "MMM", "MRK", "MSFT", "MU", "NEE", "NFLX", "NOC", "NOW",
    "NVDA", "ORCL", "PANW", "PEP", "PG", "PGR", "PLD", "PM", "QCOM", "REGN",
    "RTX", "SBUX", "SCHW", "SLB", "SO", "SPGI", "SYK", "TJX", "TMO", "TMUS",
    "TSLA", "TXN", "UNH", "UNP", "V", "VRTX", "VZ", "WMT", "XOM"
]

# =============================================================================
# BACKFILL SETTINGS
# =============================================================================

# How many days of historical tick data to fetch
# 365 days = 12 months of data for ML training (need 2000+ samples minimum)
BACKFILL_DAYS = 365

# Rate limiting to avoid Alpaca API throttling
# Alpaca allows 200 requests/minute
# With 0.35s delay: 1/0.35 = 2.86 requests/second = 171 requests/minute (safe)
RATE_LIMIT_DELAY = 0.35  # seconds between API calls

# =============================================================================
# EXTENDED TRADING HOURS (Eastern Time)
# =============================================================================

# Alpaca provides tick data during extended hours
# Pre-market: 4:00 AM - 9:30 AM ET
# Regular: 9:30 AM - 4:00 PM ET
# After-hours: 4:00 PM - 8:00 PM ET

PREMARKET_START_HOUR = 4    # 4 AM ET
REGULAR_START_HOUR = 9      # 9:30 AM ET
REGULAR_START_MINUTE = 30
REGULAR_END_HOUR = 16       # 4 PM ET
REGULAR_END_MINUTE = 0
AFTERHOURS_END_HOUR = 20    # 8 PM ET

# For backfill and live collection, use extended hours
# This captures pre/post market activity which can be informative
COLLECTION_START_HOUR = PREMARKET_START_HOUR
COLLECTION_END_HOUR = AFTERHOURS_END_HOUR

# =============================================================================
# TICK IMBALANCE BAR SETTINGS
# =============================================================================

# Target number of bars per trading day
# With SIP data: ~50 bars/day (captures major moves)
# With IEX data: ~5-10 bars/day (much less data)
TARGET_BARS_PER_DAY_SIP = 50
TARGET_BARS_PER_DAY_IEX = 5

# Use appropriate target based on current feed
TARGET_BARS_PER_DAY = TARGET_BARS_PER_DAY_SIP if USE_SIP else TARGET_BARS_PER_DAY_IEX

# Initial threshold for tick imbalance bars
# Calibrated by scripts/calibrate_threshold.py to produce ~5 bars/day with IEX
# Value is cumulative imbalance threshold before sampling a new bar
# Calibration result: 70.0 → ~4.5 bars/day with IEX data (125 trading days)
INITIAL_IMBALANCE_THRESHOLD = 70.0

# =============================================================================
# STORAGE ESTIMATES
# =============================================================================

# Rough storage estimates (for planning):
#
# IEX DATA (free):
# - SPY: ~50,000 trades/day * 100 bytes ≈ 5 MB/day
# - 3 symbols * 365 days ≈ 5.5 GB total
#
# SIP DATA (after upgrade):
# - SPY: ~500,000 trades/day * 100 bytes ≈ 50 MB/day
# - 3 symbols * 365 days ≈ 55 GB total
#
# 2TB SSD has plenty of space for both scenarios

# =============================================================================
# VALIDATION
# =============================================================================

def validate_tick_config():
    """
    Validate tick configuration on import.

    This function checks that all critical settings are valid before
    any tick data operations begin. Think of it like a pre-flight checklist.

    Raises:
        ValueError: If any configuration is invalid
        RuntimeError: If external dependencies are missing
    """
    errors = []

    # 1. Check database path is accessible
    db_path = Path(TICK_DB_PATH)
    if not db_path.parent.exists():
        errors.append(
            f"Database directory does not exist: {db_path.parent}. "
            f"Ensure DATA_PATH is correctly set in environment."
        )

    # 2. Check API credentials
    if not ALPACA_API_KEY:
        errors.append("ALPACA_API_KEY not set")

    if not ALPACA_API_SECRET:
        errors.append("ALPACA_API_SECRET not set")

    # 3. Validate symbols list
    if not SYMBOLS:
        errors.append("SYMBOLS list is empty")

    for symbol in SYMBOLS:
        if not symbol or not symbol.strip():
            errors.append(f"Invalid symbol: '{symbol}'")

    # 4. Validate backfill days
    if BACKFILL_DAYS <= 0:
        errors.append(f"BACKFILL_DAYS must be positive, got {BACKFILL_DAYS}")

    if BACKFILL_DAYS > 730:
        errors.append(
            f"BACKFILL_DAYS too large ({BACKFILL_DAYS}). "
            f"Consider shorter period to avoid excessive API usage (max 2 years)."
        )

    # 5. Validate rate limit delay
    if RATE_LIMIT_DELAY < 0.1:
        errors.append(
            f"RATE_LIMIT_DELAY too small ({RATE_LIMIT_DELAY}s). "
            f"Risk hitting Alpaca rate limits (200 req/min)."
        )

    # 6. Validate trading hours
    if not (0 <= COLLECTION_START_HOUR < 24):
        errors.append(f"Invalid COLLECTION_START_HOUR: {COLLECTION_START_HOUR}")

    if not (0 <= COLLECTION_END_HOUR < 24):
        errors.append(f"Invalid COLLECTION_END_HOUR: {COLLECTION_END_HOUR}")

    if COLLECTION_START_HOUR >= COLLECTION_END_HOUR:
        errors.append(
            f"COLLECTION_START_HOUR ({COLLECTION_START_HOUR}) must be "
            f"< COLLECTION_END_HOUR ({COLLECTION_END_HOUR})"
        )

    # 7. Validate bar targets
    if TARGET_BARS_PER_DAY <= 0:
        errors.append(f"TARGET_BARS_PER_DAY must be positive, got {TARGET_BARS_PER_DAY}")

    # 8. Validate imbalance threshold
    if INITIAL_IMBALANCE_THRESHOLD <= 0:
        errors.append(
            f"INITIAL_IMBALANCE_THRESHOLD must be positive, "
            f"got {INITIAL_IMBALANCE_THRESHOLD}"
        )

    # Report errors if any
    if errors:
        error_msg = "\n".join([f"  ❌ {e}" for e in errors])
        raise ValueError(
            f"\n{'=' * 80}\n"
            f"TICK CONFIG VALIDATION FAILED\n"
            f"{'=' * 80}\n"
            f"{error_msg}\n"
            f"{'=' * 80}\n"
        )

    return True


def get_tick_config_summary():
    """
    Get human-readable summary of tick configuration.

    Returns:
        dict: Configuration summary for logging

    This is useful for debugging and confirming settings when the bot starts.
    """
    return {
        'data_feed': FEED_NAME,
        'using_sip': USE_SIP,
        'database_path': TICK_DB_PATH,
        'symbols': SYMBOLS,
        'backfill_days': BACKFILL_DAYS,
        'target_bars_per_day': TARGET_BARS_PER_DAY,
        'collection_hours': f"{COLLECTION_START_HOUR}:00 - {COLLECTION_END_HOUR}:00 ET",
        'rate_limit_delay': f"{RATE_LIMIT_DELAY}s",
    }


# =============================================================================
# OPTIMAL TRADING PARAMETERS (From Parameter Sweep)
# =============================================================================

# Best parameters identified by parameter sweep on 2025-12-30
# Based on 48 configurations tested with Sharpe ratio optimization
# Backtest performance: Sharpe 3.53, Win Rate 73.1%, 52 trades

# Signal filtering
OPTIMAL_META_THRESHOLD = 0.001   # 0.1% - Meta model confidence filter
OPTIMAL_PROB_THRESHOLD = 0.015   # 1.5% - Primary model probability threshold

# Exit parameters
OPTIMAL_PROFIT_TARGET = 0.04     # 4.0% - Take profit level
OPTIMAL_STOP_LOSS = 0.02         # 2.0% - Stop loss level
OPTIMAL_MAX_HOLDING_BARS = 30    # bars - Maximum holding period (30 bars → ~30-35% neutral labels)

# Risk management
OPTIMAL_RISK_REWARD_RATIO = 2.0  # Risk $1 to make $2
MAX_POSITION_SIZE_PCT = 0.10     # 10% - Maximum position size
MAX_DAILY_DRAWDOWN_PCT = 0.15    # 15% - Daily drawdown alert threshold

# Backtest metrics (for reference)
BACKTEST_SHARPE_RATIO = 3.53
BACKTEST_WIN_RATE = 0.731
BACKTEST_NUM_TRADES = 52
BACKTEST_MAX_DRAWDOWN = -0.0779

# =============================================================================
# FRACTIONAL DIFFERENCING PARAMETER
# =============================================================================

# Optimal d value for making time series stationary
# Found by analyzing 7,286 tick imbalance bars (2020-2025)
# d=0.30 achieves stationarity (p=0.0405) while preserving 70% of memory
# Lower d = more memory = better predictions
OPTIMAL_FRACTIONAL_D = 0.30

# =============================================================================
# PORTABILITY HELPER
# =============================================================================

def should_use_tick_bars() -> bool:
    """
    Auto-detect whether to use tick bars based on database existence.

    This makes the bot portable - it will automatically use tick bars if
    the database exists, otherwise fall back to Alpaca API bars.

    Returns:
        bool: True if tick database exists and is accessible, False otherwise
    """
    try:
        # Check if database file exists
        if not TICK_DB_PATH.exists():
            print(f"ℹ️  Tick database not found at {TICK_DB_PATH}")
            print(f"ℹ️  Will use Alpaca API for real-time bars")
            return False

        # Check if database is readable (has content)
        if TICK_DB_PATH.stat().st_size == 0:
            print(f"⚠️  Tick database exists but is empty: {TICK_DB_PATH}")
            print(f"ℹ️  Will use Alpaca API for real-time bars")
            return False

        print(f"✓ Tick database found: {TICK_DB_PATH} ({TICK_DB_PATH.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"✓ Will use tick imbalance bars from database")
        return True

    except Exception as e:
        print(f"⚠️  Error checking tick database: {e}")
        print(f"ℹ️  Will use Alpaca API for real-time bars")
        return False

# =============================================================================
# MARKET TIMEZONE (Eastern Time)
# =============================================================================

from zoneinfo import ZoneInfo

# US Eastern timezone - the market's official timezone
# All time comparisons, market hours checks, and trading day logic use this
# Do NOT use datetime.now() - use datetime.now(MARKET_TZ) instead
MARKET_TZ = ZoneInfo("America/New_York")

# Market hours (in Eastern Time)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# =============================================================================
# AUTO-VALIDATION ON IMPORT
# =============================================================================

# Validate configuration when this module is imported
# This catches errors early before any tick operations begin
try:
    validate_tick_config()
    print(f"✓ Tick configuration loaded successfully ({FEED_NAME})")
except (ValueError, RuntimeError) as e:
    # Re-raise validation errors to prevent bot from starting with bad config
    raise
