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
from dotenv import load_dotenv

try:
    from alpaca.data.enums import DataFeed
except ImportError:  # Allows non-Alpaca scripts/tests to run without alpaca-py.
    DataFeed = None

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
if DataFeed is not None:
    DATA_FEED = DataFeed.SIP if USE_SIP else DataFeed.IEX
else:
    DATA_FEED = None

# Readable name for logging
FEED_NAME = "SIP (Algo Trader Plus)" if USE_SIP else "IEX (Free)"

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
# Get data path from environment variable, with default to /Volumes/vault
# This points to your external drive to store the large tick database
DATA_BASE_PATH = Path(os.environ.get("DATA_PATH", "/Volumes/vault"))

# Create base directory if it doesn't exist
DATA_BASE_PATH.mkdir(parents=True, exist_ok=True)

# Database path
TICK_DB_PATH = DATA_BASE_PATH / "trading_data" / "tick-data-storage.db"

# Create parent directories for the database
TICK_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

print(f"✓ Tick configuration loaded successfully (Data path: {DATA_BASE_PATH})")

# =============================================================================
# CUSUM FILTER WINDOW (Tick-Level)
# =============================================================================
# Expand CUSUM events to include ticks within this window (seconds)
CUSUM_EVENT_WINDOW_SECONDS = int(os.environ.get("CUSUM_EVENT_WINDOW_SECONDS", "30"))

# =============================================================================
# ALPACA API CREDENTIALS
# =============================================================================

# Load from environment variables (same as main config)
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_API_SECRET = os.getenv('ALPACA_API_SECRET')

# Paper trading mode (True = paper, False = live)
# ALWAYS start with paper trading before going live!
ALPACA_PAPER = True

# =============================================================================
# TRADING SYMBOLS
# =============================================================================

from config.universe import build_universe

SYMBOL_TIER = os.environ.get("SYMBOL_TIER", "tier_1")
MAX_SYMBOLS = int(os.environ.get("MAX_SYMBOLS", "0")) or None

SYMBOLS, SYMBOL_TIERS, SYMBOL_BACKFILL_DAYS = build_universe(
    tier=SYMBOL_TIER,
    max_symbols=MAX_SYMBOLS
)

# =============================================================================
# BACKFILL SETTINGS
# =============================================================================

# How many days of historical tick data to fetch (fallback)
# Per-symbol values are provided in SYMBOL_BACKFILL_DAYS.
BACKFILL_DAYS = int(os.environ.get("BACKFILL_DAYS", "365"))

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

def ensure_alpaca_credentials() -> None:
    """Raise if Alpaca API credentials are missing."""
    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        raise ValueError(
            "Alpaca API credentials not found! "
            "Set ALPACA_API_KEY and ALPACA_API_SECRET in your .env file."
        )


def validate_tick_config(require_alpaca_credentials: bool = False):
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

    # 2. Check API credentials (optional)
    if require_alpaca_credentials:
        if not ALPACA_API_KEY:
            errors.append("ALPACA_API_KEY not set")
        if not ALPACA_API_SECRET:
            errors.append("ALPACA_API_SECRET not set")
    else:
        if not ALPACA_API_KEY or not ALPACA_API_SECRET:
            logger.warning(
                "Alpaca API credentials missing. Backfill/live trading will fail until set."
            )

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


# OPTIMAL TRADING PARAMETERS (Locked for Tuning)
# Based on user request to lock in these values for further tuning.
# Last updated: 2026-01-14

OPTIMAL_META_THRESHOLD = 0.001   # 0.1% - Meta model confidence filter
OPTIMAL_PROB_THRESHOLD = 0.015   # 1.5% - Primary model probability threshold

OPTIMAL_PROFIT_TARGET = 2.5      # Profit-taking multiplier vs. volatility
OPTIMAL_STOP_LOSS = 2.5          # Stop-loss multiplier vs. volatility
OPTIMAL_MAX_HOLDING_BARS = 10    # bars - Maximum holding period

OPTIMAL_RISK_REWARD_RATIO = 1.0  # Risk $1 to make $1
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
    validate_tick_config(require_alpaca_credentials=False)
    print(f"✓ Tick configuration loaded successfully ({FEED_NAME})")
except (ValueError, RuntimeError) as e:
    # Re-raise validation errors to prevent bot from starting with bad config
    raise
