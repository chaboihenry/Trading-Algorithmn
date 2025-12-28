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

# Human-readable name for logging
FEED_NAME = "SIP (Algo Trader Plus)" if USE_SIP else "IEX (Free)"

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# Path to existing SQLite database on external SSD
# Database already exists but has no tables yet (will be created by init script)
TICK_DB_PATH = "/Volumes/Vault/trading_data/tick-data-storage.db"

# Verify the Vault is mounted before proceeding
VAULT_PATH = Path("/Volumes/Vault")
if not VAULT_PATH.exists():
    raise RuntimeError(
        f"External SSD not mounted at {VAULT_PATH}! "
        f"Please connect the Vault drive before running tick data operations."
    )

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

# =============================================================================
# TRADING SYMBOLS
# =============================================================================

# Start with liquid ETFs (high tick volume, good for testing)
# SPY: S&P 500 - most liquid security in the world
# QQQ: NASDAQ-100 - tech heavy
# IWM: Russell 2000 - small caps
SYMBOLS = ["SPY", "QQQ", "IWM"]

# =============================================================================
# BACKFILL SETTINGS
# =============================================================================

# How many days of historical tick data to fetch
# 60 days = ~3 months of data for training
BACKFILL_DAYS = 60

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
# - 3 symbols * 60 days ≈ 900 MB total
#
# SIP DATA (after upgrade):
# - SPY: ~500,000 trades/day * 100 bytes ≈ 50 MB/day
# - 3 symbols * 60 days ≈ 9 GB total
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
        RuntimeError: If external dependencies (like Vault drive) are missing
    """
    errors = []

    # 1. Check database path is accessible
    db_path = Path(TICK_DB_PATH)
    if not db_path.parent.exists():
        errors.append(
            f"Database directory does not exist: {db_path.parent}. "
            f"Ensure Vault drive is mounted."
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

    if BACKFILL_DAYS > 365:
        errors.append(
            f"BACKFILL_DAYS too large ({BACKFILL_DAYS}). "
            f"Consider shorter period to avoid excessive API usage."
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
