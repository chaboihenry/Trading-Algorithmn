"""
Liquidity Tier Classification System

This module classifies stocks into liquidity tiers based on average daily volume
and assigns optimal backfill days for each tier.

ADAPTIVE BACKFILL STRATEGY:
- Tier 1 (High): 365 days → Gets 1M+ ticks easily
- Tier 2 (Medium): 500 days → Gets 800k-1M ticks
- Tier 3 (Low): 700 days → Gets 600k-900k ticks
- Tier 4 (Ultra-low): EXCLUDE → Not suitable for ML trading

This approach optimizes:
1. API usage (don't over-backfill high liquidity stocks)
2. Data quality (get enough data for low liquidity stocks)
3. Database size (exclude penny stocks)
4. Trading viability (focus on liquid stocks)

When scaling to thousands of stocks:
- Automatically classify based on volume
- Backfill with appropriate days per tier
- Exclude ultra-low liquidity (saves time/space)
"""

from enum import Enum
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class LiquidityTier(Enum):
    """
    Liquidity tier classification based on average daily volume.
    """
    HIGH = 1        # Mega caps, major ETFs (10M+ avg volume)
    MEDIUM = 2      # Large/mid caps (1M-10M avg volume)
    LOW = 3         # Small caps with decent volume (500k-1M avg volume)
    ULTRA_LOW = 4   # Penny stocks, micro caps (<500k avg volume) - EXCLUDE


# Backfill days per tier for OPTIMAL data collection
BACKFILL_DAYS_BY_TIER = {
    LiquidityTier.HIGH: 365,      # 1 year - sufficient for high liquidity
    LiquidityTier.MEDIUM: 500,    # ~1.4 years - more days for medium liquidity
    LiquidityTier.LOW: 700,       # ~2 years - maximum for low liquidity
    LiquidityTier.ULTRA_LOW: 0,   # Don't backfill - will be excluded
}


# Average volume thresholds (shares per day)
VOLUME_THRESHOLDS = {
    LiquidityTier.HIGH: 10_000_000,      # 10M+ shares/day
    LiquidityTier.MEDIUM: 1_000_000,     # 1M-10M shares/day
    LiquidityTier.LOW: 500_000,          # 500k-1M shares/day
    LiquidityTier.ULTRA_LOW: 0,          # <500k shares/day - EXCLUDE
}


# Minimum bars required for ML training
MIN_BARS_FOR_TRAINING = 200


def classify_by_volume(avg_daily_volume: float) -> LiquidityTier:
    """
    Classify a stock into a liquidity tier based on average daily volume.

    Args:
        avg_daily_volume: Average daily trading volume in shares

    Returns:
        LiquidityTier enum value

    Example:
        >>> classify_by_volume(50_000_000)  # SPY
        LiquidityTier.HIGH

        >>> classify_by_volume(2_000_000)  # Mid cap
        LiquidityTier.MEDIUM

        >>> classify_by_volume(200_000)  # Penny stock
        LiquidityTier.ULTRA_LOW
    """
    if avg_daily_volume >= VOLUME_THRESHOLDS[LiquidityTier.HIGH]:
        return LiquidityTier.HIGH
    elif avg_daily_volume >= VOLUME_THRESHOLDS[LiquidityTier.MEDIUM]:
        return LiquidityTier.MEDIUM
    elif avg_daily_volume >= VOLUME_THRESHOLDS[LiquidityTier.LOW]:
        return LiquidityTier.LOW
    else:
        return LiquidityTier.ULTRA_LOW


def get_backfill_days(tier: LiquidityTier) -> int:
    """
    Get optimal backfill days for a liquidity tier.

    Args:
        tier: LiquidityTier enum value

    Returns:
        Number of days to backfill

    Example:
        >>> get_backfill_days(LiquidityTier.HIGH)
        365

        >>> get_backfill_days(LiquidityTier.MEDIUM)
        500
    """
    return BACKFILL_DAYS_BY_TIER[tier]


def should_include_symbol(tier: LiquidityTier) -> bool:
    """
    Determine if a symbol should be included in the trading universe.

    Args:
        tier: LiquidityTier enum value

    Returns:
        True if symbol should be included, False if it should be excluded

    Example:
        >>> should_include_symbol(LiquidityTier.HIGH)
        True

        >>> should_include_symbol(LiquidityTier.ULTRA_LOW)
        False
    """
    return tier != LiquidityTier.ULTRA_LOW


# Predefined tier assignments for current S&P 100 symbols
# This avoids needing to fetch volume data for known stocks
KNOWN_SYMBOL_TIERS = {
    # TIER 1 - HIGH LIQUIDITY (Mega caps, major ETFs)
    'SPY': LiquidityTier.HIGH,
    'QQQ': LiquidityTier.HIGH,
    'IWM': LiquidityTier.HIGH,
    'AAPL': LiquidityTier.HIGH,
    'MSFT': LiquidityTier.HIGH,
    'NVDA': LiquidityTier.HIGH,
    'AMZN': LiquidityTier.HIGH,
    'GOOGL': LiquidityTier.HIGH,
    'GOOG': LiquidityTier.HIGH,
    'META': LiquidityTier.HIGH,
    'TSLA': LiquidityTier.HIGH,
    'AMD': LiquidityTier.HIGH,
    'NFLX': LiquidityTier.HIGH,
    'INTC': LiquidityTier.HIGH,
    'AVGO': LiquidityTier.HIGH,
    'MU': LiquidityTier.HIGH,
    'BAC': LiquidityTier.HIGH,
    'JPM': LiquidityTier.HIGH,
    'WMT': LiquidityTier.HIGH,
    'XOM': LiquidityTier.HIGH,
    'V': LiquidityTier.HIGH,
    'MA': LiquidityTier.HIGH,
    'PG': LiquidityTier.HIGH,
    'JNJ': LiquidityTier.HIGH,
    'CVX': LiquidityTier.HIGH,
    'LLY': LiquidityTier.HIGH,
    'UNH': LiquidityTier.HIGH,
    'HD': LiquidityTier.HIGH,
    'PFE': LiquidityTier.HIGH,
    'ABBV': LiquidityTier.HIGH,

    # TIER 2 - MEDIUM LIQUIDITY (Most S&P 100)
    'ORCL': LiquidityTier.MEDIUM,
    'SLB': LiquidityTier.MEDIUM,
    'KO': LiquidityTier.MEDIUM,
    'MRK': LiquidityTier.MEDIUM,
    'C': LiquidityTier.MEDIUM,
    'CMCSA': LiquidityTier.MEDIUM,
    'VZ': LiquidityTier.MEDIUM,
    'LRCX': LiquidityTier.MEDIUM,
    'NEE': LiquidityTier.MEDIUM,
    'SCHW': LiquidityTier.MEDIUM,
    'SBUX': LiquidityTier.MEDIUM,
    'CRM': LiquidityTier.MEDIUM,
    'BSX': LiquidityTier.MEDIUM,
    'APH': LiquidityTier.MEDIUM,
    'MDLZ': LiquidityTier.MEDIUM,
    'AMAT': LiquidityTier.MEDIUM,
    'COP': LiquidityTier.MEDIUM,
    'TXN': LiquidityTier.MEDIUM,
    'QCOM': LiquidityTier.MEDIUM,
    'GILD': LiquidityTier.MEDIUM,
    'BA': LiquidityTier.MEDIUM,
    'PEP': LiquidityTier.MEDIUM,
    'PM': LiquidityTier.MEDIUM,
    'PANW': LiquidityTier.MEDIUM,
    'GE': LiquidityTier.MEDIUM,
    'ADBE': LiquidityTier.MEDIUM,
    'ABT': LiquidityTier.MEDIUM,
    'DHR': LiquidityTier.MEDIUM,
    'SO': LiquidityTier.MEDIUM,
    'TMUS': LiquidityTier.MEDIUM,
    'IBM': LiquidityTier.MEDIUM,
    'TMO': LiquidityTier.MEDIUM,
    'NOW': LiquidityTier.MEDIUM,
    'ACN': LiquidityTier.MEDIUM,
    'UNP': LiquidityTier.MEDIUM,
    'BRK.B': LiquidityTier.MEDIUM,
    'ETN': LiquidityTier.MEDIUM,
    'RTX': LiquidityTier.MEDIUM,
    'ADI': LiquidityTier.MEDIUM,
    'TJX': LiquidityTier.MEDIUM,
    'MCD': LiquidityTier.MEDIUM,
    'INTU': LiquidityTier.MEDIUM,
    'ISRG': LiquidityTier.MEDIUM,
    'AXP': LiquidityTier.MEDIUM,
    'COST': LiquidityTier.MEDIUM,
    'PGR': LiquidityTier.MEDIUM,
    'CAT': LiquidityTier.MEDIUM,
    'HON': LiquidityTier.MEDIUM,
    'LOW': LiquidityTier.MEDIUM,
    'LIN': LiquidityTier.MEDIUM,
    'DE': LiquidityTier.MEDIUM,
    'PLD': LiquidityTier.MEDIUM,

    # TIER 3 - LOW LIQUIDITY (Smaller large caps)
    'AMGN': LiquidityTier.LOW,
    'ELV': LiquidityTier.LOW,
    'AMT': LiquidityTier.LOW,
    'MMM': LiquidityTier.LOW,
    'BDX': LiquidityTier.LOW,
    'CME': LiquidityTier.LOW,
    'SYK': LiquidityTier.LOW,
    'DUK': LiquidityTier.LOW,
    'MMC': LiquidityTier.LOW,
    'CI': LiquidityTier.LOW,
    'SPGI': LiquidityTier.LOW,
    'VRTX': LiquidityTier.LOW,
    'CB': LiquidityTier.LOW,
    'REGN': LiquidityTier.LOW,
    'NOC': LiquidityTier.LOW,
    'BKNG': LiquidityTier.LOW,
    'BLK': LiquidityTier.LOW,
    'AON': LiquidityTier.LOW,
    'GD': LiquidityTier.LOW,
    'ITW': LiquidityTier.LOW,
}


def get_symbol_tier(symbol: str, avg_volume: float = None) -> LiquidityTier:
    """
    Get liquidity tier for a symbol.

    Uses predefined tier for known symbols, otherwise classifies by volume.

    Args:
        symbol: Stock ticker
        avg_volume: Average daily volume (optional, for unknown symbols)

    Returns:
        LiquidityTier enum value

    Example:
        >>> get_symbol_tier('AAPL')
        LiquidityTier.HIGH

        >>> get_symbol_tier('UNKNOWN_STOCK', avg_volume=5_000_000)
        LiquidityTier.MEDIUM
    """
    # Check if we have a predefined tier
    if symbol in KNOWN_SYMBOL_TIERS:
        return KNOWN_SYMBOL_TIERS[symbol]

    # Otherwise classify by volume
    if avg_volume is not None:
        return classify_by_volume(avg_volume)

    # Default to MEDIUM if we don't have volume data
    logger.warning(f"Unknown symbol {symbol}, no volume data - defaulting to MEDIUM tier")
    return LiquidityTier.MEDIUM


def get_tier_summary() -> Dict[str, List[str]]:
    """
    Get summary of known symbols by tier.

    Returns:
        Dictionary mapping tier names to lists of symbols
    """
    summary = {
        'HIGH': [],
        'MEDIUM': [],
        'LOW': [],
        'ULTRA_LOW': []
    }

    for symbol, tier in KNOWN_SYMBOL_TIERS.items():
        summary[tier.name].append(symbol)

    return summary


def print_tier_configuration():
    """Print liquidity tier configuration for reference."""
    print("=" * 80)
    print("LIQUIDITY TIER CONFIGURATION")
    print("=" * 80)
    print("\nTIER DEFINITIONS:")
    print(f"  Tier 1 (HIGH): {VOLUME_THRESHOLDS[LiquidityTier.HIGH]:,}+ shares/day")
    print(f"    → Backfill: {BACKFILL_DAYS_BY_TIER[LiquidityTier.HIGH]} days")
    print(f"  Tier 2 (MEDIUM): {VOLUME_THRESHOLDS[LiquidityTier.MEDIUM]:,}-{VOLUME_THRESHOLDS[LiquidityTier.HIGH]-1:,} shares/day")
    print(f"    → Backfill: {BACKFILL_DAYS_BY_TIER[LiquidityTier.MEDIUM]} days")
    print(f"  Tier 3 (LOW): {VOLUME_THRESHOLDS[LiquidityTier.LOW]:,}-{VOLUME_THRESHOLDS[LiquidityTier.MEDIUM]-1:,} shares/day")
    print(f"    → Backfill: {BACKFILL_DAYS_BY_TIER[LiquidityTier.LOW]} days")
    print(f"  Tier 4 (ULTRA_LOW): <{VOLUME_THRESHOLDS[LiquidityTier.LOW]:,} shares/day")
    print(f"    → EXCLUDED (not suitable for ML trading)")

    summary = get_tier_summary()
    print("\n" + "=" * 80)
    print("KNOWN SYMBOLS BY TIER:")
    print("=" * 80)
    for tier_name in ['HIGH', 'MEDIUM', 'LOW']:
        symbols = summary[tier_name]
        print(f"\n{tier_name} ({len(symbols)} symbols):")
        print("  " + ", ".join(sorted(symbols)))

    print("\n" + "=" * 80)


if __name__ == '__main__':
    # Print configuration when run directly
    print_tier_configuration()
