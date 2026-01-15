"""
Universe builder for symbol selection and tiered backfill.
"""

from typing import Dict, List, Tuple, Optional
import logging

from config.all_symbols import get_symbols_by_tier, SYMBOL_METADATA
from config.liquidity_tiers import (
    get_symbol_tier,
    should_include_symbol,
    get_backfill_days,
    LiquidityTier
)

logger = logging.getLogger(__name__)


def build_universe(
    tier: str = "tier_1",
    max_symbols: Optional[int] = None
) -> Tuple[List[str], Dict[str, LiquidityTier], Dict[str, int]]:
    """
    Build a deduped universe with tier assignments and backfill days.

    Args:
        tier: Tier name from config.all_symbols (tier_1..tier_5)
        max_symbols: Optional cap on number of symbols

    Returns:
        symbols: Ordered list of symbols
        symbol_tiers: Mapping of symbol -> LiquidityTier
        backfill_days: Mapping of symbol -> backfill days
    """
    symbols = get_symbols_by_tier(tier)
    seen = set()
    deduped = []
    for symbol in symbols:
        if symbol not in seen:
            seen.add(symbol)
            deduped.append(symbol)

    symbol_tiers: Dict[str, LiquidityTier] = {}
    backfill_days: Dict[str, int] = {}
    selected: List[str] = []

    for symbol in deduped:
        avg_volume = SYMBOL_METADATA.get(symbol, {}).get("avg_volume")
        tier_value = get_symbol_tier(symbol, avg_volume=avg_volume)

        if not should_include_symbol(tier_value):
            continue

        symbol_tiers[symbol] = tier_value
        backfill_days[symbol] = get_backfill_days(tier_value)
        selected.append(symbol)

        if max_symbols and len(selected) >= max_symbols:
            break

    logger.info(
        f"Universe built: {len(selected)} symbols "
        f"(tier={tier}, max_symbols={max_symbols or 'none'})"
    )

    return selected, symbol_tiers, backfill_days
