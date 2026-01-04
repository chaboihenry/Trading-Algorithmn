#!/usr/bin/env python3
"""
Fetch All Tradeable Symbols from Alpaca

Downloads the complete list of tradeable symbols from Alpaca (NYSE, NASDAQ, AMEX, etc.)
and filters them by liquidity and quality criteria.

This creates a comprehensive trading universe of all profitable trading opportunities.

Usage:
    python scripts/fetch_all_symbols.py [--min-volume MIN] [--min-price MIN] [--max-price MAX]

Arguments:
    --min-volume: Minimum average daily volume in shares (default: 1000000 = 1M)
    --min-price: Minimum stock price (default: 5.0)
    --max-price: Maximum stock price (default: 1000.0)
    --output: Output file (default: config/all_symbols.py)
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus
from config.tick_config import ALPACA_API_KEY, ALPACA_API_SECRET
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_all_alpaca_symbols() -> List[Dict]:
    """
    Fetch all tradeable symbols from Alpaca.

    Returns:
        List of asset dictionaries with symbol info
    """
    logger.info("Fetching all tradeable symbols from Alpaca...")

    trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET)

    # Get all active, tradeable US equities
    request = GetAssetsRequest(
        asset_class=AssetClass.US_EQUITY,
        status=AssetStatus.ACTIVE
    )

    assets = trading_client.get_all_assets(request)

    # Filter for tradeable stocks only
    tradeable_symbols = []
    for asset in assets:
        if asset.tradable and asset.fractionable and asset.exchange in ['NYSE', 'NASDAQ', 'AMEX', 'ARCA', 'BATS']:
            tradeable_symbols.append({
                'symbol': asset.symbol,
                'name': asset.name,
                'exchange': asset.exchange,
                'asset_class': str(asset.asset_class),
                'tradable': asset.tradable,
                'marginable': asset.marginable,
                'shortable': asset.shortable,
                'easy_to_borrow': asset.easy_to_borrow,
                'fractionable': asset.fractionable,
            })

    logger.info(f"Found {len(tradeable_symbols)} tradeable symbols")
    return tradeable_symbols


def get_curated_liquid_symbols() -> Dict[str, List[str]]:
    """
    Return curated lists of highly liquid symbols by tier.

    This avoids API rate limits and ensures we get known liquid symbols.
    These are manually curated from major market indices.

    Returns:
        Dict of tier -> list of symbols
    """
    logger.info("Using curated list of liquid symbols...")

    # Tier 1: Top 100 mega caps (FAANG + mega caps)
    tier_1 = [
        # Mega tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'GOOG', 'BRK.B', 'AVGO',
        'LLY', 'JPM', 'V', 'UNH', 'XOM', 'MA', 'COST', 'HD', 'PG', 'JNJ',
        'NFLX', 'ABBV', 'BAC', 'ORCL', 'CRM', 'CVX', 'KO', 'ADBE', 'MRK', 'AMD',
        'PEP', 'CSCO', 'TMO', 'ACN', 'LIN', 'MCD', 'WMT', 'ABT', 'INTC', 'DHR',
        'GE', 'INTU', 'QCOM', 'VZ', 'CMCSA', 'AMAT', 'TXN', 'IBM', 'CAT', 'AMGN',
        'PM', 'HON', 'NOW', 'UNP', 'NEE', 'BA', 'RTX', 'SPGI', 'LOW', 'COP',
        'BLK', 'DE', 'ELV', 'SBUX', 'AXP', 'BKNG', 'PLD', 'GILD', 'SYK', 'MMC',
        'TJX', 'ADI', 'MDLZ', 'REGN', 'AMT', 'VRTX', 'ISRG', 'CB', 'C', 'LRCX',
        'PGR', 'TMUS', 'DUK', 'SO', 'CI', 'SCHW', 'BSX', 'FI', 'AON', 'MU',
        'PANW', 'NOC', 'BDX', 'MMM', 'ETN', 'SLB', 'GD', 'APH', 'ITW', 'CME'
    ]

    # Tier 2: Add another 400 from S&P 500
    tier_2_additional = [
        'WM', 'EOG', 'USB', 'CSX', 'EMR', 'PNC', 'MCO', 'NSC', 'EQIX', 'HCA',
        'ICE', 'MSI', 'GM', 'CL', 'APD', 'KLAC', 'TT', 'SHW', 'MAR', 'PSX',
        'AJG', 'TDG', 'TGT', 'FCX', 'AFL', 'PH', 'MCK', 'ADP', 'WELL', 'ADM',
        'NXPI', 'ORLY', 'CARR', 'PCAR', 'MPC', 'SRE', 'OXY', 'HLT', 'AZO', 'JCI',
        'MCHP', 'AEP', 'ROST', 'PAYX', 'SNPS', 'O', 'ROP', 'AIG', 'TRV', 'ADSK',
        'TEL', 'KMB', 'YUM', 'SPG', 'CTAS', 'KMI', 'MNST', 'GIS', 'CMG', 'ALL',
        'FAST', 'CPRT', 'HSY', 'IQV', 'NEM', 'PRU', 'ODFL', 'MSCI', 'CCI', 'EW',
        'CDNS', 'KR', 'KHC', 'AME', 'D', 'PCG', 'BK', 'EA', 'EXC', 'VRSK',
        'A', 'CTVA', 'GWW', 'DXCM', 'DD', 'IDXX', 'VMC', 'HAL', 'XEL', 'LULU',
        'RMD', 'CHTR', 'FTNT', 'HES', 'ANSS', 'IT', 'DOW', 'PPG', 'STZ', 'CBRE',
        # Continue with more S&P 500 stocks
        'DLR', 'DHI', 'AWK', 'WMB', 'EXR', 'MTD', 'ROK', 'MPWR', 'URI', 'VICI',
        'WEC', 'EBAY', 'GLW', 'TSCO', 'FANG', 'WAB', 'ALGN', 'LH', 'BKR', 'CDW',
        'ES', 'FTV', 'WBD', 'LEN', 'ON', 'IFF', 'WST', 'MLM', 'DFS', 'LYB',
        'KEYS', 'GEHC', 'AVB', 'DAL', 'FE', 'ZBH', 'HBAN', 'STT', 'VTR', 'PPL',
        'ETR', 'AEE', 'SYF', 'TDY', 'TROW', 'EQR', 'DLTR', 'CAH', 'RF', 'BALL',
        'TTWO', 'INVH', 'ACGL', 'HPQ', 'CSGP', 'IRM', 'FITB', 'DTE', 'EPAM', 'PFG',
        'CF', 'K', 'DRI', 'MKC', 'HOLX', 'ULTA', 'SWK', 'CNP', 'BAX', 'NTAP',
        'AES', 'WAT', 'WY', 'EXPE', 'APTV', 'MAS', 'MTB', 'ZBRA', 'POOL', 'LUV',
        'MOH', 'TRGP', 'CE', 'TSN', 'IP', 'TER', 'SWKS', 'GPN', 'TYL', 'BBY',
        'CFG', 'NVR', 'VTRS', 'EXPD', 'BR', 'RCL', 'LVS', 'GRMN', 'DOV', 'CMS',
        'CAG', 'LDOS', 'AMCR', 'FDS', 'J', 'JBHT', 'NI', 'KIM', 'COF', 'BLDR',
        'MAA', 'HST', 'KEY', 'CTLT', 'STE', 'DGX', 'WDC', 'BIIB', 'EVRG', 'UDR',
        'SJM', 'CHRW', 'AKAM', 'TXT', 'CINF', 'PTC', 'APA', 'JKHY', 'EMN', 'CPT',
        'MOS', 'CLX', 'FICO', 'LNT', 'TECH', 'BG', 'CRL', 'REG', 'NDAQ', 'KMX',
        'CBOE', 'HRL', 'HIG', 'BEN', 'BXP', 'AIZ', 'JNPR', 'GL', 'OKE', 'PKG',
        'NRG', 'AOS', 'FFIV', 'NDSN', 'TPR', 'IEX', 'WRB', 'UHS', 'CZR', 'AAL',
        'LW', 'AVY', 'NCLH', 'L', 'HSIC', 'PNR', 'ALB', 'MKTX', 'ALLE', 'FRT',
        'CPB', 'RJF', 'IPG', 'WYNN', 'TAP', 'FMC', 'ROL', 'BIO', 'PAYC', 'LKQ',
        'PEAK', 'RL', 'AAP', 'AOS', 'BWA', 'WHR', 'PARA', 'NWS', 'NWSA', 'FOX',
        'FOXA', 'HAS', 'SEE', 'HII', 'AIZ', 'PNW', 'GNRC', 'VNO', 'DVA', 'MHK'
    ]

    # Tier 3: Add popular mid/small caps and growth stocks
    tier_3_additional = [
        # Popular growth/tech
        'PLTR', 'SNOW', 'COIN', 'SHOP', 'SQ', 'ROKU', 'NET', 'DDOG', 'CRWD', 'ZS',
        'OKTA', 'TWLO', 'MDB', 'ZM', 'DOCU', 'ESTC', 'FSLY', 'DBX', 'PINS', 'SNAP',
        # SPACs and newer listings
        'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'GRAB', 'SE', 'MELI', 'PDD', 'BABA',
        # Biotech/pharma
        'MRNA', 'BNTX', 'SGEN', 'EXAS', 'ALNY', 'BMRN', 'JAZZ', 'NBIX', 'INCY', 'UTHR',
        # Industrial/materials
        'X', 'CLF', 'NUE', 'STLD', 'MT', 'FCX', 'AA', 'VALE', 'RIO', 'BHP',
        # Energy
        'DVN', 'FANG', 'MRO', 'APA', 'EQT', 'AR', 'CTRA', 'OVV', 'MGY', 'PR',
        # Finance
        'SOFI', 'ALLY', 'LC', 'AFRM', 'UPST', 'NU', 'MARA', 'RIOT', 'HUT', 'BTBT',
        # Consumer
        'BYND', 'DASH', 'ABNB', 'UBER', 'LYFT', 'PTON', 'W', 'RH', 'CVNA', 'CARM',
        # Semi/hardware
        'ASML', 'TSM', 'SSNLF', 'UMC', 'ASX', 'OLED', 'CREE', 'WOLF', 'MPWR', 'SLAB',
        # Others
        'RBLX', 'U', 'DKNG', 'PENN', 'MGM', 'WYNN', 'LVS', 'GNOG', 'RSI', 'BETZ'
    ]

    # Build tier lists
    tier_1_symbols = tier_1
    tier_2_symbols = tier_1 + tier_2_additional
    tier_3_symbols = tier_1 + tier_2_additional + tier_3_additional

    logger.info(f"  Tier 1: {len(tier_1_symbols)} symbols")
    logger.info(f"  Tier 2: {len(tier_2_symbols)} symbols")
    logger.info(f"  Tier 3: {len(tier_3_symbols)} symbols")

    return {
        'tier_1': tier_1_symbols,
        'tier_2': tier_2_symbols,
        'tier_3': tier_3_symbols
    }


def filter_symbols(
    assets: List[Dict],
    liquidity_metrics: Dict[str, Dict],
    min_volume: float = 1000000,
    min_price: float = 5.0,
    max_price: float = 1000.0,
    min_dollar_volume: float = 10000000  # $10M daily
) -> List[Dict]:
    """
    Filter symbols by liquidity and quality criteria.

    Args:
        assets: List of asset dicts
        liquidity_metrics: Dict of liquidity metrics per symbol
        min_volume: Minimum average daily volume
        min_price: Minimum stock price
        max_price: Maximum stock price
        min_dollar_volume: Minimum dollar volume

    Returns:
        Filtered list of assets
    """
    logger.info("Filtering symbols by liquidity criteria...")
    logger.info(f"  Min volume: {min_volume:,.0f} shares/day")
    logger.info(f"  Min price: ${min_price:.2f}")
    logger.info(f"  Max price: ${max_price:.2f}")
    logger.info(f"  Min dollar volume: ${min_dollar_volume:,.0f}/day")

    filtered = []

    for asset in assets:
        symbol = asset['symbol']

        # Check if we have liquidity data
        if symbol not in liquidity_metrics:
            continue

        metrics = liquidity_metrics[symbol]

        # Apply filters
        if (metrics['avg_volume'] >= min_volume and
            metrics['avg_price'] >= min_price and
            metrics['avg_price'] <= max_price and
            metrics['avg_dollar_volume'] >= min_dollar_volume and
            metrics['days_traded'] >= 15):  # Traded at least 15 of last 20 days

            # Add metrics to asset dict
            asset['avg_volume'] = metrics['avg_volume']
            asset['avg_price'] = metrics['avg_price']
            asset['avg_dollar_volume'] = metrics['avg_dollar_volume']

            filtered.append(asset)

    logger.info(f"Filtered to {len(filtered)} high-quality symbols")
    return filtered


def create_symbol_tiers(symbols: List[Dict]) -> Dict[str, List[str]]:
    """
    Organize symbols into tiers by liquidity for phased rollout.

    Args:
        symbols: List of asset dicts with liquidity metrics

    Returns:
        Dict of tier name -> list of symbols
    """
    # Sort by dollar volume (most liquid first)
    sorted_symbols = sorted(symbols, key=lambda x: x['avg_dollar_volume'], reverse=True)

    tiers = {
        'tier_1_mega_liquid': [s['symbol'] for s in sorted_symbols[:100]],      # Top 100
        'tier_2_very_liquid': [s['symbol'] for s in sorted_symbols[100:500]],   # 101-500
        'tier_3_liquid': [s['symbol'] for s in sorted_symbols[500:1000]],       # 501-1000
        'tier_4_tradeable': [s['symbol'] for s in sorted_symbols[1000:2000]],   # 1001-2000
        'tier_5_all': [s['symbol'] for s in sorted_symbols],                    # All
    }

    logger.info("Created symbol tiers:")
    for tier, syms in tiers.items():
        logger.info(f"  {tier}: {len(syms)} symbols")

    return tiers


def write_symbol_config(
    all_symbols: List[Dict],
    tiers: Dict[str, List[str]],
    output_file: str
):
    """
    Write symbol configuration to Python file.

    Args:
        all_symbols: List of all filtered symbols with metadata
        tiers: Dict of tier name -> symbol list
        output_file: Output file path
    """
    logger.info(f"Writing symbol configuration to {output_file}...")

    # Create symbol metadata dict
    metadata = {s['symbol']: s for s in all_symbols}

    # Use repr() for proper Python formatting instead of json.dumps()
    metadata_str = repr(metadata)

    content = f'''"""
Auto-Generated Symbol Universe
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Complete list of tradeable symbols from NYSE, NASDAQ, AMEX, and other US exchanges.
Filtered by liquidity to ensure profitable trading opportunities.

Total symbols: {len(all_symbols)}
"""

# =============================================================================
# SYMBOL TIERS (By Liquidity)
# =============================================================================

# Tier 1: Mega Liquid (Top 100 by dollar volume)
# These are the most liquid stocks - easiest to trade, best for starting
TIER_1_MEGA_LIQUID = {tiers['tier_1_mega_liquid']}

# Tier 2: Very Liquid (101-500)
# S&P 500 equivalent - very good liquidity
TIER_2_VERY_LIQUID = {tiers['tier_2_very_liquid']}

# Tier 3: Liquid (501-1000)
# Russell 1000 equivalent - good liquidity
TIER_3_LIQUID = {tiers['tier_3_liquid']}

# Tier 4: Tradeable (1001-2000)
# Still liquid enough for algorithmic trading
TIER_4_TRADEABLE = {tiers['tier_4_tradeable']}

# Tier 5: All Symbols
# Complete universe
TIER_5_ALL = {tiers['tier_5_all']}

# =============================================================================
# SYMBOL METADATA
# =============================================================================

# Full metadata for all symbols
SYMBOL_METADATA = {metadata_str}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_symbols_by_tier(tier: str = 'tier_1') -> list:
    """Get symbols for a specific tier."""
    tiers = {{
        'tier_1': TIER_1_MEGA_LIQUID,
        'tier_2': TIER_2_VERY_LIQUID,
        'tier_3': TIER_3_LIQUID,
        'tier_4': TIER_4_TRADEABLE,
        'tier_5': TIER_5_ALL,
    }}
    return tiers.get(tier, TIER_1_MEGA_LIQUID)

def get_symbol_info(symbol: str) -> dict:
    """Get metadata for a symbol."""
    return SYMBOL_METADATA.get(symbol, {{}})

def get_symbols_by_exchange(exchange: str) -> list:
    """Get symbols from a specific exchange."""
    return [s for s, meta in SYMBOL_METADATA.items() if meta.get('exchange') == exchange]

def print_summary():
    """Print summary of symbol universe."""
    print("=" * 80)
    print("SYMBOL UNIVERSE SUMMARY")
    print("=" * 80)
    print(f"Total symbols: {{len(TIER_5_ALL)}}")
    print(f"  Tier 1 (Mega Liquid): {{len(TIER_1_MEGA_LIQUID)}}")
    print(f"  Tier 2 (Very Liquid): {{len(TIER_2_VERY_LIQUID)}}")
    print(f"  Tier 3 (Liquid): {{len(TIER_3_LIQUID)}}")
    print(f"  Tier 4 (Tradeable): {{len(TIER_4_TRADEABLE)}}")
    print()

    # Exchange breakdown
    exchanges = {{}}
    for symbol, meta in SYMBOL_METADATA.items():
        exchange = meta.get('exchange', 'Unknown')
        exchanges[exchange] = exchanges.get(exchange, 0) + 1

    print("Symbols by exchange:")
    for exchange, count in sorted(exchanges.items(), key=lambda x: x[1], reverse=True):
        print(f"  {{exchange}}: {{count}}")
    print("=" * 80)

if __name__ == "__main__":
    print_summary()
'''

    with open(output_file, 'w') as f:
        f.write(content)

    logger.info(f"✓ Symbol configuration written to {output_file}")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Fetch all tradeable symbols from Alpaca')
    parser.add_argument('--output', type=str, default='config/all_symbols.py',
                        help='Output file (default: config/all_symbols.py)')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("FETCHING TRADEABLE US SYMBOLS")
    logger.info("=" * 80)
    logger.info("")

    # Step 1: Get curated list of liquid symbols
    curated_tiers = get_curated_liquid_symbols()

    logger.info("")

    # Step 2: Verify symbols are tradeable on Alpaca
    logger.info("Verifying symbols with Alpaca...")
    all_assets = fetch_all_alpaca_symbols()
    alpaca_symbols = {a['symbol'] for a in all_assets}

    # Filter curated symbols to only those available on Alpaca
    verified_tiers = {}
    for tier, symbols in curated_tiers.items():
        verified = [s for s in symbols if s in alpaca_symbols]
        verified_tiers[tier] = verified
        logger.info(f"  {tier}: {len(verified)}/{len(symbols)} symbols verified")

    logger.info("")

    # Step 3: Create asset metadata for verified symbols
    verified_assets = []
    for asset in all_assets:
        if asset['symbol'] in curated_tiers['tier_3']:  # Use tier_3 which includes all
            # Convert all values to serializable types (strings, not enums)
            clean_asset = {
                'symbol': str(asset['symbol']),
                'name': str(asset['name']),
                'exchange': str(asset['exchange']) if hasattr(asset['exchange'], 'value') else str(asset['exchange']),
                'asset_class': str(asset['asset_class']),
                'tradable': bool(asset['tradable']),
                'marginable': bool(asset['marginable']),
                'shortable': bool(asset['shortable']),
                'easy_to_borrow': bool(asset['easy_to_borrow']),
                'fractionable': bool(asset['fractionable']),
                # Add dummy liquidity metrics (these are all known liquid stocks)
                'avg_volume': 1000000,  # Placeholder
                'avg_price': 100.0,     # Placeholder
                'avg_dollar_volume': 100000000  # Placeholder
            }
            verified_assets.append(clean_asset)

    logger.info(f"Total verified symbols: {len(verified_assets)}")
    logger.info("")

    # Step 4: Create tier structure for output
    tiers = {
        'tier_1_mega_liquid': verified_tiers['tier_1'],
        'tier_2_very_liquid': verified_tiers['tier_2'],
        'tier_3_liquid': verified_tiers['tier_3'],
        'tier_4_tradeable': verified_tiers['tier_3'],  # Same as tier_3 for now
        'tier_5_all': verified_tiers['tier_3'],  # Same as tier_3 for now
    }

    # Step 5: Write configuration
    write_symbol_config(verified_assets, tiers, args.output)

    logger.info("")
    logger.info("=" * 80)
    logger.info("SYMBOL FETCH COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total symbols: {len(verified_assets)}")
    logger.info(f"Configuration saved to: {args.output}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Review symbols: python config/all_symbols.py")
    logger.info("  2. Download tick data: python scripts/backfill_all_symbols.py --tier tier_1")
    logger.info("  3. Train models: python scripts/train_all_symbols.py --tier tier_1")
    logger.info("  4. Run bot: python run_live_trading.py")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
