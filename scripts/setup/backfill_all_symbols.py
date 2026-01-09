#!/usr/bin/env python3
"""
Multi-Symbol Tick Data Backfill

Downloads historical tick data for ALL US stocks.
Runs backfill_ticks.py for each symbol in parallel for efficiency.

Usage:
    python scripts/setup/backfill_all_symbols.py [--tier TIER] [--parallel N]

Arguments:
    --tier: Which tier to backfill (tier_1, tier_2, tier_3, tier_4, tier_5)
            tier_1: Top 100 most liquid (START HERE)
            tier_2: Top 500 (S&P 500 level)
            tier_3: Top 1000 (Russell 1000)
            tier_4: Top 2000
            tier_5: ALL liquid stocks
            Default: tier_1
    --parallel: Number of parallel downloads (default: 8)
    --days: How many days of history to download (default: 365)
"""

import sys
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def backfill_symbol(symbol: str, days: int = 365) -> dict:
    """
    Run backfill_ticks.py for a single symbol.

    Args:
        symbol: Stock/ETF symbol
        days: Number of days to backfill

    Returns:
        dict with symbol, success status, and message
    """
    logger.info(f"[{symbol}] Starting tick data backfill ({days} days)...")

    try:
        # Run backfill_ticks.py with symbol argument
        cmd = [
            sys.executable,
            'scripts/setup/backfill_ticks.py',
            '--symbol', symbol,
            '--days', str(days)
        ]

        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout per symbol
        )

        if result.returncode == 0:
            logger.info(f"[{symbol}] ✓ Backfill completed successfully")
            return {
                'symbol': symbol,
                'success': True,
                'message': 'Backfill completed'
            }
        else:
            logger.error(f"[{symbol}] ✗ Backfill failed: {result.stderr}")
            return {
                'symbol': symbol,
                'success': False,
                'message': result.stderr
            }

    except subprocess.TimeoutExpired:
        logger.error(f"[{symbol}] ✗ Backfill timed out after 30 minutes")
        return {
            'symbol': symbol,
            'success': False,
            'message': 'Timeout'
        }
    except Exception as e:
        logger.error(f"[{symbol}] ✗ Backfill error: {e}")
        return {
            'symbol': symbol,
            'success': False,
            'message': str(e)
        }


def main():
    """Run multi-symbol backfill."""
    parser = argparse.ArgumentParser(description='Backfill tick data for all US stocks')
    parser.add_argument('--tier', type=str, default='tier_1',
                        choices=['tier_1', 'tier_2', 'tier_3', 'tier_4', 'tier_5'],
                        help='Which tier to backfill (tier_1=top 100, tier_5=all)')
    parser.add_argument('--parallel', type=int, default=8,
                        help='Number of parallel downloads')
    parser.add_argument('--days', type=int, default=365,
                        help='Days of history to download')

    args = parser.parse_args()

    # Import and get symbols for this tier
    try:
        from config.all_symbols import get_symbols_by_tier
        symbols = get_symbols_by_tier(args.tier)
    except ImportError:
        logger.error("all_symbols.py not found!")
        logger.error("Run: python scripts/setup/fetch_all_symbols.py first")
        return 1

    logger.info("=" * 80)
    logger.info(f"BACKFILLING ALL US STOCKS - {args.tier.upper()}")
    logger.info("=" * 80)
    logger.info(f"Symbols to backfill: {len(symbols)}")
    logger.info(f"First 10 symbols: {', '.join(symbols[:10])}")
    logger.info(f"Parallel downloads: {args.parallel}")
    logger.info(f"Days of history: {args.days}")
    logger.info("=" * 80)

    # Run backfills in parallel
    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        # Submit all tasks
        futures = {
            executor.submit(backfill_symbol, symbol, args.days): symbol
            for symbol in symbols
        }

        # Collect results as they complete
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"[{symbol}] Unexpected error: {e}")
                results.append({
                    'symbol': symbol,
                    'success': False,
                    'message': str(e)
                })

    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("BACKFILL SUMMARY")
    logger.info("=" * 80)

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    logger.info(f"Total symbols: {len(results)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")

    if successful:
        logger.info("")
        logger.info("✓ Successfully backfilled:")
        for r in successful:
            logger.info(f"  - {r['symbol']}")

    if failed:
        logger.info("")
        logger.error("✗ Failed to backfill:")
        for r in failed:
            logger.error(f"  - {r['symbol']}: {r['message']}")

    logger.info("=" * 80)

    # Return non-zero exit code if any failed
    return 0 if len(failed) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
