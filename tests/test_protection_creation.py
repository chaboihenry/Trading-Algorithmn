"""
Test script to create missing take-profit orders.

This script:
1. Connects to Alpaca
2. Checks all positions
3. Creates missing take-profit orders (even when market is closed)
4. Shows before/after status

Run this to fix all positions that are missing take-profit orders.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
from risk.stop_loss_manager import ensure_all_positions_protected
from data.market_data import get_market_data_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Test creating take-profit orders for all positions."""

    logger.info("=" * 80)
    logger.info("POSITION PROTECTION TEST")
    logger.info("=" * 80)

    # Get current positions
    market_data = get_market_data_client()
    positions = market_data.get_positions()

    logger.info(f"\nFound {len(positions)} positions:")
    for pos in positions:
        logger.info(f"  - {pos.symbol}: {pos.qty} shares @ ${float(pos.current_price):.2f}")

    if not positions:
        logger.info("No positions to protect!")
        return

    logger.info("\n" + "=" * 80)
    logger.info("Creating missing take-profit orders...")
    logger.info("=" * 80)

    # Create missing protection
    success = ensure_all_positions_protected()

    if success:
        logger.info("\n✅ ALL POSITIONS FULLY PROTECTED!")
        logger.info("All positions now have both stop-loss AND take-profit orders.")
    else:
        logger.warning("\n⚠️  Some positions may still lack full protection.")
        logger.warning("Check logs above for specific errors.")

    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)
    logger.info("\nRun 'python monitoring/dashboard.py' to verify all positions are protected.")


if __name__ == "__main__":
    main()
