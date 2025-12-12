#!/usr/bin/env python
"""
Test script to manually trigger a trading iteration immediately.
This allows you to test the bot without waiting for the next scheduled iteration.
"""

import os
import logging
from lumibot.brokers import Alpaca
from combined_strategy import CombinedStrategy

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG to see position attributes
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Get API credentials
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_API_SECRET')

    if not api_key or not api_secret:
        print("❌ Error: Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables")
        return

    print("=" * 80)
    print("MANUAL ITERATION TEST")
    print("=" * 80)
    print("This will run ONE iteration of the trading strategy immediately.")
    print("=" * 80)
    print()

    # Create broker connection
    alpaca_config = {
        "API_KEY": api_key,
        "API_SECRET": api_secret,
        "PAPER": True,
    }
    broker = Alpaca(alpaca_config)

    # Create strategy instance with broker
    parameters = {
        'db_path': '/Volumes/Vault/85_assets_prediction.db',
        'retrain': False
    }

    print("Initializing strategy...")
    strategy = CombinedStrategy(broker=broker)

    # Initialize with parameters
    strategy.initialize(parameters)

    print()
    print("=" * 80)
    print("RUNNING TRADING ITERATION NOW")
    print("=" * 80)
    print()

    # Manually trigger one iteration
    try:
        strategy.on_trading_iteration()
    except Exception as e:
        logger.error(f"Error during iteration: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 80)
    print("ITERATION COMPLETE")
    print("=" * 80)
    print()
    print("Check the output above to see:")
    print("  - Portfolio value")
    print("  - Risk management checks")
    print("  - Any triggered stop-loss or take-profit orders")
    print("  - New trading signals")
    print()

if __name__ == "__main__":
    main()
