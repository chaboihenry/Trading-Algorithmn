"""
Live Trading Script for Alpaca Paper Trading

This script runs the combined strategy with real-time data on Alpaca's
paper trading platform.

The bot:
- Wakes up once per day
- Fetches news from the last 3 days
- Runs FinBERT sentiment analysis
- Checks pairs trading opportunities
- Uses meta-learner to combine signals
- Makes trading decisions
- Sleeps for 24 hours and repeats

Usage:
    # Set environment variables first:
    export ALPACA_API_KEY="your_key"
    export ALPACA_API_SECRET="your_secret"

    # Run the bot:
    python live_trader.py

    # Or with custom strategy:
    python live_trader.py --strategy sentiment  # For sentiment-only
    python live_trader.py --strategy pairs      # For pairs-only
    python live_trader.py --strategy combined   # For combined (default)
"""

import os
import logging
import argparse
from datetime import datetime
from lumibot.brokers import Alpaca
from lumibot.traders import Trader

# Import strategies
from sentiment_strategy import SentimentStrategy
from pairs_strategy import PairsStrategy
from combined_strategy import CombinedStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'live_trading_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LiveTrader:
    """
    Manages live trading with Alpaca paper trading account.
    """

    def __init__(self, strategy_name: str = 'combined'):
        """
        Initialize live trader.

        Args:
            strategy_name: Which strategy to run ('sentiment', 'pairs', or 'combined')
        """
        self.strategy_name = strategy_name

        # Get Alpaca credentials from environment
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret = os.getenv('ALPACA_API_SECRET')

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Alpaca credentials not found!\n"
                "Please set ALPACA_API_KEY and ALPACA_API_SECRET environment variables.\n\n"
                "Example:\n"
                "  export ALPACA_API_KEY='your_key_here'\n"
                "  export ALPACA_API_SECRET='your_secret_here'\n"
            )

        # Configure Alpaca broker
        self.alpaca_config = {
            "API_KEY": self.api_key,
            "API_SECRET": self.api_secret,
            "PAPER": True,  # Paper trading (change to False for live trading)
        }

        logger.info("=" * 80)
        logger.info("LIVE TRADER INITIALIZATION")
        logger.info("=" * 80)
        logger.info(f"Strategy: {strategy_name}")
        logger.info(f"Trading Mode: Paper Trading")
        logger.info(f"Start Time: {datetime.now()}")
        logger.info("=" * 80)

    def get_strategy(self):
        """
        Get the configured strategy instance.

        Returns:
            Strategy instance
        """
        parameters = {
            'db_path': '/Volumes/Vault/85_assets_prediction.db'
        }

        if self.strategy_name == 'sentiment':
            logger.info("Initializing Sentiment Strategy")
            return SentimentStrategy()

        elif self.strategy_name == 'pairs':
            logger.info("Initializing Pairs Strategy")
            return PairsStrategy()

        elif self.strategy_name == 'combined':
            logger.info("Initializing Combined Strategy (Meta-Learner)")
            return CombinedStrategy()

        else:
            raise ValueError(f"Unknown strategy: {self.strategy_name}")

    def run(self):
        """
        Start live trading.

        This will:
        1. Connect to Alpaca
        2. Initialize the strategy
        3. Start the trading loop
        4. Run indefinitely until manually stopped
        """
        logger.info("Starting live trading...")

        try:
            # Create broker connection
            broker = Alpaca(self.alpaca_config)

            # Get strategy
            strategy = self.get_strategy()

            # Create trader
            trader = Trader()
            trader.add_strategy(strategy)

            logger.info("=" * 80)
            logger.info("TRADING BOT STARTED")
            logger.info("=" * 80)
            logger.info("The bot is now running and will check for trading opportunities every 24 hours.")
            logger.info("Press Ctrl+C to stop the bot.")
            logger.info("=" * 80)

            # Run all strategies
            trader.run_all()

        except KeyboardInterrupt:
            logger.info("\n" + "=" * 80)
            logger.info("TRADING BOT STOPPED BY USER")
            logger.info("=" * 80)
            logger.info(f"Stop Time: {datetime.now()}")
            logger.info("All positions have been maintained. Restart the bot to continue trading.")
            logger.info("=" * 80)

        except Exception as e:
            logger.error("=" * 80)
            logger.error("CRITICAL ERROR IN TRADING BOT")
            logger.error("=" * 80)
            logger.error(f"Error: {e}")
            logger.exception("Full traceback:")
            logger.error("=" * 80)
            logger.error("The bot has stopped. Please fix the error and restart.")
            logger.error("=" * 80)
            raise


def check_account_status():
    """
    Check Alpaca account status before starting.

    This helps verify:
    - Credentials are correct
    - Account is active
    - We have access to paper trading
    """
    logger.info("Checking Alpaca account status...")

    try:
        import alpaca_trade_api as tradeapi

        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_API_SECRET')
        base_url = 'https://paper-api.alpaca.markets'  # Paper trading URL

        api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

        # Get account info
        account = api.get_account()

        logger.info("-" * 80)
        logger.info("ACCOUNT STATUS")
        logger.info("-" * 80)
        logger.info(f"Account Number: {account.account_number}")
        logger.info(f"Status: {account.status}")
        logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
        logger.info(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
        logger.info(f"Cash: ${float(account.cash):,.2f}")
        logger.info("-" * 80)

        if account.status != 'ACTIVE':
            logger.warning(f"Account status is {account.status}, not ACTIVE!")
            logger.warning("Trading may not work properly.")

        return True

    except Exception as e:
        logger.error(f"Error checking account status: {e}")
        logger.error("Please verify your Alpaca credentials are correct.")
        return False


def main():
    """Main entry point for live trading."""
    parser = argparse.ArgumentParser(description='Run live trading bot')
    parser.add_argument(
        '--strategy',
        type=str,
        default='combined',
        choices=['sentiment', 'pairs', 'combined'],
        help='Which strategy to run (default: combined)'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check account status, don\'t start trading'
    )

    args = parser.parse_args()

    # Check account status first
    if not check_account_status():
        logger.error("Account check failed. Please fix issues before starting bot.")
        return

    if args.check_only:
        logger.info("Account check complete. Exiting (--check-only flag was set).")
        return

    # Create and run trader
    trader = LiveTrader(strategy_name=args.strategy)
    trader.run()


if __name__ == "__main__":
    main()
