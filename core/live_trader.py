"""
Live Trading Script for RiskLabAI-Powered Trading Bot

This script runs the state-of-the-art RiskLabAI trading strategy with real-time
data on Alpaca's paper/live trading platform.

The RiskLabAI bot:
- Uses information-driven data structures (dollar bars, volume bars, imbalance bars)
- Applies triple-barrier labeling with volatility-adaptive targets
- Employs meta-labeling for intelligent bet sizing
- Generates stationary features via fractional differentiation
- Filters events using CUSUM for meaningful signals
- Validates with purged K-fold cross-validation
- Optimizes portfolios using Hierarchical Risk Parity (HRP)

Based on cutting-edge research from Marcos López de Prado:
- "Advances in Financial Machine Learning"
- "Machine Learning for Asset Managers"

Usage:
    # Set environment variables first:
    export ALPACA_API_KEY="your_key"
    export ALPACA_API_SECRET="your_secret"

    # Run the bot (paper trading):
    python core/live_trader.py --paper

    # Run live trading (REAL MONEY - use with caution):
    python core/live_trader.py --live

    # Check account status only:
    python core/live_trader.py --check-only
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lumibot.brokers import Alpaca
from lumibot.traders import Trader

# Import RiskLabAI strategy (THE ONLY STRATEGY NOW)
from core.risklabai_combined import RiskLabAICombined

# Import connection manager
from utils.connection_manager import get_connection_manager

# Logs directory
LOGS_DIR = PROJECT_ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True)

# Configure logging
log_file = LOGS_DIR / f'risklabai_live_{datetime.now().strftime("%Y%m%d")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"Logging to: {log_file}")


class RiskLabAILiveTrader:
    """
    Manages live trading with RiskLabAI strategy on Alpaca.

    This is THE trading system - all old strategies have been replaced
    with RiskLabAI's cutting-edge financial ML techniques.
    """

    def __init__(self, paper_trading: bool = True, trading_symbols: list = None):
        """
        Initialize RiskLabAI live trader.

        Args:
            paper_trading: True for paper trading, False for live trading
            trading_symbols: List of symbols to trade (default: ['SPY', 'QQQ', 'IWM'])
        """
        self.paper_trading = paper_trading
        self.trading_symbols = trading_symbols or ['SPY', 'QQQ', 'IWM']

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
            "PAPER": paper_trading,
            "stream_timeout": 300,  # 5 min timeout
            "stream_reconnect": False,  # Disable auto-reconnect
        }

        logger.info("=" * 80)
        logger.info("RISKLABAI LIVE TRADER INITIALIZATION")
        logger.info("=" * 80)
        logger.info(f"Trading Mode: {'PAPER TRADING' if paper_trading else '🚨 LIVE TRADING (REAL MONEY) 🚨'}")
        logger.info(f"Symbols: {', '.join(self.trading_symbols)}")
        logger.info(f"Start Time: {datetime.now()}")
        logger.info("=" * 80)

        if not paper_trading:
            logger.warning("🚨" * 20)
            logger.warning("LIVE TRADING MODE - REAL MONEY AT RISK")
            logger.warning("🚨" * 20)

    def run(self):
        """
        Start live trading with RiskLabAI strategy.

        This will:
        1. Connect to Alpaca
        2. Initialize RiskLabAI strategy
        3. Start the trading loop
        4. Run indefinitely until manually stopped
        """
        logger.info("Starting RiskLabAI trading system...")

        conn_manager = None

        try:
            # Create broker connection
            broker = Alpaca(self.alpaca_config)

            # Initialize connection manager
            conn_manager = get_connection_manager(broker)

            # Set up RiskLabAI strategy parameters
            parameters = {
                'symbols': self.trading_symbols,
                'min_training_bars': 500,  # Need sufficient data for training
                'retrain_days': 7,  # Retrain weekly
                'model_path': 'models/risklabai_models.pkl'
            }

            # Create RiskLabAI strategy instance
            strategy = RiskLabAICombined(broker=broker, parameters=parameters)

            # Create trader
            trader = Trader()
            trader.add_strategy(strategy)

            logger.info("=" * 80)
            logger.info("🚀 RISKLABAI TRADING BOT STARTED 🚀")
            logger.info("=" * 80)
            logger.info("Using cutting-edge financial ML from Marcos López de Prado")
            logger.info("Features:")
            logger.info("  ✓ Information-driven data structures")
            logger.info("  ✓ Triple-barrier labeling")
            logger.info("  ✓ Meta-labeling for bet sizing")
            logger.info("  ✓ Fractional differentiation")
            logger.info("  ✓ CUSUM event filtering")
            logger.info("  ✓ Purged K-fold cross-validation")
            logger.info("  ✓ Hierarchical Risk Parity (HRP)")
            logger.info("=" * 80)
            logger.info("The bot checks for opportunities every hour.")
            logger.info("Press Ctrl+C to stop.")
            logger.info("=" * 80)

            # Run the strategy
            trader.run_all()

        except KeyboardInterrupt:
            logger.info("\n" + "=" * 80)
            logger.info("RISKLABAI BOT STOPPED BY USER")
            logger.info("=" * 80)
            logger.info(f"Stop Time: {datetime.now()}")
            logger.info("All positions have been maintained.")
            logger.info("=" * 80)

        except Exception as e:
            logger.error("=" * 80)
            logger.error("CRITICAL ERROR IN RISKLABAI TRADING BOT")
            logger.error("=" * 80)
            logger.error(f"Error: {e}")
            logger.exception("Full traceback:")
            logger.error("=" * 80)
            raise

        finally:
            # Cleanup connections
            if conn_manager:
                logger.info("\nCleaning up connections...")
                conn_manager.cleanup_all()


def check_account_status():
    """
    Check Alpaca account status before starting.

    Verifies:
    - Credentials are correct
    - Account is active
    - Sufficient buying power
    """
    logger.info("Checking Alpaca account status...")

    try:
        from alpaca.trading.client import TradingClient

        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_API_SECRET')

        if not api_key or not api_secret:
            logger.error("Alpaca credentials not found in environment!")
            return False

        # Create client
        client = TradingClient(api_key, api_secret, paper=True)

        # Get account info
        account = client.get_account()

        logger.info("=" * 80)
        logger.info("ALPACA ACCOUNT STATUS")
        logger.info("=" * 80)
        logger.info(f"Account Number: {account.account_number}")
        logger.info(f"Status: {account.status}")
        logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
        logger.info(f"Portfolio Value: ${float(account.equity):,.2f}")
        logger.info(f"Cash: ${float(account.cash):,.2f}")
        logger.info("=" * 80)

        if account.status.value != 'ACTIVE':
            logger.warning(f"⚠️  Account status is {account.status}, not ACTIVE!")
            logger.warning("Trading may not work properly.")
            return False

        if float(account.buying_power) < 1000:
            logger.warning(f"⚠️  Low buying power: ${float(account.buying_power):,.2f}")
            logger.warning("Consider funding account for better trading.")

        return True

    except Exception as e:
        logger.error(f"Error checking account status: {e}")
        logger.error("Please verify Alpaca credentials are correct.")
        return False


def main():
    """Main entry point for RiskLabAI live trading."""
    parser = argparse.ArgumentParser(
        description='Run RiskLabAI-powered trading bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Paper trading (safe - no real money)
  python core/live_trader.py --paper

  # Live trading (REAL MONEY - use with caution)
  python core/live_trader.py --live

  # Custom symbols
  python core/live_trader.py --paper --symbols SPY QQQ IWM AAPL MSFT

  # Check account status only
  python core/live_trader.py --check-only
        """
    )

    parser.add_argument(
        '--paper',
        action='store_true',
        help='Run in paper trading mode (recommended for testing)'
    )
    parser.add_argument(
        '--live',
        action='store_true',
        help='Run in live trading mode (REAL MONEY)'
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['SPY', 'QQQ', 'IWM'],
        help='Symbols to trade (default: SPY QQQ IWM)'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check account status, don\'t start trading'
    )

    args = parser.parse_args()

    # Determine trading mode
    if args.live and args.paper:
        logger.error("Cannot specify both --paper and --live. Choose one.")
        return

    if not args.live and not args.paper and not args.check_only:
        logger.error("Must specify --paper or --live (or --check-only)")
        logger.info("For safety, paper trading is NOT the default.")
        logger.info("Run with --paper to test, or --live for real money trading.")
        return

    paper_trading = args.paper or args.check_only

    # Check account status
    if not check_account_status():
        logger.error("Account check failed. Please fix issues before starting bot.")
        return

    if args.check_only:
        logger.info("Account check complete. Exiting (--check-only flag was set).")
        return

    # Final confirmation for live trading
    if args.live:
        logger.warning("=" * 80)
        logger.warning("🚨 LIVE TRADING MODE - REAL MONEY AT RISK 🚨")
        logger.warning("=" * 80)
        logger.warning("This will trade with REAL MONEY on your Alpaca account.")
        logger.warning("Make sure you understand the risks and have tested thoroughly.")
        logger.warning("=" * 80)

        response = input("Type 'CONFIRM' to proceed with live trading: ")
        if response.strip() != 'CONFIRM':
            logger.info("Live trading cancelled.")
            return

    # Create and run trader
    trader = RiskLabAILiveTrader(
        paper_trading=paper_trading,
        trading_symbols=args.symbols
    )
    trader.run()


if __name__ == "__main__":
    main()
