#!/usr/bin/env python
"""
Main entry point for the live trading bot.
"""

import logging
import os
import sys
import time

import alpaca_trade_api as tradeapi

from strategies.stacked_ensemble import StackedEnsemble

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Trader:
    """
    The main trading bot class.
    """

    def __init__(self, dry_run: bool = False):
        """
        Initialize the trader.

        Args:
            dry_run: If True, simulate trades instead of executing them.
        """
        self.dry_run = dry_run
        self.running = True

        # Initialize Alpaca API
        self.api = self._init_alpaca_api()

        # Initialize the stacked ensemble strategy
        self.stacked_ensemble = StackedEnsemble(api=self.api)

    def _init_alpaca_api(self):
        """Initializes the Alpaca trade API."""
        api_key = os.environ.get("ALPACA_API_KEY")
        api_secret = os.environ.get("ALPACA_API_SECRET")
        base_url = "https://paper-api.alpaca.markets"  # Use paper trading endpoint

        if not api_key or not api_secret:
            logger.error("Alpaca API key and secret must be set as environment variables.")
            sys.exit(1)

        try:
            api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
            account = api.get_account()
            logger.info(f"Connected to Alpaca paper trading account: {account.id}")
            return api
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca API: {e}")
            sys.exit(1)

    def run(self, interval: int = 60):
        """
        The main trading loop.

        Args:
            interval: The interval in seconds to wait between each trading cycle.
        """
        logger.info("🚀 Starting live trading bot...")
        while self.running:
            try:
                self.execute_trading_cycle()
                logger.info(f"Sleeping for {interval} seconds...")
                time.sleep(interval)
            except KeyboardInterrupt:
                self.running = False
                logger.info("🛑 Live trading bot stopped by user.")
            except Exception as e:
                logger.error(f"An error occurred in the trading loop: {e}", exc_info=True)
                time.sleep(interval)  # Wait before retrying

    def execute_trading_cycle(self):
        """Executes a single cycle of the trading logic."""
        logger.info("🔄 Executing new trading cycle...")

        # 1. Generate signals from the stacked ensemble
        signals = self.stacked_ensemble.generate_signals()

        if signals.empty:
            logger.info("No trading signals generated in this cycle.")
            return

        # 2. Execute trades based on the signals
        for _, signal in signals.iterrows():
            logger.info(f"Processing signal for {signal['symbol_ticker']}")
            if not self.dry_run:
                self.place_order(signal)
            else:
                logger.info(f"[DRY RUN] Would place {signal['signal_type']} order for {signal['symbol_ticker']}")

    def place_order(self, signal):
        """
        Places a trade order on Alpaca.

        Args:
            signal: The trading signal object.
        """
        try:
            self.api.submit_order(
                symbol=signal['symbol_ticker'],
                qty=1,  # For now, trade 1 share
                side=signal['signal_type'].lower(),
                type='market',
                time_in_force='gtc'
            )
            logger.info(f"Placed {signal['signal_type']} order for 1 share of {signal['symbol_ticker']}")
        except Exception as e:
            logger.error(f"Failed to place order for {signal['symbol_ticker']}: {e}")

def main():
    import argparse
    from backtester import Backtester
    from strategies.pairs_trading import PairsTradingStrategy

    parser = argparse.ArgumentParser(description='Run the live trading bot.')
    parser.add_argument('--dry-run', action='store_true',
                       help='Simulate trades without executing them.')
    parser.add_argument('--interval', type=int, default=60,
                       help='Interval in seconds between trading cycles.')
    parser.add_argument('--backtest', action='store_true',
                       help='Run a backtest instead of live trading.')

    args = parser.parse_args()

    if args.backtest:
        # --- Backtesting Mode ---
        logger.info("Starting backtest...")
        
        # 1. Initialize the strategy
        #    Note: We are not passing the api object, so it will use historical data
        strategy = StackedEnsemble()

        # 2. Load historical data
        #    Using PairsTradingStrategy's data loading method for now
        data_loader = PairsTradingStrategy()
        historical_data = data_loader._get_price_data()
        
        if historical_data.empty:
            logger.error("No historical data found for backtesting. Exiting.")
            sys.exit(1)
        
        # Reformat data for backtester
        historical_data = historical_data.set_index('price_date')

        # 3. Initialize and run the backtester
        backtester = Backtester(strategy, initial_capital=100000.0)
        results = backtester.run(historical_data)

        # 4. Print results
        logger.info("\n--- Backtest Results ---")
        logger.info(f"Initial Portfolio Value: ${backtester.initial_capital:,.2f}")
        logger.info(f"Final Portfolio Value:   ${results['portfolio_value'].iloc[-1]:,.2f}")
        
        total_return = (results['portfolio_value'].iloc[-1] / backtester.initial_capital) - 1
        logger.info(f"Total Return:            {total_return:.2%}")
        
        logger.info("\n--- Portfolio Value Over Time ---")
        logger.info(results[['portfolio_value', 'cumulative_returns']])

    else:
        # --- Live Trading Mode ---
        trader = Trader(dry_run=args.dry_run)
        trader.run(interval=args.interval)

if __name__ == "__main__":
    main()

