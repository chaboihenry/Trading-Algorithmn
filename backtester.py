#!/usr/bin/env python
"""
Backtesting engine for the trading bot.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

class Backtester:
    """
    A simple backtesting engine.
    """

    def __init__(self, strategy, initial_capital=100000.0):
        """
        Initialize the backtester.

        Args:
            strategy: The trading strategy to backtest.
            initial_capital: The initial capital for the backtest.
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.history = []

    def run(self, data: pd.DataFrame):
        """
        Run the backtest.

        Args:
            data: The historical data to backtest on.
        """
        logger.info("🚀 Starting backtest...")

        # Group data by date
        for date, daily_data in data.groupby(data.index.date):
            logger.info(f"Processing date: {date}")

            # Generate signals
            signals = self.strategy.generate_signals(daily_data)

            if not signals.empty:
                # Execute trades
                self.execute_trades(signals, daily_data)

            # Record portfolio value
            portfolio_value = self.calculate_portfolio_value(daily_data)
            self.history.append({'date': date, 'portfolio_value': portfolio_value})

        logger.info("✅ Backtest complete.")
        return self.get_results()

    def execute_trades(self, signals: pd.DataFrame, daily_data: pd.DataFrame):
        """
        Execute trades based on the signals.

        Args:
            signals: The trading signals.
            daily_data: The daily market data.
        """
        for _, signal in signals.iterrows():
            symbol = signal['symbol_ticker']
            signal_type = signal['signal_type']
            price = daily_data[daily_data['symbol_ticker'] == symbol]['close'].iloc[0]
            qty = 100  # Simple fixed quantity for now

            if signal_type == 'BUY':
                if self.capital >= qty * price:
                    self.capital -= qty * price
                    self.positions[symbol] = self.positions.get(symbol, 0) + qty
                    logger.info(f"BOUGHT {qty} shares of {symbol} at {price}")
                else:
                    logger.warning(f"Not enough capital to buy {qty} shares of {symbol}")
            elif signal_type == 'SELL':
                if self.positions.get(symbol, 0) >= qty:
                    self.capital += qty * price
                    self.positions[symbol] -= qty
                    logger.info(f"SOLD {qty} shares of {symbol} at {price}")
                else:
                    logger.warning(f"Not enough shares of {symbol} to sell")

    def calculate_portfolio_value(self, daily_data: pd.DataFrame) -> float:
        """
        Calculate the current portfolio value.

        Args:
            daily_data: The daily market data.

        Returns:
            The current portfolio value.
        """
        portfolio_value = self.capital
        for symbol, qty in self.positions.items():
            if qty > 0:
                try:
                    price = daily_data[daily_data['symbol_ticker'] == symbol]['close'].iloc[0]
                    portfolio_value += qty * price
                except IndexError:
                    # Stock not in daily data, use last known price from self.history
                    if len(self.history) > 0:
                        # This is a simplification. A real backtester would need to handle this better.
                        pass # For now, we just don't add the value if the price is not available
        return portfolio_value

    def get_results(self) -> pd.DataFrame:
        """
        Get the backtest results.

        Returns:
            A DataFrame with the backtest results.
        """
        results_df = pd.DataFrame(self.history)
        results_df['date'] = pd.to_datetime(results_df['date'])
        results_df = results_df.set_index('date')
        results_df['returns'] = results_df['portfolio_value'].pct_change()
        results_df['cumulative_returns'] = (1 + results_df['returns']).cumprod() - 1
        return results_df
