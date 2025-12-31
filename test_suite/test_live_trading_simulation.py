#!/usr/bin/env python3
"""
Live Trading Bot Simulation Test

This script simulates the live trading bot outside of market hours using historical data.
It tests the complete pipeline: data loading → bar generation → signal generation → order execution.

Usage:
    python test_suite/test_live_trading_simulation.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy
from config.tick_config import (
    TICK_DB_PATH,
    INITIAL_IMBALANCE_THRESHOLD,
    OPTIMAL_PROFIT_TARGET,
    OPTIMAL_STOP_LOSS,
    OPTIMAL_MAX_HOLDING_BARS,
    OPTIMAL_FRACTIONAL_D,
    OPTIMAL_META_THRESHOLD,
    OPTIMAL_PROB_THRESHOLD
)
from data.tick_storage import TickStorage
from data.tick_to_bars import generate_bars_from_ticks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockAlpacaBroker:
    """Mock Alpaca broker for testing without live market connection."""

    def __init__(self, starting_cash=100000):
        self.cash = starting_cash
        self.positions = {}  # symbol -> quantity
        self.orders = []
        self.order_id = 1000

    def get_cash(self):
        return self.cash

    def get_position(self, symbol):
        """Return mock position or None."""
        if symbol in self.positions and self.positions[symbol] != 0:
            position = Mock()
            position.symbol = symbol
            position.quantity = self.positions[symbol]
            position.avg_entry_price = 400.0  # Mock SPY price
            return position
        return None

    def get_positions(self):
        """Return list of mock positions."""
        positions = []
        for symbol, qty in self.positions.items():
            if qty != 0:
                position = Mock()
                position.symbol = symbol
                position.quantity = qty
                position.avg_entry_price = 400.0
                positions.append(position)
        return positions

    def submit_order(self, symbol, qty, side, order_type='market', **kwargs):
        """Mock order submission."""
        order = Mock()
        order.id = f"order_{self.order_id}"
        order.symbol = symbol
        order.qty = qty
        order.side = side
        order.status = 'filled'

        self.order_id += 1
        self.orders.append(order)

        # Update positions
        if side == 'buy':
            self.positions[symbol] = self.positions.get(symbol, 0) + qty
            self.cash -= qty * 400.0  # Mock price
        elif side == 'sell':
            self.positions[symbol] = self.positions.get(symbol, 0) - qty
            self.cash += qty * 400.0

        logger.info(f"Mock order: {side.upper()} {qty} {symbol} @ $400 (order_id={order.id})")
        return order

    def get_last_price(self, symbol):
        """Return mock current price."""
        return 400.0  # Mock SPY price

    def get_portfolio_value(self):
        """Calculate mock portfolio value."""
        position_value = sum(qty * 400.0 for qty in self.positions.values())
        return self.cash + position_value


class LiveTradingSimulator:
    """Simulates live trading bot behavior with historical data."""

    def __init__(self, model_path, symbol='SPY', simulation_bars=500):
        self.symbol = symbol
        self.simulation_bars = simulation_bars
        self.model_path = model_path

        # Initialize mock broker
        self.broker = MockAlpacaBroker(starting_cash=100000)

        # Initialize RiskLabAI strategy
        logger.info("=" * 80)
        logger.info("INITIALIZING RISKLABAI STRATEGY")
        logger.info("=" * 80)

        self.strategy = RiskLabAIStrategy(
            profit_taking=OPTIMAL_PROFIT_TARGET,
            stop_loss=OPTIMAL_STOP_LOSS,
            max_holding=OPTIMAL_MAX_HOLDING_BARS,
            d=OPTIMAL_FRACTIONAL_D,
            n_cv_splits=5,
            force_directional=True,
            neutral_threshold=0.00001
        )

        # Load pre-trained models
        try:
            self.strategy.load_models(model_path)
            logger.info(f"✓ Loaded models from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

        # Simulation state
        self.bars_history = []
        self.signals_generated = []
        self.trades_executed = []

    def load_historical_data(self):
        """Load historical tick data and generate bars."""
        logger.info("=" * 80)
        logger.info("LOADING HISTORICAL DATA")
        logger.info("=" * 80)

        storage = TickStorage(TICK_DB_PATH)

        # Get date range
        date_range = storage.get_date_range(self.symbol)
        if not date_range:
            raise ValueError(f"No tick data found for {self.symbol}")

        earliest, latest = date_range
        logger.info(f"Available data: {earliest} to {latest}")

        # Load last N days of ticks
        ticks = storage.load_ticks(self.symbol, limit=5000000)  # Load plenty of data
        storage.close()

        if not ticks:
            raise ValueError(f"Failed to load ticks for {self.symbol}")

        logger.info(f"✓ Loaded {len(ticks):,} ticks")

        # Generate tick imbalance bars
        logger.info(f"Generating tick imbalance bars (threshold={INITIAL_IMBALANCE_THRESHOLD})...")
        bars_list = generate_bars_from_ticks(ticks, threshold=INITIAL_IMBALANCE_THRESHOLD)

        if not bars_list:
            raise ValueError("Failed to generate bars from ticks")

        logger.info(f"✓ Generated {len(bars_list)} bars")

        # Convert to DataFrame
        bars_df = pd.DataFrame(bars_list)
        bars_df['bar_end'] = pd.to_datetime(bars_df['bar_end'])
        if bars_df['bar_end'].dt.tz is not None:
            bars_df['bar_end'] = bars_df['bar_end'].dt.tz_localize(None)
        bars_df.set_index('bar_end', inplace=True)

        # Use last N bars for simulation (to ensure we have enough data)
        simulation_bars = bars_df.iloc[-self.simulation_bars:].copy()

        logger.info(f"✓ Using {len(simulation_bars)} bars for simulation")
        logger.info(f"  Date range: {simulation_bars.index[0]} to {simulation_bars.index[-1]}")
        logger.info("")

        return simulation_bars

    def simulate_trading_day(self, bars_up_to_now):
        """Simulate one trading iteration (like on_trading_iteration)."""

        # Need at least 50 bars for feature calculation
        if len(bars_up_to_now) < 50:
            logger.info(f"Insufficient bars ({len(bars_up_to_now)}), skipping...")
            return None

        # Get signal from RiskLabAI
        try:
            signal, bet_size = self.strategy.predict(
                bars_up_to_now,
                prob_threshold=OPTIMAL_PROB_THRESHOLD,
                meta_threshold=OPTIMAL_META_THRESHOLD
            )

            logger.info(f"Prediction: signal={signal}, bet_size={bet_size:.2f}")
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None

        # Record signal
        signal_record = {
            'timestamp': bars_up_to_now.index[-1],
            'signal': signal,
            'bet_size': bet_size,
            'portfolio_value': self.broker.get_portfolio_value()
        }
        self.signals_generated.append(signal_record)

        # Execute trade if signal
        if signal == 0:
            logger.info(f"✗ Neutral signal (no trade)")
            return signal_record
        elif bet_size < 0.5:
            logger.info(f"✗ Signal filtered: bet_size={bet_size:.2f} < 0.5")
            return signal_record

        logger.info(f"✅ SIGNAL={signal}, Bet size={bet_size:.2f}")

        # Calculate position size (Kelly sizing)
        portfolio_value = self.broker.get_portfolio_value()
        kelly_fraction = 0.5  # Half-Kelly
        position_value = portfolio_value * kelly_fraction * bet_size

        # Get current position
        current_position = self.broker.get_position(self.symbol)

        # Execute based on signal
        if signal == 1:  # Long
            if current_position is None or current_position.quantity <= 0:
                # Enter long
                price = self.broker.get_last_price(self.symbol)
                quantity = int(position_value / price)

                if quantity > 0:
                    self.broker.submit_order(self.symbol, quantity, 'buy')

                    trade_record = {
                        'timestamp': bars_up_to_now.index[-1],
                        'action': 'BUY',
                        'quantity': quantity,
                        'price': price,
                        'value': quantity * price
                    }
                    self.trades_executed.append(trade_record)
                    logger.info(f"LONG {self.symbol}: {quantity} shares @ ${price:.2f}")
            else:
                logger.info(f"Already long, no action")

        elif signal == -1:  # Short (close long for now)
            if current_position is not None and current_position.quantity > 0:
                # Close long position
                quantity = current_position.quantity
                price = self.broker.get_last_price(self.symbol)

                self.broker.submit_order(self.symbol, quantity, 'sell')

                trade_record = {
                    'timestamp': bars_up_to_now.index[-1],
                    'action': 'SELL',
                    'quantity': quantity,
                    'price': price,
                    'value': quantity * price
                }
                self.trades_executed.append(trade_record)
                logger.info(f"CLOSE LONG {self.symbol}: {quantity} shares @ ${price:.2f}")
            else:
                logger.info(f"Short signal but no position to close")

        return signal_record

    def run_simulation(self):
        """Run the complete simulation."""
        logger.info("=" * 80)
        logger.info("STARTING LIVE TRADING SIMULATION")
        logger.info("=" * 80)
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Simulation bars: {self.simulation_bars}")
        logger.info(f"Starting portfolio value: ${self.broker.get_portfolio_value():,.2f}")
        logger.info("")

        # Load historical data
        all_bars = self.load_historical_data()

        # Simulate trading day by day (iterate through bars)
        logger.info("=" * 80)
        logger.info("RUNNING SIMULATION")
        logger.info("=" * 80)

        # Simulate every 10th bar (to reduce computation)
        simulation_interval = 10
        iterations = 0

        for i in range(50, len(all_bars), simulation_interval):
            bars_up_to_now = all_bars.iloc[:i].copy()
            current_time = bars_up_to_now.index[-1]

            logger.info("")
            logger.info(f"--- Iteration {iterations + 1} @ {current_time} ---")

            signal_record = self.simulate_trading_day(bars_up_to_now)
            iterations += 1

        logger.info("")
        logger.info(f"Completed {iterations} iterations")

        # Print results
        self.print_results()

    def print_results(self):
        """Print simulation results."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("SIMULATION RESULTS")
        logger.info("=" * 80)

        # Portfolio metrics
        final_value = self.broker.get_portfolio_value()
        starting_value = 100000
        total_return = (final_value - starting_value) / starting_value * 100

        logger.info(f"Starting portfolio value: ${starting_value:,.2f}")
        logger.info(f"Final portfolio value: ${final_value:,.2f}")
        logger.info(f"Total return: {total_return:.2f}%")
        logger.info("")

        # Signal distribution
        if self.signals_generated:
            signals_df = pd.DataFrame(self.signals_generated)
            signal_counts = signals_df['signal'].value_counts()

            total_signals = len(signals_df)
            short_pct = signal_counts.get(-1, 0) / total_signals * 100
            neutral_pct = signal_counts.get(0, 0) / total_signals * 100
            long_pct = signal_counts.get(1, 0) / total_signals * 100

            logger.info(f"Signal distribution ({total_signals} total):")
            logger.info(f"  Short (-1):  {signal_counts.get(-1, 0):4d} ({short_pct:5.1f}%)")
            logger.info(f"  Neutral (0): {signal_counts.get(0, 0):4d} ({neutral_pct:5.1f}%)")
            logger.info(f"  Long (1):    {signal_counts.get(1, 0):4d} ({long_pct:5.1f}%)")
            logger.info("")

        # Trade summary
        logger.info(f"Trades executed: {len(self.trades_executed)}")
        if self.trades_executed:
            trades_df = pd.DataFrame(self.trades_executed)
            logger.info(f"  Buy orders: {len(trades_df[trades_df['action'] == 'BUY'])}")
            logger.info(f"  Sell orders: {len(trades_df[trades_df['action'] == 'SELL'])}")
        logger.info("")

        # Final positions
        positions = self.broker.get_positions()
        logger.info(f"Final positions: {len(positions)}")
        for pos in positions:
            logger.info(f"  {pos.symbol}: {pos.quantity} shares")
        logger.info("")

        # Orders history
        logger.info(f"Total orders submitted: {len(self.broker.orders)}")
        logger.info("=" * 80)


def main():
    """Run live trading simulation."""
    try:
        # Model path
        model_path = "models/risklabai_tick_models_aggressive.pkl"

        # Check if model exists
        if not Path(model_path).exists():
            logger.error(f"Model not found at {model_path}")
            logger.error("Run scripts/retrain_aggressive.py first to create the model")
            return 1

        # Create and run simulator
        simulator = LiveTradingSimulator(
            model_path=model_path,
            symbol='SPY',
            simulation_bars=500  # Simulate last 500 bars
        )

        simulator.run_simulation()

        logger.info("✓ Simulation completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
