"""
Lumibot Strategy Wrapper for RiskLabAI

This file connects RiskLabAI's signal generation to Lumibot's
trade execution and broker integration.

Keeps all existing infrastructure:
- Alpaca broker connection
- Stop-loss/take-profit management
- Connection manager for sockets
- Risk management
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from lumibot.strategies import Strategy
from typing import Dict, Tuple, Optional

# Import our RiskLabAI strategy
from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy

# Import existing infrastructure
from config.settings import (
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    MAX_POSITION_PCT,
    TRADING_SYMBOLS
)

logger = logging.getLogger(__name__)


class RiskLabAICombined(Strategy):
    """
    Lumibot strategy using RiskLabAI for signal generation.

    This class:
    1. Uses RiskLabAI for sophisticated signal generation
    2. Leverages existing Lumibot infrastructure for execution
    3. Preserves all risk management from original bot

    Attributes:
        risklabai: RiskLabAI strategy instance
        symbols: Trading symbols
        models_trained: Flag indicating if models are ready
        last_train_date: Last model training date
    """

    # Check for signals every hour
    SLEEPTIME = "1H"

    def initialize(self, parameters: Optional[Dict] = None):
        """
        Initialize strategy with RiskLabAI components.

        Args:
            parameters: Strategy parameters
        """
        logger.info("=" * 80)
        logger.info("INITIALIZING RISKLABAI COMBINED STRATEGY")
        logger.info("=" * 80)

        # Initialize RiskLabAI strategy
        self.risklabai = RiskLabAIStrategy(
            profit_taking=2.0,
            stop_loss=2.0,
            max_holding=10,
            n_cv_splits=5
        )

        # Trading symbols
        if parameters and 'symbols' in parameters:
            self.symbols = parameters['symbols']
        else:
            self.symbols = TRADING_SYMBOLS

        # Track state
        self.last_train_date = None
        self.models_trained = False
        self.min_training_bars = parameters.get('min_training_bars', 500) if parameters else 500
        self.retrain_days = parameters.get('retrain_days', 7) if parameters else 7

        # Model storage path
        self.model_path = parameters.get('model_path', 'models/risklabai_models.pkl') if parameters else 'models/risklabai_models.pkl'

        # Try to load existing models
        try:
            self.risklabai.load_models(self.model_path)
            self.models_trained = True
            self.last_train_date = datetime.now()
            logger.info(f"Loaded existing models from {self.model_path}")
        except Exception as e:
            logger.info(f"No existing models found: {e}")

        logger.info(f"Trading {len(self.symbols)} symbols: {self.symbols}")
        logger.info("=" * 80)

    def on_trading_iteration(self):
        """
        Main trading logic - called every iteration.
        """
        logger.info("-" * 60)
        logger.info(f"Trading iteration: {datetime.now()}")
        logger.info("-" * 60)

        # Step 1: Check if we need to retrain (weekly)
        if self._should_retrain():
            logger.info("Retraining models...")
            self._train_models()

        if not self.models_trained:
            logger.warning("Models not trained, skipping iteration")
            return

        # Step 2: Generate signals for each symbol
        for symbol in self.symbols:
            try:
                self._process_symbol(symbol)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue

    def _should_retrain(self) -> bool:
        """Check if models need retraining."""
        if self.last_train_date is None:
            return True

        days_since_train = (datetime.now() - self.last_train_date).days
        should_retrain = days_since_train >= self.retrain_days

        if should_retrain:
            logger.info(f"Time to retrain: {days_since_train} days since last training")

        return should_retrain

    def _train_models(self):
        """Train RiskLabAI models."""
        logger.info("=" * 60)
        logger.info("TRAINING RISKLABAI MODELS")
        logger.info("=" * 60)

        # Use first symbol for training (or combine multiple symbols)
        training_symbol = self.symbols[0]

        # Get historical data
        bars = self._get_historical_bars(training_symbol, self.min_training_bars)

        if bars is None or len(bars) < self.min_training_bars:
            logger.warning(f"Insufficient data for training: {len(bars) if bars is not None else 0} bars")
            return

        # Train
        results = self.risklabai.train(bars)

        if results['success']:
            self.models_trained = True
            self.last_train_date = datetime.now()

            # Save models
            try:
                self.risklabai.save_models(self.model_path)
                logger.info(f"Models saved to {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to save models: {e}")

            logger.info(f"Training successful!")
            logger.info(f"  Samples: {results['n_samples']}")
            logger.info(f"  Primary accuracy: {results['primary_accuracy']:.3f}")
            logger.info(f"  Meta accuracy: {results['meta_accuracy']:.3f}")
            logger.info(f"  Top features: {results['top_features'][:3]}")
        else:
            logger.warning(f"Training failed: {results.get('reason')}")

    def _get_historical_bars(
        self,
        symbol: str,
        length: int,
        timeframe: str = "day"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical bar data for a symbol.

        Args:
            symbol: Stock symbol
            length: Number of bars to fetch
            timeframe: Bar timeframe ("minute", "day", etc.)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            bars = self.get_historical_prices(
                symbol,
                length,
                timeframe
            )

            if bars is None:
                return None

            # Convert to DataFrame with expected format
            df = bars.df

            # Ensure we have required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.warning(f"Missing required columns for {symbol}")
                return None

            return df

        except Exception as e:
            logger.error(f"Error fetching bars for {symbol}: {e}")
            return None

    def _process_symbol(self, symbol: str):
        """Generate and execute signal for one symbol."""
        # Get recent bars (need enough for feature calculation)
        bars = self._get_historical_bars(symbol, 100)

        if bars is None or len(bars) < 50:
            logger.debug(f"{symbol}: Insufficient data ({len(bars) if bars is not None else 0} bars)")
            return

        # Get signal from RiskLabAI
        signal, bet_size = self.risklabai.predict(bars)

        if signal == 0 or bet_size < 0.5:
            logger.debug(f"{symbol}: No signal (signal={signal}, bet_size={bet_size:.2f})")
            return

        logger.info(f"{symbol}: Signal={signal}, Bet size={bet_size:.2f}")

        # Calculate position size
        portfolio_value = self.get_portfolio_value()
        position_value = portfolio_value * MAX_POSITION_PCT * bet_size

        # Get current position
        current_position = self.get_position(symbol)

        # Execute trade based on signal
        if signal == 1:
            # Long signal
            if current_position is None or current_position.quantity <= 0:
                self._enter_long(symbol, position_value)
            else:
                logger.debug(f"{symbol}: Already long, no action")

        elif signal == -1:
            # Short signal
            if current_position is None or current_position.quantity >= 0:
                # Close long position first if exists
                if current_position is not None and current_position.quantity > 0:
                    self._close_position(symbol)
                # Note: Short selling requires margin account
                # For now, we'll just avoid going short
                logger.info(f"{symbol}: Short signal but not implementing short selling")
            else:
                logger.debug(f"{symbol}: Already short, no action")

    def _enter_long(self, symbol: str, position_value: float):
        """Enter long position with risk management."""
        try:
            price = self.get_last_price(symbol)

            if price is None or price <= 0:
                logger.warning(f"{symbol}: Invalid price {price}")
                return

            quantity = int(position_value / price)

            if quantity <= 0:
                logger.debug(f"{symbol}: Position value too small")
                return

            logger.info(f"LONG {symbol}: {quantity} shares @ ${price:.2f} (total: ${position_value:.2f})")

            # Create bracket order with stop-loss and take-profit
            order = self.create_order(
                symbol,
                quantity,
                "buy",
                take_profit_price=price * (1 + TAKE_PROFIT_PCT),
                stop_loss_price=price * (1 - STOP_LOSS_PCT)
            )
            self.submit_order(order)

        except Exception as e:
            logger.error(f"Error entering long for {symbol}: {e}")

    def _close_position(self, symbol: str):
        """Close existing position."""
        try:
            position = self.get_position(symbol)

            if position is None:
                return

            logger.info(f"CLOSING {symbol}: {abs(position.quantity)} shares")

            order = self.create_order(
                symbol,
                abs(position.quantity),
                "sell" if position.quantity > 0 else "buy"
            )
            self.submit_order(order)

        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")

    def on_abrupt_closing(self):
        """Handle strategy shutdown."""
        logger.info("Strategy shutting down...")

        # Save models if trained
        if self.models_trained:
            try:
                self.risklabai.save_models(self.model_path)
                logger.info(f"Models saved to {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to save models on shutdown: {e}")

    def trace_stats(self, context, snapshot_before):
        """
        Called after on_trading_iteration to log statistics.

        Args:
            context: Trading context
            snapshot_before: Portfolio snapshot before iteration
        """
        # Get current portfolio value
        portfolio_value = self.get_portfolio_value()

        # Get positions
        positions = self.get_positions()

        # Log stats
        row = {
            'datetime': datetime.now(),
            'portfolio_value': portfolio_value,
            'num_positions': len(positions) if positions else 0,
            'models_trained': self.models_trained,
            'last_train_date': self.last_train_date
        }

        logger.info(f"Portfolio: ${portfolio_value:,.2f}, Positions: {row['num_positions']}")

        return row
