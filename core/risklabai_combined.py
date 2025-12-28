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

# Import GARCH volatility filter
from utils.garch_filter import GARCHVolatilityFilter

# Import existing infrastructure
from backup.settings import (
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    MAX_POSITION_PCT,
    TRADING_SYMBOLS
)

logger = logging.getLogger(__name__)


class KellyCriterion:
    """
    Calculate optimal position sizing using Kelly Criterion.

    The Kelly Criterion maximizes long-term growth rate while managing risk.
    Formula: f* = (p * b - q) / b
    where:
        p = win probability
        q = loss probability (1 - p)
        b = win/loss ratio
    """

    @staticmethod
    def calculate_kelly(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        fraction: float = 0.5  # Half-Kelly for safety
    ) -> float:
        """
        Calculate Kelly position size.

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average win percentage (e.g., 0.5 for 0.5%)
            avg_loss: Average loss percentage (e.g., 0.5 for 0.5%)
            fraction: Kelly fraction (0.5 = half-Kelly, safer than full Kelly)

        Returns:
            Position size as fraction of capital (0-1)
        """
        if avg_win <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.05  # Default 5% if parameters invalid

        loss_rate = 1 - win_rate
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

        # Kelly formula
        kelly = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio

        # Apply fraction and cap at reasonable maximum
        position_size = max(0.01, min(kelly * fraction, 0.15))  # Between 1% and 15%

        return position_size


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

        # Check for parameters set as instance attribute (Lumibot pattern)
        if parameters is None and hasattr(self, 'parameters'):
            parameters = self.parameters

        # Initialize RiskLabAI strategy with OPTIMIZED parameters
        profit_taking = parameters.get('profit_taking', 0.5) if parameters else 0.5
        stop_loss = parameters.get('stop_loss', 0.5) if parameters else 0.5
        max_holding = parameters.get('max_holding', 20) if parameters else 20

        self.risklabai = RiskLabAIStrategy(
            profit_taking=profit_taking,
            stop_loss=stop_loss,
            max_holding=max_holding,
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

        # NEW: Tick bars support
        self.use_tick_bars = parameters.get('use_tick_bars', False) if parameters else False

        # Model storage path
        self.model_path = parameters.get('model_path', 'models/risklabai_models.pkl') if parameters else 'models/risklabai_models.pkl'

        # NEW: Profitability tracking
        self.enable_profitability_tracking = parameters.get('enable_profitability_tracking', False) if parameters else False
        self.profitability_tracker = None
        self.last_portfolio_snapshot = None

        if self.enable_profitability_tracking:
            from utils.profitability_tracker import ProfitabilityTracker
            self.profitability_tracker = ProfitabilityTracker()
            logger.info("✓ Profitability tracking enabled")

        # NEW: Risk Management Parameters
        self.daily_loss_limit = parameters.get('daily_loss_limit_pct', 0.03) if parameters else 0.03  # 3% max daily loss
        self.max_drawdown_limit = parameters.get('max_drawdown_pct', 0.10) if parameters else 0.10  # 10% max drawdown
        self.drawdown_warning_level = parameters.get('drawdown_warning_pct', 0.05) if parameters else 0.05  # 5% warning
        self.max_consecutive_losses = parameters.get('max_consecutive_losses', 3) if parameters else 3
        self.max_trades_per_day = parameters.get('max_trades_per_day', 15) if parameters else 15

        # NEW: Kelly Criterion parameters
        self.use_kelly_sizing = parameters.get('use_kelly_sizing', True) if parameters else True
        self.kelly_fraction = parameters.get('kelly_fraction', 0.5) if parameters else 0.5  # Half-Kelly
        self.estimated_win_rate = parameters.get('estimated_win_rate', 0.5656) if parameters else 0.5656  # From 365-day model

        # Risk tracking state
        self.daily_start_value = None
        self.daily_start_date = None
        self.peak_portfolio_value = None
        self.consecutive_losses = 0
        self.trades_today = 0
        self.last_trade_result = None
        self.risk_halt_reason = None

        # NEW: GARCH Volatility Filter
        # Complements CUSUM (training filter) with GARCH (prediction filter)
        # - CUSUM: Prevents overfitting during training
        # - GARCH: Identifies high-volatility trading opportunities
        self.use_garch_filter = parameters.get('use_garch_filter', True) if parameters else True
        if self.use_garch_filter:
            garch_lookback = parameters.get('garch_lookback', 100) if parameters else 100
            garch_percentile = parameters.get('garch_percentile', 0.60) if parameters else 0.60
            self.garch_filter = GARCHVolatilityFilter(
                lookback_period=garch_lookback,
                volatility_percentile=garch_percentile,
                min_observations=50
            )
            logger.info("=" * 60)
            logger.info("GARCH VOLATILITY FILTER ENABLED")
            logger.info("=" * 60)
            logger.info(f"  Lookback: {garch_lookback} bars")
            logger.info(f"  Activation threshold: {garch_percentile:.0%}th percentile")
            logger.info(f"  Purpose: Activate RiskLabAI only in high-volatility regimes")
            logger.info("=" * 60)
        else:
            self.garch_filter = None

        logger.info("=" * 60)
        logger.info("RISK MANAGEMENT ENABLED")
        logger.info("=" * 60)
        logger.info(f"  Daily loss limit: {self.daily_loss_limit:.1%}")
        logger.info(f"  Max drawdown: {self.max_drawdown_limit:.1%}")
        logger.info(f"  Drawdown warning: {self.drawdown_warning_level:.1%}")
        logger.info(f"  Max consecutive losses: {self.max_consecutive_losses}")
        logger.info(f"  Max trades/day: {self.max_trades_per_day}")
        logger.info(f"  Kelly sizing: {'Enabled' if self.use_kelly_sizing else 'Disabled'}")
        logger.info(f"  Kelly fraction: {self.kelly_fraction:.1%} (Half-Kelly)")
        logger.info("=" * 60)

        # Try to load existing models
        try:
            self.risklabai.load_models(self.model_path)
            self.models_trained = True
            self.last_train_date = datetime.now()
            logger.info(f"✓ Loaded tick-based models from {self.model_path}")
        except Exception as e:
            logger.info(f"No existing models found: {e}")

        logger.info(f"Trading {len(self.symbols)} symbols: {self.symbols}")
        logger.info(f"Using tick bars: {self.use_tick_bars}")
        logger.info("=" * 80)

    def on_trading_iteration(self):
        """
        Main trading logic with comprehensive risk controls.
        """
        logger.info("-" * 60)
        logger.info(f"Trading iteration: {datetime.now()}")
        logger.info("-" * 60)

        # Step 0: Check if market is open
        try:
            if not self._is_market_open():
                logger.info("Market is currently closed. Skipping trading iteration.")
                return
        except Exception as e:
            logger.warning(f"Could not check market status: {e}. Continuing anyway...")

        # Step 0.5: Initialize daily tracking
        current_date = datetime.now().date()
        current_value = self.get_portfolio_value()

        if self.daily_start_date is None or current_date != self.daily_start_date:
            # New trading day
            self.daily_start_value = current_value
            self.daily_start_date = current_date
            self.trades_today = 0
            logger.info(f"📅 New trading day: Starting value = ${current_value:,.2f}")

        if self.peak_portfolio_value is None:
            self.peak_portfolio_value = current_value

        # Step 0.6: Check if trading is halted due to risk limits
        if self.risk_halt_reason:
            logger.warning(f"⛔ TRADING HALTED: {self.risk_halt_reason}")
            logger.warning("Manual intervention required to resume trading.")
            return

        # Step 0.7: Daily Loss Limit Check
        daily_pnl = current_value - self.daily_start_value
        daily_pnl_pct = daily_pnl / self.daily_start_value if self.daily_start_value > 0 else 0

        if daily_pnl_pct <= -self.daily_loss_limit:
            self.risk_halt_reason = f"Daily loss limit hit: {daily_pnl_pct:.2%} (limit: {self.daily_loss_limit:.2%})"
            logger.error("=" * 80)
            logger.error("🚨 DAILY LOSS LIMIT EXCEEDED")
            logger.error("=" * 80)
            logger.error(f"  Daily P&L: ${daily_pnl:,.2f} ({daily_pnl_pct:.2%})")
            logger.error(f"  Limit: {self.daily_loss_limit:.2%}")
            logger.error(f"  Trading halted for today. Reassess tomorrow.")
            logger.error("=" * 80)
            return

        # Step 0.8: Drawdown Check
        self.peak_portfolio_value = max(self.peak_portfolio_value, current_value)
        drawdown = (current_value - self.peak_portfolio_value) / self.peak_portfolio_value

        if drawdown <= -self.max_drawdown_limit:
            self.risk_halt_reason = f"Max drawdown exceeded: {drawdown:.2%} (limit: {self.max_drawdown_limit:.2%})"
            logger.error("=" * 80)
            logger.error("🚨 MAXIMUM DRAWDOWN EXCEEDED")
            logger.error("=" * 80)
            logger.error(f"  Peak: ${self.peak_portfolio_value:,.2f}")
            logger.error(f"  Current: ${current_value:,.2f}")
            logger.error(f"  Drawdown: {drawdown:.2%}")
            logger.error(f"  Limit: {self.max_drawdown_limit:.2%}")
            logger.error(f"  ALL TRADING HALTED - Manual intervention required")
            logger.error("=" * 80)
            return

        # Warning at 5% drawdown
        if drawdown <= -self.drawdown_warning_level:
            logger.warning("=" * 80)
            logger.warning(f"⚠️  DRAWDOWN WARNING: {drawdown:.2%}")
            logger.warning(f"  Approaching max drawdown limit ({self.max_drawdown_limit:.2%})")
            logger.warning(f"  Position sizes will be reduced")
            logger.warning("=" * 80)

        # Step 0.9: Consecutive Loss Check
        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.warning("=" * 80)
            logger.warning(f"⛔ {self.consecutive_losses} CONSECUTIVE LOSSES")
            logger.warning(f"  Trading paused for reassessment")
            logger.warning(f"  Consider:")
            logger.warning(f"    - Market regime change?")
            logger.warning(f"    - Model needs retraining?")
            logger.warning(f"    - Parameters need adjustment?")
            logger.warning("=" * 80)
            return

        # Step 0.10: Max Trades Per Day Check
        if self.trades_today >= self.max_trades_per_day:
            logger.info(f"ℹ️  Max trades per day reached ({self.max_trades_per_day})")
            logger.info(f"  No more trades today to prevent overtrading")
            return

        # Risk checks passed - log status
        logger.info(f"💰 Portfolio: ${current_value:,.2f} (Daily: {daily_pnl_pct:+.2%}, DD: {drawdown:.2%})")
        logger.info(f"📊 Risk Status: {self.consecutive_losses} losses, {self.trades_today}/{self.max_trades_per_day} trades")

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

    def _is_market_open(self) -> bool:
        """
        Check if the market is currently open.

        Returns:
            bool: True if market is open, False otherwise
        """
        try:
            # Use Lumibot's built-in method if available
            if hasattr(self, 'get_datetime'):
                current_time = self.get_datetime()
            else:
                current_time = datetime.now()

            # Check if it's a weekday (Monday=0, Sunday=6)
            if current_time.weekday() >= 5:  # Saturday or Sunday
                return False

            # Check market hours (9:30 AM - 4:00 PM ET)
            market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)

            return market_open <= current_time <= market_close

        except Exception as e:
            logger.warning(f"Error checking market hours: {e}")
            # If we can't determine market status, assume it's open to avoid blocking trades
            return True

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
        """Generate and execute signal for one symbol with Kelly Criterion sizing."""
        # Get recent bars (need enough for feature calculation)
        bars = self._get_historical_bars(symbol, 100)

        if bars is None or len(bars) < 50:
            logger.debug(f"{symbol}: Insufficient data ({len(bars) if bars is not None else 0} bars)")
            return

        # NEW: GARCH Volatility Filter - Check BEFORE calling RiskLabAI
        # Only activate RiskLabAI during high-volatility regimes
        if self.use_garch_filter and self.garch_filter is not None:
            should_trade, garch_info = self.garch_filter.should_trade(bars['close'])

            logger.debug(f"{symbol}: GARCH check - Vol={garch_info['forecasted_vol']:.4f}, "
                        f"Threshold={garch_info['threshold']:.4f}, Trade={should_trade}")

            if not should_trade:
                logger.debug(f"{symbol}: GARCH filter blocked - volatility too low "
                           f"({garch_info['forecasted_vol']:.4f} < {garch_info['threshold']:.4f})")
                return
            else:
                logger.info(f"{symbol}: ✓ GARCH filter passed - high volatility regime detected "
                          f"(vol={garch_info['forecasted_vol']:.4f}, threshold={garch_info['threshold']:.4f})")

        # Get signal from RiskLabAI (only if GARCH says trade)
        signal, bet_size = self.risklabai.predict(bars)

        if signal == 0 or bet_size < 0.5:
            logger.debug(f"{symbol}: No signal (signal={signal}, bet_size={bet_size:.2f})")
            return

        logger.info(f"{symbol}: Signal={signal}, Meta confidence={bet_size:.2f}")

        # Calculate Kelly position size
        portfolio_value = self.get_portfolio_value()

        if self.use_kelly_sizing:
            # Calculate Kelly fraction
            kelly_calc = KellyCriterion()
            kelly_fraction = kelly_calc.calculate_kelly(
                win_rate=self.estimated_win_rate,
                avg_win=self.risklabai.labeler.profit_taking_mult,  # 0.5%
                avg_loss=self.risklabai.labeler.stop_loss_mult,      # 0.5%
                fraction=self.kelly_fraction  # Half-Kelly
            )

            # Adjust for drawdown (reduce size in drawdown)
            drawdown = (portfolio_value - self.peak_portfolio_value) / self.peak_portfolio_value
            if drawdown < -self.drawdown_warning_level:
                # Reduce position size during drawdown
                drawdown_multiplier = 1.0 + (drawdown / self.drawdown_warning_level)
                kelly_fraction *= max(0.5, drawdown_multiplier)  # Reduce by up to 50%
                logger.info(f"  📉 Drawdown adjustment: position size reduced by {(1-max(0.5, drawdown_multiplier))*100:.0f}%")

            # Apply Kelly sizing with meta-model confidence
            position_value = portfolio_value * kelly_fraction * bet_size

            logger.info(f"  💰 Kelly sizing: {kelly_fraction:.2%} of ${portfolio_value:,.2f} × {bet_size:.2f} confidence = ${position_value:,.2f}")
        else:
            # Fallback to original sizing
            position_value = portfolio_value * MAX_POSITION_PCT * bet_size
            logger.info(f"  💰 Fixed sizing: {MAX_POSITION_PCT:.2%} × {bet_size:.2f} = ${position_value:,.2f}")

        # Get current position
        current_position = self.get_position(symbol)

        # Execute trade based on signal
        if signal == 1:
            # Long signal
            if current_position is None or current_position.quantity <= 0:
                self._enter_long(symbol, position_value)
                self.trades_today += 1
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

        # NEW: Record daily portfolio snapshot for profitability tracking
        if self.profitability_tracker:
            current_date = datetime.now().date()

            # Only record once per day
            if self.last_portfolio_snapshot != current_date:
                self.profitability_tracker.record_daily_snapshot(portfolio_value)
                self.last_portfolio_snapshot = current_date

                # Save summary periodically (every day)
                summary = self.profitability_tracker.save_summary()

                # Log profitability status
                metrics = self.profitability_tracker.calculate_metrics()
                logger.info("=" * 60)
                logger.info(f"PROFITABILITY UPDATE (Day {len(self.profitability_tracker.portfolio_values)})")
                logger.info(f"  Return: {metrics['cumulative_return']:+.2f}%")
                logger.info(f"  Win Rate: {metrics['win_rate']:.1f}%")
                logger.info(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
                logger.info(f"  Max DD: {metrics['max_drawdown']:.2f}%")
                logger.info(f"  Trades: {metrics['num_trades']}")

                ready = summary['profitability_criteria']['ready_for_live']
                logger.info(f"  Live Ready: {'✓ YES' if ready else '✗ NO'}")
                logger.info("=" * 60)

        return row
