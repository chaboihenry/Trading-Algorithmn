"""
RiskLabAI Trading Strategy - Lumibot Integration

Production-grade quantitative trading strategy implementing institutional-level
financial machine learning techniques from Marcos López de Prado's research.

Core RiskLabAI Components:
- Tick imbalance bars (market microstructure)
- CUSUM event filtering (significant moves only)
- Triple-barrier labeling (realistic trade outcomes)
- Fractional differentiation (stationarity with memory preservation)
- Purged K-fold cross-validation (no look-ahead bias)
- Meta-labeling for bet sizing (Kelly Criterion)

Broker Integration:
- Alpaca API for market data and execution
- Lumibot framework for strategy orchestration
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from lumibot.strategies import Strategy
from typing import Dict, Tuple, Optional

# Import our RiskLabAI strategy
from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy

# Import tick data infrastructure
from data.tick_storage import TickStorage
from data.tick_to_bars import generate_bars_from_ticks
from config.tick_config import (
    TICK_DB_PATH,
    INITIAL_IMBALANCE_THRESHOLD,
    OPTIMAL_PROFIT_TARGET,
    OPTIMAL_STOP_LOSS,
    MAX_POSITION_SIZE_PCT,
    SYMBOLS
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
    Production RiskLabAI quantitative trading strategy.

    Implements institutional-grade financial ML:
    1. RiskLabAI signal generation (López de Prado methodology)
    2. Tick imbalance bars for market microstructure
    3. Triple-barrier labeling with Kelly Criterion position sizing
    4. Industry-standard risk management (stop-loss, take-profit, drawdown limits)
    5. Per-symbol ML models with purged K-fold validation

    Attributes:
        symbol_models: Dictionary of per-symbol RiskLabAI strategy instances
        symbols: Trading symbols with trained models
        use_tick_bars: Whether to use tick imbalance bars (default: True)
        kelly_fraction: Kelly Criterion fraction for position sizing (default: 0.1)
    """

    # Check for sell signals (profit/loss thresholds) every minute
    # Buy signals are checked every hour (less time-sensitive)
    SLEEPTIME = "1M"

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

        # Store RiskLabAI parameters for creating per-symbol strategies
        self.risklabai_params = {
            'profit_taking': parameters.get('profit_taking', 0.5) if parameters else 0.5,
            'stop_loss': parameters.get('stop_loss', 0.5) if parameters else 0.5,
            'max_holding': parameters.get('max_holding', 20) if parameters else 20,
            'd': parameters.get('d', 1.0) if parameters else 1.0,
        }

        # Create a template RiskLabAI instance (for accessing labeler params)
        self.risklabai = RiskLabAIStrategy(
            profit_taking=self.risklabai_params['profit_taking'],
            stop_loss=self.risklabai_params['stop_loss'],
            max_holding=self.risklabai_params['max_holding'],
            d=self.risklabai_params['d'],
            n_cv_splits=5
        )

        # NEW: Per-symbol model storage
        self.symbol_models = {}  # Dict[symbol -> RiskLabAIStrategy instance]

        # Trading symbols
        if parameters and 'symbols' in parameters:
            self.symbols = parameters['symbols']
        else:
            self.symbols = SYMBOLS

        # Track state
        self.last_train_date = None
        self.models_trained = False
        self.min_training_bars = parameters.get('min_training_bars', 500) if parameters else 500
        self.retrain_days = parameters.get('retrain_days', 7) if parameters else 7

        # NEW: Tick bars support
        self.use_tick_bars = parameters.get('use_tick_bars', False) if parameters else False
        self.imbalance_threshold = parameters.get('imbalance_threshold', INITIAL_IMBALANCE_THRESHOLD) if parameters else INITIAL_IMBALANCE_THRESHOLD

        # Signal generation thresholds
        self.meta_threshold = parameters.get('meta_threshold', 0.001) if parameters else 0.001
        self.prob_threshold = parameters.get('prob_threshold', 0.015) if parameters else 0.015

        # Model storage directory (each symbol gets its own model file)
        self.models_dir = 'models'

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
        
        # Buy signal check frequency tracking
        self.last_buy_signal_check = None  # Track when we last checked for buy signals

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

        # Load per-symbol models
        logger.info("=" * 60)
        logger.info("LOADING PER-SYMBOL MODELS")
        logger.info("=" * 60)

        models_loaded = 0
        models_missing = []

        for symbol in self.symbols:
            model_path = f"{self.models_dir}/risklabai_{symbol}_models.pkl"
            try:
                # Create a new RiskLabAI instance for this symbol
                symbol_strategy = RiskLabAIStrategy(
                    profit_taking=self.risklabai_params['profit_taking'],
                    stop_loss=self.risklabai_params['stop_loss'],
                    max_holding=self.risklabai_params['max_holding'],
                    d=self.risklabai_params['d'],
                    n_cv_splits=5
                )

                # Load the trained model
                symbol_strategy.load_models(model_path)
                self.symbol_models[symbol] = symbol_strategy
                models_loaded += 1
                logger.info(f"  ✓ {symbol}: Loaded from {model_path}")

            except Exception as e:
                models_missing.append(symbol)
                logger.warning(f"  ✗ {symbol}: No model found ({model_path})")

        logger.info("=" * 60)
        logger.info(f"Models loaded: {models_loaded}/{len(self.symbols)}")

        if models_missing:
            logger.warning(f"Missing models for: {', '.join(models_missing)}")
            logger.warning(f"Run: python scripts/setup/train_all_symbols.py --phase phase_X")
            logger.warning(f"These symbols will be skipped during trading.")

        self.models_trained = models_loaded > 0
        if self.models_trained:
            self.last_train_date = datetime.now()

        logger.info("=" * 60)
        logger.info(f"Trading {len(self.symbol_models)} symbols with trained models")
        logger.info(f"Using tick bars: {self.use_tick_bars}")
        logger.info("=" * 80)

        # Lumibot framework compatibility
        # Some versions expect _backtesting_start attribute
        if not hasattr(self, '_backtesting_start'):
            self._backtesting_start = None

        # CRITICAL: Check existing positions immediately on startup
        # This ensures we don't wait for first iteration to check P/L thresholds
        logger.info("=" * 80)
        logger.info("🚨 STARTUP POSITION CHECK")
        logger.info("=" * 80)
        try:
            self._check_existing_positions()
            logger.info("✓ Startup position check complete")
        except Exception as e:
            logger.warning(f"Could not check positions on startup: {e}")
            logger.warning("Will check on first iteration")

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

        # Step 1: CHECK EXISTING POSITIONS FIRST (before generating new signals)
        # This ensures we capture profits and cut losses immediately
        # This runs every minute for time-sensitive sell signals
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: CHECKING EXISTING POSITIONS FOR EXIT TRIGGERS")
        logger.info("=" * 80)
        self._check_existing_positions()

        # Step 2: Check if we need to retrain (weekly)
        if self._should_retrain():
            logger.info("Retraining models...")
            self._train_models()

        if not self.models_trained:
            logger.warning("Models not trained, skipping iteration")
            return

        # Step 3: Generate buy signals for each symbol (only every hour)
        # Buy signals are less time-sensitive, so we check less frequently
        current_time = datetime.now()
        should_check_buy_signals = False
        
        if self.last_buy_signal_check is None:
            # First time - check immediately
            should_check_buy_signals = True
            self.last_buy_signal_check = current_time
        else:
            # Check if an hour has passed since last buy signal check
            time_since_last_check = current_time - self.last_buy_signal_check
            if time_since_last_check >= timedelta(hours=1):
                should_check_buy_signals = True
                self.last_buy_signal_check = current_time
        
        if should_check_buy_signals:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 2: GENERATING SIGNALS FOR NEW ENTRIES (Hourly Check)")
            logger.info("=" * 80)
            for symbol in self.symbol_models.keys():
                try:
                    self._process_symbol(symbol)
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    continue
        else:
            time_until_next_check = timedelta(hours=1) - (current_time - self.last_buy_signal_check)
            logger.info(f"⏰ Buy signal check skipped (next check in {time_until_next_check})")

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

            # Save per-symbol models
            try:
                for symbol, strategy in self.symbol_models.items():
                    model_path = f"{self.models_dir}/risklabai_{symbol}_models.pkl"
                    strategy.save_models(model_path)
                logger.info(f"Models saved for {len(self.symbol_models)} symbols")
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

        If use_tick_bars is True, fetches tick imbalance bars from database.
        Otherwise, fetches regular OHLC bars from broker.

        Args:
            symbol: Stock symbol
            length: Number of bars to fetch
            timeframe: Bar timeframe ("minute", "day", etc.) - ignored if use_tick_bars=True

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # NEW: Fetch tick imbalance bars from database
            if self.use_tick_bars:
                logger.debug(f"{symbol}: Fetching tick imbalance bars from database")

                # Load ticks from database
                storage = TickStorage(TICK_DB_PATH)
                ticks = storage.load_ticks(symbol)
                storage.close()

                if not ticks:
                    logger.warning(f"{symbol}: No ticks found in database")
                    return None

                # Generate imbalance bars from ticks
                bars = generate_bars_from_ticks(ticks, threshold=self.imbalance_threshold)

                if not bars:
                    logger.warning(f"{symbol}: No bars generated from ticks")
                    return None

                # Convert to DataFrame with proper datetime index
                df = pd.DataFrame(bars)

                # Set timestamp as index (required for fractional differencing)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')
                elif 'bar_end' in df.columns:
                    df['bar_end'] = pd.to_datetime(df['bar_end'])
                    df = df.set_index('bar_end')

                # Take most recent bars
                if len(df) > length:
                    df = df.tail(length)

                logger.debug(f"{symbol}: Loaded {len(df)} tick imbalance bars with datetime index")

                return df

            # ORIGINAL: Fetch regular OHLC bars from broker
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

    def _check_existing_positions(self):
        """
        Check all existing positions and execute exits based on profit/loss thresholds.

        Uses Alpaca's API directly to get unrealized P&L data.
        """
        try:
            # Get thresholds
            profit_target_pct = self.risklabai.labeler.profit_taking_mult  # 0.04 (4%)
            stop_loss_pct = self.risklabai.labeler.stop_loss_mult          # 0.02 (2%)

            # Access Alpaca API directly through broker
            if not hasattr(self, 'broker') or not hasattr(self.broker, 'api'):
                logger.warning("Cannot access broker API, skipping position check")
                return

            # Get positions directly from Alpaca API (has unrealized_plpc)
            try:
                alpaca_positions = self.broker.api.get_all_positions()
            except Exception as e:
                logger.error(f"Error getting positions from Alpaca API: {e}")
                return

            if not alpaca_positions:
                logger.debug("No open positions to check")
                return

            logger.info("=" * 80)
            logger.info(f"📊 CHECKING {len(alpaca_positions)} OPEN POSITIONS")
            logger.info("=" * 80)

            for position in alpaca_positions:
                try:
                    symbol = position.symbol
                    quantity = float(position.qty)

                    # Get P&L directly from Alpaca
                    pnl_pct = float(position.unrealized_plpc)  # Percentage as decimal (e.g., 0.066 = 6.6%)
                    pnl_dollars = float(position.unrealized_pl)
                    entry_price = float(position.avg_entry_price)
                    current_price = float(position.current_price)

                    logger.info("-" * 80)
                    logger.info(f"{symbol}: {quantity} shares")
                    logger.info(f"  Entry: ${entry_price:.2f}")
                    logger.info(f"  Current: ${current_price:.2f}")
                    logger.info(f"  P&L: ${pnl_dollars:+,.2f} ({pnl_pct:+.2%}) [from Alpaca]")

                    # CHECK EXIT CONDITIONS
                    exit_reason = None

                    # 1. PROFIT TARGET HIT
                    if pnl_pct >= profit_target_pct:
                        exit_reason = f"PROFIT TARGET HIT: {pnl_pct:+.2%} >= {profit_target_pct:.2%}"

                    # 2. STOP LOSS HIT
                    elif pnl_pct <= -stop_loss_pct:
                        exit_reason = f"STOP LOSS HIT: {pnl_pct:+.2%} <= {-stop_loss_pct:.2%}"

                    # EXECUTE EXIT IF THRESHOLD HIT
                    if exit_reason:
                        logger.info("=" * 80)
                        logger.info(f"🔴 EXITING POSITION: {symbol}")
                        logger.info("=" * 80)
                        logger.info(f"  Reason: {exit_reason}")
                        logger.info(f"  Realized P&L: ${pnl_dollars:+,.2f} ({pnl_pct:+.2%})")
                        logger.info("=" * 80)

                        # Issue independent SELL order
                        self._exit_position(symbol, exit_reason)
                    else:
                        logger.info(f"  ✓ Holding (Target: {profit_target_pct:.1%}, Stop: {-stop_loss_pct:.1%})")

                except Exception as e:
                    logger.error(f"{symbol}: Error processing position: {e}")
                    continue

            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Error checking existing positions: {e}")

    def _exit_position(self, symbol: str, reason: str):
        """
        Exit a position by issuing an independent SELL order.

        Args:
            symbol: Stock symbol to exit
            reason: Human-readable reason for exit (for logging)
        """
        try:
            position = self.get_position(symbol)

            if position is None or position.quantity <= 0:
                logger.warning(f"{symbol}: No position to exit")
                return

            quantity = abs(position.quantity)

            # Issue simple SELL order (bot manages independently)
            order = self.create_order(symbol, quantity, "sell")
            self.submit_order(order)

            logger.info(f"✅ SELL ORDER SUBMITTED: {quantity} shares of {symbol}")
            logger.info(f"   Exit reason: {reason}")

        except Exception as e:
            logger.error(f"Error exiting position for {symbol}: {e}")

    def _process_symbol(self, symbol: str):
        """Generate and execute signal for one symbol with Kelly Criterion sizing."""
        # Check if we have a trained model for this symbol
        if symbol not in self.symbol_models:
            logger.debug(f"{symbol}: No trained model available, skipping")
            return

        # Get recent bars (need enough for feature calculation)
        bars = self._get_historical_bars(symbol, 100)

        if bars is None or len(bars) < 50:
            logger.info(f"{symbol}: Insufficient data ({len(bars) if bars is not None else 0} bars)")
            return

        # Get signal from RiskLabAI model (includes CUSUM filtering for event detection)
        symbol_strategy = self.symbol_models[symbol]
        signal, bet_size = symbol_strategy.predict(
            bars,
            prob_threshold=self.prob_threshold,  # 0.015 (1.5%)
            meta_threshold=self.meta_threshold   # 0.001 (0.1%)
        )

        if signal == 0 or bet_size < 0.5:
            logger.info(f"{symbol}: No signal (signal={signal}, bet_size={bet_size:.2f})")
            return

        logger.info(f"{symbol}: ✅ SIGNAL={signal}, Meta confidence={bet_size:.2f} "
                   f"(thresholds: prob={self.prob_threshold}, meta={self.meta_threshold})")

        # Calculate Kelly position size
        portfolio_value = self.get_portfolio_value()

        if self.use_kelly_sizing:
            # Calculate Kelly fraction
            kelly_calc = KellyCriterion()
            kelly_fraction = kelly_calc.calculate_kelly(
                win_rate=self.estimated_win_rate,
                avg_win=self.risklabai.labeler.profit_taking_mult,  # 4% (0.04)
                avg_loss=self.risklabai.labeler.stop_loss_mult,      # 2% (0.02)
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
            position_value = portfolio_value * MAX_POSITION_SIZE_PCT * bet_size
            logger.info(f"  💰 Fixed sizing: {MAX_POSITION_SIZE_PCT:.2%} × {bet_size:.2f} = ${position_value:,.2f}")

        # Get current position (with safeguard logging)
        current_position = self.get_position(symbol)

        # Log what get_position actually returns for debugging
        if current_position is not None:
            logger.info(f"{symbol}: Current position detected - Symbol={getattr(current_position, 'symbol', 'N/A')}, "
                       f"Quantity={getattr(current_position, 'quantity', 'N/A')}")
        else:
            logger.info(f"{symbol}: No current position")

        # Execute trade based on signal
        if signal == 1:
            # Long signal
            if current_position is None or current_position.quantity <= 0:
                self._enter_long(symbol, position_value)
                self.trades_today += 1
            else:
                logger.info(f"{symbol}: Already long, no action")

        elif signal == -1:
            # Short signal
            if current_position is None or current_position.quantity >= 0:
                # Close long position first if exists
                if current_position is not None and current_position.quantity > 0:
                    self._exit_position(symbol, "SHORT SIGNAL - Closing long first")
                # Note: Short selling requires margin account
                # For now, we'll just avoid going short
                logger.info(f"{symbol}: Short signal but not implementing short selling")
            else:
                logger.info(f"{symbol}: Already short, no action")

    def _enter_long(self, symbol: str, position_value: float):
        """Enter long position with simple BUY order (bot manages exits independently)."""
        try:
            price = self.get_last_price(symbol)

            if price is None or price <= 0:
                logger.warning(f"{symbol}: Invalid price {price}")
                return

            quantity = int(position_value / price)

            if quantity <= 0:
                logger.info(f"{symbol}: Position value too small")
                return

            # Simple BUY order - no bracket orders (bot manages exits)
            order = self.create_order(symbol, quantity, "buy")
            self.submit_order(order)

            # Log entry with target thresholds
            profit_target_pct = self.risklabai.labeler.profit_taking_mult  # 0.04 (4%)
            stop_loss_pct = self.risklabai.labeler.stop_loss_mult          # 0.02 (2%)

            logger.info("=" * 80)
            logger.info(f"🔵 ENTERED LONG: {symbol}")
            logger.info("=" * 80)
            logger.info(f"  Quantity: {quantity} shares @ ${price:.2f}")
            logger.info(f"  Total: ${position_value:.2f}")
            logger.info(f"  📈 Profit Target: ${price * (1 + profit_target_pct):.2f} (+{profit_target_pct:.1%})")
            logger.info(f"  📉 Stop Loss: ${price * (1 - stop_loss_pct):.2f} (-{stop_loss_pct:.1%})")
            logger.info(f"  Entry Time: {datetime.now()}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Error entering long for {symbol}: {e}")

    def on_abrupt_closing(self):
        """Handle strategy shutdown."""
        logger.info("Strategy shutting down...")

        # Save models if trained
        if self.models_trained:
            try:
                for symbol, strategy in self.symbol_models.items():
                    model_path = f"{self.models_dir}/risklabai_{symbol}_models.pkl"
                    strategy.save_models(model_path)
                logger.info(f"Models saved for {len(self.symbol_models)} symbols on shutdown")
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

        # Get positions (filter out cash/USD - only count actual stock positions)
        positions = self.get_positions()

        # Filter to only stock positions with non-zero quantity
        stock_positions = []
        if positions:
            for position in positions:
                # Filter out cash and positions with zero quantity
                if hasattr(position, 'symbol') and position.symbol not in ['USD', 'CASH'] and hasattr(position, 'quantity') and position.quantity != 0:
                    stock_positions.append(position)

        # Log stats
        row = {
            'datetime': datetime.now(),
            'portfolio_value': portfolio_value,
            'num_positions': len(stock_positions),
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
