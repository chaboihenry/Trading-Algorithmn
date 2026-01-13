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
import json
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from lumibot.strategies import Strategy
from typing import Dict, Tuple, Optional

# Import our RiskLabAI strategy
from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy
from risklabai.sampling.cusum_filter import CUSUMEventFilter

# Import tick data infrastructure
from data.tick_storage import TickStorage
from data.tick_to_bars import generate_bars_from_ticks
from data.model_storage import ModelStorage
from config.tick_config import (
    TICK_DB_PATH,
    INITIAL_IMBALANCE_THRESHOLD,
    CUSUM_EVENT_WINDOW_SECONDS,
    OPTIMAL_PROFIT_TARGET,
    OPTIMAL_STOP_LOSS,
    MAX_POSITION_SIZE_PCT,
    SYMBOLS
)
from utils.market_calendar import (
    is_market_open,
    is_trading_day,
    now_et,
    time_until_market_open,
    MARKET_TZ
)

logger = logging.getLogger(__name__)

# NOTE: Timezone utilities are now imported from utils.market_calendar
# This ensures consistent use of Eastern Time across the entire codebase

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


class CircuitBreaker:
    """
    Circuit breaker pattern to halt trading on anomalous conditions.

    Protects against:
    - Excessive daily losses (default: 3%)
    - Large drawdowns (default: 10%)
    - Consecutive losing trades (default: 5)
    - Excessive trading frequency (default: 10 trades/hour)

    Once tripped, requires manual reset or will auto-reset next day.
    """

    def __init__(self,
                 max_daily_loss: float = 0.03,
                 max_drawdown: float = 0.10,
                 max_consecutive_losses: int = 5,
                 max_trades_per_hour: int = 10):
        """
        Initialize circuit breaker with safety thresholds.

        Args:
            max_daily_loss: Maximum daily loss (0.03 = 3%)
            max_drawdown: Maximum drawdown from peak (0.10 = 10%)
            max_consecutive_losses: Maximum consecutive losing trades
            max_trades_per_hour: Maximum trades allowed per hour
        """
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.max_consecutive_losses = max_consecutive_losses
        self.max_trades_per_hour = max_trades_per_hour

        self.is_tripped = False
        self.trip_reason = None
        self.trip_timestamp = None
        self.hourly_trades = []

    def check(self,
              portfolio_value: float,
              daily_start_value: float,
              peak_value: float,
              consecutive_losses: int,
              trade_history: list) -> Tuple[bool, str]:
        """
        Check if circuit breaker should trip.

        Args:
            portfolio_value: Current portfolio value
            daily_start_value: Portfolio value at market open
            peak_value: Highest portfolio value ever reached
            consecutive_losses: Number of consecutive losing trades
            trade_history: List of recent trades with timestamps

        Returns:
            Tuple of (should_trip: bool, reason: str)
        """
        # Daily loss check
        if daily_start_value and daily_start_value > 0:
            daily_pnl = (portfolio_value - daily_start_value) / daily_start_value
            if daily_pnl <= -self.max_daily_loss:
                return True, f"Daily loss limit exceeded: {daily_pnl:.2%} (limit: {-self.max_daily_loss:.2%})"

        # Drawdown check
        if peak_value and peak_value > 0:
            drawdown = (portfolio_value - peak_value) / peak_value
            if drawdown <= -self.max_drawdown:
                return True, f"Max drawdown exceeded: {drawdown:.2%} (limit: {-self.max_drawdown:.2%})"

        # Consecutive losses check
        if consecutive_losses >= self.max_consecutive_losses:
            return True, f"Consecutive losses limit: {consecutive_losses} (limit: {self.max_consecutive_losses})"

        # Trades per hour check
        hour_ago = datetime.now() - timedelta(hours=1)
        recent_trades = [t for t in trade_history if 'timestamp' in t and t['timestamp'] > hour_ago]
        if len(recent_trades) >= self.max_trades_per_hour:
            return True, f"Too many trades per hour: {len(recent_trades)} (limit: {self.max_trades_per_hour})"

        return False, ""

    def trip(self, reason: str):
        """Trip the circuit breaker with a reason."""
        self.is_tripped = True
        self.trip_reason = reason
        self.trip_timestamp = datetime.now()
        logger.error("=" * 80)
        logger.error(f"🚨 CIRCUIT BREAKER TRIPPED: {reason}")
        logger.error(f"   Timestamp: {self.trip_timestamp}")
        logger.error("   Trading halted until manual reset or next market day")
        logger.error("=" * 80)

    def reset(self):
        """Reset circuit breaker (manual intervention or daily reset)."""
        if self.is_tripped:
            logger.warning("=" * 60)
            logger.warning("⚠️  CIRCUIT BREAKER RESET")
            logger.warning(f"   Previous trip reason: {self.trip_reason}")
            logger.warning(f"   Tripped at: {self.trip_timestamp}")
            logger.warning("   Trading resumed")
            logger.warning("=" * 60)

        self.is_tripped = False
        self.trip_reason = None
        self.trip_timestamp = None

    def should_auto_reset(self) -> bool:
        """Check if circuit breaker should auto-reset (new trading day)."""
        if not self.is_tripped or not self.trip_timestamp:
            return False

        # Auto-reset if it's a new trading day
        now = datetime.now()
        if now.date() > self.trip_timestamp.date():
            return True

        return False


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
            'd': parameters.get('d', None) if parameters else None,
        }

        # Create a template RiskLabAI instance (for accessing labeler params)
        self.risklabai = RiskLabAIStrategy(
            profit_taking=self.risklabai_params['profit_taking'],
            stop_loss=self.risklabai_params['stop_loss'],
            max_holding=self.risklabai_params['max_holding'],
            d=self.risklabai_params['d'],
            n_cv_splits=5
        )
        self.cusum_filter = CUSUMEventFilter()

        # NEW: Per-symbol model storage
        self.symbol_models = {}  # Dict[symbol -> RiskLabAIStrategy instance]

        # NEW: Model storage with S3 (primary) + local cache support
        self.model_storage = ModelStorage(local_dir=self.models_dir)

        # HRP optimization output (computed after retraining)
        self.hrp_weights = None

        # Verify S3 is configured (required for production/portable use)
        if not self.model_storage.s3_client:
            logger.warning("=" * 60)
            logger.warning("⚠️  AWS S3 NOT CONFIGURED")
            logger.warning("   Models will only be stored/loaded locally")
            logger.warning("   This bot will NOT be portable to other machines")
            logger.warning("   ")
            logger.warning("   To enable S3 storage, set these in .env:")
            logger.warning("   - AWS_ACCESS_KEY_ID")
            logger.warning("   - AWS_SECRET_ACCESS_KEY")
            logger.warning("   - AWS_REGION")
            logger.warning("   - S3_MODEL_BUCKET")
            logger.warning("=" * 60)

        # Trading symbols
        if parameters and 'symbols' in parameters:
            self.symbols = parameters['symbols']
        else:
            self.symbols = SYMBOLS

        # Track state
        self.last_train_date = None
        self.models_trained = False
        self.min_training_bars = parameters.get('min_training_bars', 500) if parameters else 500
        self.retrain_days = parameters.get('retrain_days', 30) if parameters else 30  # Retrain every 30 days (was 7)

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

        # DYNAMIC WIN RATE: Track actual trade history instead of hardcoded estimate
        self.trade_history = []  # List of completed trades with P&L
        self.dynamic_win_rate = 0.50  # Start at 50%, will update based on actual results

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

        # Stop loss cooldown tracking - prevent re-buying after stop loss
        self.stop_loss_cooldowns = {}  # {symbol: datetime_of_stop_loss}
        self.stop_loss_cooldown_days = 7  # Don't re-buy for 7 days after stop loss

        # NEW: Circuit breaker for anomalous conditions
        max_trades_per_hour = parameters.get('max_trades_per_hour', 10) if parameters else 10
        self.circuit_breaker = CircuitBreaker(
            max_daily_loss=self.daily_loss_limit,
            max_drawdown=self.max_drawdown_limit,
            max_consecutive_losses=self.max_consecutive_losses,
            max_trades_per_hour=max_trades_per_hour
        )

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
        logger.info(f"  Stop loss cooldown: {self.stop_loss_cooldown_days} days (prevents re-buying after stop)")
        logger.info("")
        logger.info("  CIRCUIT BREAKER:")
        logger.info(f"    Max daily loss: {self.circuit_breaker.max_daily_loss:.1%}")
        logger.info(f"    Max drawdown: {self.circuit_breaker.max_drawdown:.1%}")
        logger.info(f"    Max consecutive losses: {self.circuit_breaker.max_consecutive_losses}")
        logger.info(f"    Max trades/hour: {self.circuit_breaker.max_trades_per_hour}")
        logger.info("=" * 60)

        # Load per-symbol models using ModelStorage
        logger.info("=" * 60)
        logger.info("LOADING PER-SYMBOL MODELS")
        logger.info("=" * 60)

        self._load_models_from_storage()

        models_loaded = len(self.symbol_models)
        models_missing = [s for s in self.symbols if s not in self.symbol_models]

        logger.info("=" * 60)
        logger.info(f"Models loaded: {models_loaded}/{len(self.symbols)}")

        if models_missing:
            logger.warning(f"Missing models for: {', '.join(models_missing)}")
            logger.warning(f"Will train on first retraining cycle")

        self.models_trained = models_loaded > 0

        logger.info("=" * 60)
        logger.info(f"Trading {len(self.symbol_models)} symbols with trained models")
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

        # NEW: Load saved state from previous run (cooldowns, trade history, etc.)
        self._load_state()

    def _save_state(self):
        """
        Save bot state to disk for crash recovery.

        State includes:
        - stop_loss_cooldowns: Symbols in cooldown period after stop loss
        - trade_history: Last 100 trades for win rate calculation
        - last_train_date: When models were last trained
        - daily_start_value: Portfolio value at start of trading day
        - peak_portfolio_value: Highest portfolio value (for drawdown tracking)

        This enables the bot to resume exactly where it left off after:
        - Ctrl+C shutdown
        - Crash/exception
        - System restart
        - Container restart (Docker/Kubernetes)
        """
        state_file = Path("bot_state.json")

        state = {
            'stop_loss_cooldowns': {
                symbol: dt.isoformat()
                for symbol, dt in self.stop_loss_cooldowns.items()
            },
            'trade_history': self.trade_history[-100:],  # Last 100 trades
            'last_train_date': self.last_train_date.isoformat() if self.last_train_date else None,
            'daily_start_value': self.daily_start_value,
            'peak_portfolio_value': self.peak_portfolio_value,
            # Circuit breaker state
            'circuit_breaker': {
                'is_tripped': self.circuit_breaker.is_tripped,
                'trip_reason': self.circuit_breaker.trip_reason,
                'trip_timestamp': self.circuit_breaker.trip_timestamp.isoformat() if self.circuit_breaker.trip_timestamp else None
            },
            'timestamp': datetime.now().isoformat()
        }

        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)

            logger.info(f"✓ State saved to {state_file}")
            logger.info(f"  Cooldowns: {len(self.stop_loss_cooldowns)}")
            logger.info(f"  Trade history: {len(self.trade_history)} trades")

        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _load_state(self):
        """
        Load state from previous run.

        Restores:
        - stop_loss_cooldowns: Prevents re-buying symbols that hit stop loss
        - trade_history: Enables accurate win rate calculation
        - last_train_date: Determines when next retraining is needed
        - daily_start_value: For daily loss limit tracking
        - peak_portfolio_value: For drawdown calculations

        If no state file exists (first run), starts with clean slate.
        If state file is corrupted, logs error and starts fresh.
        """
        state_file = Path("bot_state.json")

        if not state_file.exists():
            logger.info("No previous state file found - starting fresh")
            return

        try:
            with open(state_file, 'r') as f:
                state = json.load(f)

            # Restore cooldowns (critical for stop loss cooldown logic)
            cooldowns_data = state.get('stop_loss_cooldowns', {})
            self.stop_loss_cooldowns = {
                symbol: datetime.fromisoformat(dt_str)
                for symbol, dt_str in cooldowns_data.items()
            }

            # Restore trade history (for win rate calculation)
            self.trade_history = state.get('trade_history', [])

            # Restore training date (if present)
            if state.get('last_train_date'):
                self.last_train_date = datetime.fromisoformat(state['last_train_date'])

            # Restore daily/peak values (if present)
            self.daily_start_value = state.get('daily_start_value')
            self.peak_portfolio_value = state.get('peak_portfolio_value')

            # Restore circuit breaker state (if present)
            cb_state = state.get('circuit_breaker', {})
            if cb_state:
                self.circuit_breaker.is_tripped = cb_state.get('is_tripped', False)
                self.circuit_breaker.trip_reason = cb_state.get('trip_reason')
                if cb_state.get('trip_timestamp'):
                    self.circuit_breaker.trip_timestamp = datetime.fromisoformat(cb_state['trip_timestamp'])

            logger.info("=" * 60)
            logger.info("STATE RESTORED FROM PREVIOUS RUN")
            logger.info("=" * 60)
            logger.info(f"  Cooldowns: {len(self.stop_loss_cooldowns)} symbols")
            logger.info(f"  Trade history: {len(self.trade_history)} trades")
            logger.info(f"  Last train: {self.last_train_date or 'Never'}")
            logger.info(f"  Circuit breaker: {'TRIPPED' if self.circuit_breaker.is_tripped else 'OK'}")
            if self.circuit_breaker.is_tripped:
                logger.info(f"    Reason: {self.circuit_breaker.trip_reason}")
                logger.info(f"    Since: {self.circuit_breaker.trip_timestamp}")
            logger.info(f"  State saved: {state.get('timestamp', 'Unknown')}")
            logger.info("=" * 60)

            # Log active cooldowns (if any)
            if self.stop_loss_cooldowns:
                logger.info("Active stop loss cooldowns:")
                for symbol, cooldown_end in self.stop_loss_cooldowns.items():
                    days_remaining = (cooldown_end - datetime.now()).days
                    if days_remaining > 0:
                        logger.info(f"  {symbol}: {days_remaining} days remaining")
                    else:
                        logger.info(f"  {symbol}: expired (will be removed)")

        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            logger.warning("Starting with fresh state")

    def _generate_order_id(self, symbol: str, signal: int) -> str:
        """
        Generate unique order ID to prevent duplicate orders.

        This provides idempotency - if the bot crashes mid-order and restarts,
        it won't accidentally place the same order twice.

        Args:
            symbol: Stock ticker (e.g., "SPY")
            signal: Trade direction (+1 long, -1 short, 0 neutral)

        Returns:
            Unique order ID string: "SPY_1_20260111143052_a3f8b2c1"

        Example:
            >>> self._generate_order_id("SPY", 1)
            "SPY_1_20260111143052_a3f8b2c1"
        """
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        unique = uuid.uuid4().hex[:8]
        return f"{symbol}_{signal}_{timestamp}_{unique}"

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

        # Step 0.5.5: Auto-reset circuit breaker on new trading day
        if self.circuit_breaker.should_auto_reset():
            logger.info("🔄 New trading day - Auto-resetting circuit breaker")
            self.circuit_breaker.reset()

        # Step 0.5.6: Check circuit breaker FIRST (consolidates all safety checks)
        should_trip, trip_reason = self.circuit_breaker.check(
            portfolio_value=current_value,
            daily_start_value=self.daily_start_value if self.daily_start_value is not None else current_value,
            peak_value=self.peak_portfolio_value if self.peak_portfolio_value is not None else current_value,
            consecutive_losses=self.consecutive_losses,
            trade_history=self.trade_history
        )

        if should_trip and not self.circuit_breaker.is_tripped:
            # Trip the circuit breaker
            self.circuit_breaker.trip(trip_reason)
            self._save_state()  # Save state immediately
            return

        if self.circuit_breaker.is_tripped:
            logger.warning("=" * 80)
            logger.warning(f"⛔ CIRCUIT BREAKER TRIPPED: {self.circuit_breaker.trip_reason}")
            logger.warning(f"   Tripped at: {self.circuit_breaker.trip_timestamp}")
            logger.warning("   Trading halted until manual reset or next market day")
            logger.warning("=" * 80)
            return

        # Step 0.6: Check if trading is halted due to risk limits (legacy check)
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

        # Display stop loss cooldown status
        if self.stop_loss_cooldowns:
            logger.info(f"🚫 Stop Loss Cooldowns: {len(self.stop_loss_cooldowns)} symbols blacklisted")
            for sym, stop_date in sorted(self.stop_loss_cooldowns.items()):
                days_remaining = self.stop_loss_cooldown_days - (datetime.now() - stop_date).days
                if days_remaining > 0:
                    logger.info(f"   {sym}: {days_remaining} days remaining (until {(stop_date + timedelta(days=self.stop_loss_cooldown_days)).strftime('%m/%d')})")

        # Step 1: CHECK EXISTING POSITIONS FIRST (before generating new signals)
        # This ensures we capture profits and cut losses immediately
        # This runs every minute for time-sensitive sell signals
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: CHECKING EXISTING POSITIONS FOR EXIT TRIGGERS")
        logger.info("=" * 80)
        self._check_existing_positions()

        # Step 2: Check if we need to retrain (monthly or performance-based)
        if self._should_retrain():
            logger.info("Initiating model retraining...")
            self._retrain_models()

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
        Check if the market is currently open (uses Eastern Time + Alpaca Calendar API).

        This now uses the MarketCalendar utility which:
        - Gets accurate trading days from Alpaca API (no hardcoded holidays!)
        - Handles NYSE holidays automatically (Christmas, Thanksgiving, etc.)
        - Uses Eastern Time for all checks

        Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday (excluding holidays)

        Returns:
            bool: True if market is open, False otherwise
        """
        try:
            # Use market_calendar utility (imported from utils.market_calendar)
            current_time = now_et()
            market_open = is_market_open()

            if market_open:
                logger.debug(f"Market check: OPEN at {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            else:
                # Check if today is a trading day
                today_is_trading_day = is_trading_day(current_time.date())

                if not today_is_trading_day:
                    logger.debug(f"Market check: CLOSED - Holiday/Weekend ({current_time.strftime('%Y-%m-%d')})")
                else:
                    # Trading day but outside market hours
                    time_to_open = time_until_market_open()
                    if time_to_open:
                        hours = time_to_open.total_seconds() / 3600
                        logger.debug(f"Market check: CLOSED - Opens in {hours:.1f} hours")
                    else:
                        logger.debug(f"Market check: CLOSED - Outside hours")

            return market_open

        except Exception as e:
            logger.warning(f"Error checking market hours: {e}")
            # If we can't determine market status, assume it's open to avoid blocking trades
            return True

    def _should_retrain(self) -> bool:
        """
        Check if models need retraining based on schedule and performance.

        Two triggers for retraining:
        1. Time-based: More than retrain_days since last training
        2. Performance-based: Recent win rate drops below 40%

        Returns:
            True if retraining is needed
        """
        # First run - always need to train or load
        if self.last_train_date is None:
            logger.info("No training date - will attempt to load or train models")
            return True

        # Time-based check
        days_since_train = (datetime.now() - self.last_train_date).days

        if days_since_train >= self.retrain_days:
            logger.info(f"⏰ Retraining due: {days_since_train} days since last train (threshold: {self.retrain_days} days)")
            return True

        # Performance-based check
        # Only check if we have enough trade history
        if len(self.trade_history) >= 20:
            recent_trades = self.trade_history[-20:]
            wins = sum(1 for t in recent_trades if t['pnl'] > 0)
            recent_win_rate = wins / 20

            if recent_win_rate < 0.40:
                logger.warning("")
                logger.warning("=" * 60)
                logger.warning(f"📉 PERFORMANCE DROP DETECTED")
                logger.warning(f"   Recent win rate: {recent_win_rate:.1%} (threshold: 40%)")
                logger.warning(f"   Triggering early retrain to adapt to market changes")
                logger.warning("=" * 60)
                return True

        return False

    def _load_models_from_storage(self):
        """
        Load all symbol models from storage (S3 primary, local cache fallback).

        This is called on startup to restore previously trained models.
        S3 is the primary storage to ensure the bot is portable across machines.
        Local storage is used only as a cache when S3 is unavailable.
        """
        logger.info("Loading models from storage (S3 primary)...")

        for symbol in self.symbols:
            try:
                model_data = self.model_storage.load_model(
                    symbol,
                    version="latest",
                    prefer_s3=True  # S3 is primary storage, local is cache only
                )

                if model_data is None:
                    logger.warning(f"  ✗ {symbol}: No saved model found")
                    continue

                # Reconstruct strategy from saved model
                strategy = RiskLabAIStrategy(
                    profit_taking=self.risklabai_params['profit_taking'],
                    stop_loss=self.risklabai_params['stop_loss'],
                    max_holding=self.risklabai_params['max_holding'],
                    d=self.risklabai_params['d'],
                    n_cv_splits=5
                )

                # Load the trained components
                strategy.primary_model = model_data.get('primary_model')
                strategy.meta_model = model_data.get('meta_model')
                strategy.scaler = model_data.get('scaler')
                strategy.label_encoder = model_data.get('label_encoder')
                strategy.feature_names = model_data.get('feature_names')
                strategy.important_features = model_data.get('important_features')
                frac_diff_d = model_data.get('frac_diff_d')
                if frac_diff_d is not None:
                    strategy.frac_diff.d = frac_diff_d
                    strategy.frac_diff._optimal_d = frac_diff_d

                self.symbol_models[symbol] = strategy

                trained_at = model_data.get('trained_at', 'unknown')
                metrics = model_data.get('training_metrics', {})
                accuracy = metrics.get('primary_accuracy', 'N/A')
                logger.info(f"  ✓ {symbol}: Loaded (trained: {trained_at[:10]}, acc: {accuracy})")

            except Exception as e:
                logger.error(f"  ✗ {symbol}: Failed to load model: {e}")
                continue

        # Set last_train_date based on loaded models
        if self.symbol_models:
            # Use the most recent training date from loaded models
            try:
                training_dates = []
                for strategy in self.symbol_models.values():
                    if hasattr(strategy, 'primary_model'):
                        # Models exist, assume they were trained recently
                        training_dates.append(datetime.now())
                if training_dates:
                    self.last_train_date = min(training_dates)  # Use earliest to trigger retrain sooner
            except Exception:
                self.last_train_date = datetime.now()

    def _retrain_models(self):
        """
        Retrain all symbol models with latest data.

        This method:
        1. Fetches fresh bar data for each symbol
        2. Creates and trains a new RiskLabAIStrategy
        3. Saves the model locally AND to S3
        4. Updates the symbol_models dictionary
        """
        logger.info("=" * 80)
        logger.info("RETRAINING MODELS")
        logger.info("=" * 80)

        successful = 0
        failed = 0
        returns_by_symbol = {}

        for symbol in self.symbols:
            logger.info(f"\n📚 Retraining {symbol}...")

            # Create new strategy instance
            strategy = RiskLabAIStrategy(
                profit_taking=self.risklabai_params['profit_taking'],
                stop_loss=self.risklabai_params['stop_loss'],
                max_holding=self.risklabai_params['max_holding'],
                d=self.risklabai_params['d'],
                n_cv_splits=5
            )

            # Train the strategy
            try:
                results = strategy.train_from_ticks(
                    symbol=symbol,
                    threshold=self.imbalance_threshold,
                    min_samples=self.min_training_bars
                )
            except Exception as e:
                logger.error(f"  {symbol}: Training exception: {e}")
                failed += 1
                continue

            if not results['success']:
                logger.error(f"  {symbol}: Training failed")
                failed += 1
                continue

            # Prepare model data for storage
            model_data = {
                'primary_model': strategy.primary_model,
                'meta_model': strategy.meta_model,
                'scaler': strategy.scaler,
                'label_encoder': strategy.label_encoder,
                'feature_names': strategy.feature_names,
                'important_features': strategy.important_features,
                'frac_diff_d': strategy.frac_diff.d,
                'training_metrics': {
                    'primary_accuracy': results.get('primary_accuracy'),
                    'meta_accuracy': results.get('meta_accuracy'),
                    'n_samples': results.get('n_samples'),
                    'training_bars': results.get('bars_count')
                }
            }

            # Save to local AND S3
            try:
                self.model_storage.save_model(symbol, model_data, upload_to_s3=True)
            except Exception as e:
                logger.error(f"  {symbol}: Failed to save model: {e}")
                # Continue anyway - model is trained in memory

            # Update in-memory model reference
            self.symbol_models[symbol] = strategy

            logger.info(f"  ✓ {symbol} retrained: bal_acc={results['primary_accuracy']:.1%}, "
                       f"samples={results['n_samples']}")
            successful += 1

            if results.get('returns') is not None:
                returns_by_symbol[symbol] = results['returns']

        # Update training timestamp
        self.last_train_date = datetime.now()

        logger.info("=" * 80)
        logger.info(f"🎓 RETRAINING COMPLETE: {successful} success, {failed} failed")
        logger.info("=" * 80)

        # HRP portfolio optimization across symbols
        if returns_by_symbol:
            returns_df = pd.concat(returns_by_symbol, axis=1).dropna()
            if not returns_df.empty:
                try:
                    self.hrp_weights = self.risklabai.optimize_portfolio(returns_df)
                    logger.info("HRP portfolio weights computed")
                except Exception as e:
                    logger.warning(f"HRP optimization failed: {e}")

        # Cleanup old model versions (keep last 5)
        for symbol in self.symbols:
            try:
                self.model_storage.delete_old_versions(symbol, keep_count=5)
            except Exception as e:
                logger.debug(f"Cleanup failed for {symbol}: {e}")

    def _get_historical_bars(
        self,
        symbol: str,
        length: int,
        timeframe: str = "day"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical bar data for a symbol.

        Always fetches tick imbalance bars from the tick database.

        Args:
            symbol: Stock symbol
            length: Number of bars to fetch
            timeframe: Bar timeframe ("minute", "day", etc.) - ignored (tick data only)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.debug(f"{symbol}: Fetching tick imbalance bars from database")

            # Load ticks from database
            storage = TickStorage(TICK_DB_PATH)
            ticks = storage.load_ticks(symbol)
            storage.close()

            if not ticks:
                logger.warning(f"{symbol}: No ticks found in database")
                return None

            # Apply CUSUM filter before building imbalance bars
            filtered_ticks = self._filter_ticks_with_cusum(ticks, symbol)
            if not filtered_ticks:
                logger.warning(f"{symbol}: No ticks after CUSUM filtering")
                return None

            # Generate imbalance bars from CUSUM-filtered ticks
            bars = generate_bars_from_ticks(filtered_ticks, threshold=self.imbalance_threshold)

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

        except Exception as e:
            logger.error(f"Error fetching bars for {symbol}: {e}")
            return None

    def _filter_ticks_with_cusum(self, ticks, symbol: str):
        """
        Apply CUSUM event filtering to ticks before imbalance bar generation.

        Returns:
            List of ticks aligned to CUSUM event timestamps.
        """
        tick_timestamps = pd.to_datetime([t[0] for t in ticks], format='ISO8601')
        tick_prices = pd.Series([t[1] for t in ticks], index=tick_timestamps)

        cusum_events = self.cusum_filter.get_events(tick_prices)
        logger.debug(f"{symbol}: CUSUM events {len(cusum_events)} from {len(ticks):,} ticks")

        if len(cusum_events) == 0:
            return []

        event_values = cusum_events.values
        tick_values = tick_timestamps.values
        max_delta = np.timedelta64(CUSUM_EVENT_WINDOW_SECONDS, 's')

        idx = np.searchsorted(event_values, tick_values)
        large_delta = np.timedelta64(10**9, 's')
        prev_delta = np.full(len(tick_values), large_delta)
        next_delta = np.full(len(tick_values), large_delta)

        has_prev = idx > 0
        has_next = idx < len(event_values)
        prev_delta[has_prev] = tick_values[has_prev] - event_values[idx[has_prev] - 1]
        next_delta[has_next] = event_values[idx[has_next]] - tick_values[has_next]

        min_delta = np.minimum(prev_delta, next_delta)
        keep_mask = min_delta <= max_delta
        filtered_ticks = [tick for tick, keep in zip(ticks, keep_mask) if keep]

        logger.debug(
            f"{symbol}: Filtered ticks {len(filtered_ticks):,} after CUSUM "
            f"(window: ±{CUSUM_EVENT_WINDOW_SECONDS}s)"
        )
        return filtered_ticks

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

    def _update_dynamic_win_rate(self):
        """
        Calculate win rate from recent trade history.

        Uses rolling window of last 50 trades to adapt to changing market conditions.
        Clamps to realistic range (35-65%) to prevent extreme values.
        """
        if len(self.trade_history) < 10:
            # Need at least 10 trades before trusting the calculation
            self.dynamic_win_rate = 0.50  # Default 50% until enough history
            logger.debug(f"Dynamic win rate: 50.0% (need {10 - len(self.trade_history)} more trades)")
            return

        # Use last 50 trades (or all if less) for rolling window
        recent = self.trade_history[-50:]
        wins = sum(1 for t in recent if t['pnl'] > 0)
        calculated_win_rate = wins / len(recent)

        # Clamp to realistic range (35% to 65%)
        # Prevents extreme values from small sample sizes
        self.dynamic_win_rate = max(0.35, min(0.65, calculated_win_rate))

        logger.info("")
        logger.info("=" * 60)
        logger.info("DYNAMIC WIN RATE UPDATE")
        logger.info("=" * 60)
        logger.info(f"  Trades analyzed: {len(recent)}")
        logger.info(f"  Wins: {wins}, Losses: {len(recent) - wins}")
        logger.info(f"  Calculated win rate: {calculated_win_rate:.1%}")
        logger.info(f"  Dynamic win rate (clamped): {self.dynamic_win_rate:.1%}")
        logger.info("=" * 60)

    def _record_trade(self, symbol: str, pnl: float, entry_price: float, exit_price: float):
        """
        Record completed trade for dynamic win rate tracking.

        Args:
            symbol: Stock symbol
            pnl: P&L in dollars (positive = win, negative = loss)
            entry_price: Entry price
            exit_price: Exit price
        """
        self.trade_history.append({
            'symbol': symbol,
            'pnl': pnl,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'timestamp': datetime.now()
        })

        logger.info(f"📊 Trade recorded: {symbol} P&L=${pnl:+,.2f} (Total trades: {len(self.trade_history)})")

        # Update dynamic win rate with new trade
        self._update_dynamic_win_rate()

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

            # Capture trade details for recording (BEFORE exit)
            # Get entry price and current price from position/broker
            try:
                # Try to get price data from Alpaca API for accuracy
                if (hasattr(self, 'broker') and
                    self.broker is not None and
                    hasattr(self.broker, 'api') and
                    self.broker.api is not None):
                    alpaca_position = self.broker.api.get_position(symbol)
                    entry_price = float(alpaca_position.avg_entry_price)
                    exit_price = float(alpaca_position.current_price)
                    pnl_dollars = float(alpaca_position.unrealized_pl)
                else:
                    # Fallback to Lumibot methods
                    entry_price = position.avg_fill_price if hasattr(position, 'avg_fill_price') else 0
                    exit_price = self.get_last_price(symbol) or 0
                    pnl_dollars = (exit_price - entry_price) * quantity
            except Exception as e:
                logger.warning(f"Could not get precise trade details for recording: {e}")
                entry_price = 0
                exit_price = 0
                pnl_dollars = 0

            # Issue simple SELL order (bot manages independently)
            order = self.create_order(symbol, quantity, "sell")
            self.submit_order(order)

            logger.info(f"✅ SELL ORDER SUBMITTED: {quantity} shares of {symbol}")
            logger.info(f"   Exit reason: {reason}")

            # Record trade for dynamic win rate calculation
            if entry_price > 0 and exit_price > 0:
                self._record_trade(symbol, pnl_dollars, entry_price, exit_price)
            else:
                logger.warning(f"   Could not record trade (invalid prices: entry=${entry_price}, exit=${exit_price})")

            # If this is a stop loss exit, add to cooldown to prevent immediate re-entry
            if "STOP LOSS" in reason.upper():
                self.stop_loss_cooldowns[symbol] = datetime.now()
                logger.warning(f"🚫 {symbol} added to stop loss cooldown for {self.stop_loss_cooldown_days} days")
                logger.warning(f"   Will not re-buy until {(datetime.now() + timedelta(days=self.stop_loss_cooldown_days)).strftime('%Y-%m-%d')}")

        except Exception as e:
            logger.error(f"Error exiting position for {symbol}: {e}")

    def _process_symbol(self, symbol: str):
        """Generate and execute signal for one symbol with Kelly Criterion sizing."""
        # Check if we have a trained model for this symbol
        if symbol not in self.symbol_models:
            logger.debug(f"{symbol}: No trained model available, skipping")
            return

        # Check if symbol is in stop loss cooldown
        if symbol in self.stop_loss_cooldowns:
            stop_loss_date = self.stop_loss_cooldowns[symbol]
            days_since_stop = (datetime.now() - stop_loss_date).days
            days_remaining = self.stop_loss_cooldown_days - days_since_stop

            if days_remaining > 0:
                logger.debug(f"{symbol}: In stop loss cooldown ({days_remaining} days remaining)")
                return
            else:
                # Cooldown expired, remove from dict
                logger.info(f"{symbol}: Stop loss cooldown expired, removing from blacklist")
                del self.stop_loss_cooldowns[symbol]

        # Get recent bars (need enough for feature calculation)
        bars = self._get_historical_bars(symbol, 100)

        if bars is None or len(bars) < 50:
            logger.info(f"{symbol}: Insufficient data ({len(bars) if bars is not None else 0} bars)")
            return

        # Get signal from RiskLabAI model (tick-level CUSUM happens before bar generation)
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
            # Calculate Kelly fraction using DYNAMIC win rate (adapts to actual performance)
            kelly_calc = KellyCriterion()
            kelly_fraction = kelly_calc.calculate_kelly(
                win_rate=self.dynamic_win_rate,  # DYNAMIC: Updates based on actual trades (was: self.estimated_win_rate)
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
            logger.info(f"      Using dynamic win rate: {self.dynamic_win_rate:.1%} (from {len(self.trade_history)} trades)")
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

        # NEW: Save state before shutdown (cooldowns, trade history, etc.)
        try:
            self._save_state()
        except Exception as e:
            logger.error(f"Failed to save state on shutdown: {e}")

        # Save models to S3 if trained (ensures latest models are backed up)
        if self.models_trained and self.model_storage.s3_client:
            try:
                for symbol, strategy in self.symbol_models.items():
                    # Prepare model data for S3 storage
                    model_data = {
                        'primary_model': strategy.primary_model,
                        'meta_model': strategy.meta_model,
                        'scaler': strategy.scaler,
                        'label_encoder': strategy.label_encoder,
                        'feature_names': strategy.feature_names,
                        'important_features': strategy.important_features,
                        'frac_diff_d': strategy.frac_diff.d,
                    }
                    # Save to S3 (will also cache locally)
                    self.model_storage.save_model(symbol, model_data, upload_to_s3=True)
                logger.info(f"Models saved to S3 for {len(self.symbol_models)} symbols on shutdown")
            except Exception as e:
                logger.error(f"Failed to save models on shutdown: {e}")
        elif self.models_trained and not self.model_storage.s3_client:
            logger.warning("⚠️  S3 not configured - models not backed up on shutdown")

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
