"""
Daily P&L Tracker and Circuit Breaker

FIXES PROBLEM 10 by:
- Tracking daily profit/loss (realized + unrealized)
- Implementing circuit breaker to stop trading on excessive losses
- Dynamically adjusting position sizes based on daily P&L
- Protecting capital from catastrophic daily losses

NOTE: Cannot guarantee zero losses (impossible), but minimizes daily loss risk.
"""

import logging
from typing import Dict, Tuple, Optional
from datetime import datetime, date
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DailyPnLSnapshot:
    """Snapshot of daily P&L at a point in time."""
    timestamp: datetime
    portfolio_value: float
    starting_value: float
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    pnl_percent: float
    trades_today: int


class DailyPnLTracker:
    """
    Tracks daily profit/loss and implements circuit breaker.

    FIXES PROBLEM 10, 16: Cannot guarantee no losses, but limits daily loss exposure.
    All default parameters now from config.settings (environment-specific).

    Features:
    - Daily P&L tracking (realized + unrealized)
    - Circuit breaker stops trading on excessive loss
    - Dynamic position sizing reduces exposure as losses mount
    - Resets daily at market open
    """

    def __init__(self,
                 max_daily_loss_pct: float = None,
                 warning_loss_pct: float = None,
                 scaling_start_loss_pct: float = None):
        """
        Initialize daily P&L tracker.

        FIXED (Problem 16): Defaults now from config.settings (environment-specific).

        Args:
            max_daily_loss_pct: Circuit breaker threshold (% of portfolio, default: from config)
            warning_loss_pct: Warning threshold (% of portfolio, default: from config)
            scaling_start_loss_pct: When to start scaling down positions (%, default: from config)
        """
        # FIXED (Problem 16): Use config defaults if not provided
        from config.settings import (
            MAX_DAILY_LOSS_PCT,
            WARNING_LOSS_PCT,
            SCALING_START_LOSS_PCT
        )

        self.max_daily_loss_pct = max_daily_loss_pct if max_daily_loss_pct is not None else MAX_DAILY_LOSS_PCT
        self.warning_loss_pct = warning_loss_pct if warning_loss_pct is not None else WARNING_LOSS_PCT
        self.scaling_start_loss_pct = scaling_start_loss_pct if scaling_start_loss_pct is not None else SCALING_START_LOSS_PCT

        # Daily tracking
        self.current_date = date.today()
        self.starting_portfolio_value: Optional[float] = None
        self.daily_snapshots: list[DailyPnLSnapshot] = []
        self.trades_today = 0

        # Circuit breaker state
        self.circuit_breaker_triggered = False
        self.circuit_breaker_time: Optional[datetime] = None

        logger.info("✅ Daily P&L Tracker initialized")
        logger.info(f"   Max daily loss: {self.max_daily_loss_pct:.1f}%")
        logger.info(f"   Warning threshold: {self.warning_loss_pct:.1f}%")
        logger.info(f"   Scaling starts at: {self.scaling_start_loss_pct:.1f}%")

    def check_new_day(self):
        """Check if it's a new trading day and reset if needed."""
        today = date.today()
        if today != self.current_date:
            logger.info("=" * 60)
            logger.info(f"NEW TRADING DAY: {today}")
            logger.info("=" * 60)

            # Log yesterday's summary if we have data
            if self.daily_snapshots:
                last_snapshot = self.daily_snapshots[-1]
                logger.info(f"Previous day summary ({self.current_date}):")
                logger.info(f"  Total P&L: ${last_snapshot.total_pnl:,.2f} ({last_snapshot.pnl_percent:+.2f}%)")
                logger.info(f"  Trades: {last_snapshot.trades_today}")

            # Reset for new day
            self.current_date = today
            self.starting_portfolio_value = None
            self.daily_snapshots = []
            self.trades_today = 0
            self.circuit_breaker_triggered = False
            self.circuit_breaker_time = None

            logger.info("Daily tracking reset for new day")
            logger.info("=" * 60)

    def update(self,
               current_portfolio_value: float,
               realized_pnl_today: float = 0.0,
               trade_executed: bool = False) -> DailyPnLSnapshot:
        """
        Update daily P&L tracking.

        Args:
            current_portfolio_value: Current total portfolio value
            realized_pnl_today: Realized P&L from trades today (optional)
            trade_executed: Whether a trade was just executed

        Returns:
            Current daily P&L snapshot
        """
        # Check if new day
        self.check_new_day()

        # Set starting value on first update of the day
        if self.starting_portfolio_value is None:
            self.starting_portfolio_value = current_portfolio_value
            logger.info(f"📊 Starting portfolio value: ${current_portfolio_value:,.2f}")

        # Calculate P&L
        total_pnl = current_portfolio_value - self.starting_portfolio_value
        unrealized_pnl = total_pnl - realized_pnl_today
        pnl_percent = (total_pnl / self.starting_portfolio_value * 100) if self.starting_portfolio_value > 0 else 0.0

        # Track trades
        if trade_executed:
            self.trades_today += 1

        # Create snapshot
        snapshot = DailyPnLSnapshot(
            timestamp=datetime.now(),
            portfolio_value=current_portfolio_value,
            starting_value=self.starting_portfolio_value,
            realized_pnl=realized_pnl_today,
            unrealized_pnl=unrealized_pnl,
            total_pnl=total_pnl,
            pnl_percent=pnl_percent,
            trades_today=self.trades_today
        )

        self.daily_snapshots.append(snapshot)

        # Check circuit breaker
        if not self.circuit_breaker_triggered:
            if pnl_percent <= -self.max_daily_loss_pct:
                self._trigger_circuit_breaker(snapshot)
            elif pnl_percent <= -self.warning_loss_pct:
                logger.warning(f"⚠️ WARNING: Daily loss at {pnl_percent:.2f}% "
                              f"(threshold: {self.warning_loss_pct:.1f}%)")

        return snapshot

    def _trigger_circuit_breaker(self, snapshot: DailyPnLSnapshot):
        """Trigger circuit breaker to stop all trading."""
        self.circuit_breaker_triggered = True
        self.circuit_breaker_time = datetime.now()

        logger.error("=" * 80)
        logger.error("🚨 CIRCUIT BREAKER TRIGGERED 🚨")
        logger.error("=" * 80)
        logger.error(f"Daily loss: {snapshot.pnl_percent:.2f}% (${snapshot.total_pnl:,.2f})")
        logger.error(f"Threshold: {self.max_daily_loss_pct:.1f}%")
        logger.error(f"Time: {self.circuit_breaker_time.strftime('%I:%M %p')}")
        logger.error("")
        logger.error("ALL TRADING STOPPED FOR TODAY")
        logger.error("Circuit breaker will reset tomorrow at market open")
        logger.error("=" * 80)

    def can_trade(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed.

        Returns:
            Tuple of (can_trade, reason)
        """
        # Check if new day (auto-reset circuit breaker)
        self.check_new_day()

        if self.circuit_breaker_triggered:
            time_str = self.circuit_breaker_time.strftime('%I:%M %p') if self.circuit_breaker_time else "unknown"
            return (False, f"circuit_breaker_triggered_at_{time_str}")

        return (True, "trading_allowed")

    def get_position_size_multiplier(self) -> float:
        """
        Get position size multiplier based on current daily P&L.

        FIXES PROBLEM 10: Reduces position sizes as daily losses increase.

        Returns:
            Multiplier (0.0 to 1.0) to apply to position sizes
            - 1.0 = full size (no losses or small gains)
            - 0.5 = half size (moderate losses)
            - 0.25 = quarter size (significant losses)
            - 0.0 = no trading (circuit breaker)
        """
        if self.circuit_breaker_triggered:
            return 0.0

        if not self.daily_snapshots:
            return 1.0  # No data yet, use full size

        current_pnl_pct = self.daily_snapshots[-1].pnl_percent

        # If profitable or small loss, use full size
        if current_pnl_pct >= -self.scaling_start_loss_pct:
            return 1.0

        # Scale down linearly from scaling_start to warning threshold
        # At scaling_start (-0.5%): 1.0x
        # At warning (-1.5%): 0.5x
        # At circuit breaker (-3.0%): 0.0x

        if current_pnl_pct >= -self.warning_loss_pct:
            # Interpolate between 1.0 and 0.5
            range_size = self.warning_loss_pct - self.scaling_start_loss_pct
            progress = (-current_pnl_pct - self.scaling_start_loss_pct) / range_size
            multiplier = 1.0 - (progress * 0.5)  # 1.0 -> 0.5
            return max(0.5, multiplier)

        elif current_pnl_pct >= -self.max_daily_loss_pct:
            # Interpolate between 0.5 and 0.25
            range_size = self.max_daily_loss_pct - self.warning_loss_pct
            progress = (-current_pnl_pct - self.warning_loss_pct) / range_size
            multiplier = 0.5 - (progress * 0.25)  # 0.5 -> 0.25
            return max(0.25, multiplier)

        else:
            # At or beyond circuit breaker
            return 0.0

    def get_daily_status(self) -> Dict[str, any]:
        """
        Get comprehensive daily P&L status.

        Returns:
            Dict with daily status metrics
        """
        if not self.daily_snapshots:
            return {
                'has_data': False,
                'can_trade': True,
                'position_size_multiplier': 1.0,
                'circuit_breaker_triggered': False
            }

        latest = self.daily_snapshots[-1]
        can_trade, trade_reason = self.can_trade()
        multiplier = self.get_position_size_multiplier()

        return {
            'has_data': True,
            'date': self.current_date.isoformat(),
            'starting_value': latest.starting_value,
            'current_value': latest.portfolio_value,
            'total_pnl': latest.total_pnl,
            'pnl_percent': latest.pnl_percent,
            'realized_pnl': latest.realized_pnl,
            'unrealized_pnl': latest.unrealized_pnl,
            'trades_today': latest.trades_today,
            'can_trade': can_trade,
            'trade_reason': trade_reason,
            'circuit_breaker_triggered': self.circuit_breaker_triggered,
            'position_size_multiplier': multiplier,
            'warning_threshold': self.warning_loss_pct,
            'circuit_breaker_threshold': self.max_daily_loss_pct
        }

    def log_daily_summary(self):
        """Log a summary of today's P&L."""
        status = self.get_daily_status()

        if not status['has_data']:
            logger.info("No daily P&L data yet")
            return

        logger.info("=" * 60)
        logger.info("DAILY P&L SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Date: {status['date']}")
        logger.info(f"Starting value: ${status['starting_value']:,.2f}")
        logger.info(f"Current value: ${status['current_value']:,.2f}")
        logger.info(f"Total P&L: ${status['total_pnl']:,.2f} ({status['pnl_percent']:+.2f}%)")
        logger.info(f"  Realized: ${status['realized_pnl']:,.2f}")
        logger.info(f"  Unrealized: ${status['unrealized_pnl']:,.2f}")
        logger.info(f"Trades today: {status['trades_today']}")
        logger.info(f"")
        logger.info(f"Risk Management:")
        logger.info(f"  Can trade: {status['can_trade']}")
        logger.info(f"  Position size: {status['position_size_multiplier'] * 100:.0f}%")

        if status['circuit_breaker_triggered']:
            logger.info(f"  🚨 CIRCUIT BREAKER ACTIVE")

        logger.info("=" * 60)


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_daily_pnl_tracker = None

def get_daily_pnl_tracker(
    max_daily_loss_pct: float = 3.0,
    warning_loss_pct: float = 1.5,
    scaling_start_loss_pct: float = 0.5
) -> DailyPnLTracker:
    """
    Get the global DailyPnLTracker instance (creates it if needed).

    Args:
        max_daily_loss_pct: Circuit breaker threshold (default: 3%)
        warning_loss_pct: Warning threshold (default: 1.5%)
        scaling_start_loss_pct: When to start scaling (default: 0.5%)

    Returns:
        Singleton DailyPnLTracker instance
    """
    global _daily_pnl_tracker

    if _daily_pnl_tracker is None:
        _daily_pnl_tracker = DailyPnLTracker(
            max_daily_loss_pct=max_daily_loss_pct,
            warning_loss_pct=warning_loss_pct,
            scaling_start_loss_pct=scaling_start_loss_pct
        )

    return _daily_pnl_tracker
