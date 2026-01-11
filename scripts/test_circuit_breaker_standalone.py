#!/usr/bin/env python3
"""
Standalone Test for Circuit Breaker (No Dependencies)

This script tests the CircuitBreaker class in isolation without
requiring lumibot or other heavy dependencies.

PROMPT 18: Circuit Breaker Pattern [D3]
"""

import logging
from datetime import datetime, timedelta
from typing import Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Copy of CircuitBreaker class for standalone testing
class CircuitBreaker:
    """Circuit breaker pattern to halt trading on anomalous conditions."""

    def __init__(self,
                 max_daily_loss: float = 0.03,
                 max_drawdown: float = 0.10,
                 max_consecutive_losses: int = 5,
                 max_trades_per_hour: int = 10):
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
        """Check if circuit breaker should trip."""
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


print("=" * 80)
print("CIRCUIT BREAKER STANDALONE TEST")
print("=" * 80)


def test_daily_loss_limit():
    """Test 1: Daily loss limit trigger."""
    print("\n" + "=" * 80)
    print("TEST 1: Daily Loss Limit Trigger (3%)")
    print("=" * 80)

    cb = CircuitBreaker(max_daily_loss=0.03)

    # Scenario: Portfolio drops from $100,000 to $96,500 (-3.5%)
    should_trip, reason = cb.check(
        portfolio_value=96500,
        daily_start_value=100000,
        peak_value=105000,
        consecutive_losses=0,
        trade_history=[]
    )

    if should_trip and "Daily loss limit" in reason:
        print(f"✓ Circuit breaker triggered: {reason}")
        return True
    else:
        print(f"✗ Circuit breaker should have triggered on 3.5% daily loss")
        return False


def test_drawdown_limit():
    """Test 2: Drawdown limit trigger."""
    print("\n" + "=" * 80)
    print("TEST 2: Drawdown Limit Trigger (10%)")
    print("=" * 80)

    cb = CircuitBreaker(max_drawdown=0.10)

    # Scenario: Portfolio at $105,000 peak, now $94,000 (-10.48% drawdown)
    # Daily start was $95,000 so daily gain is +1.05% (not a loss)
    should_trip, reason = cb.check(
        portfolio_value=94000,
        daily_start_value=95000,  # Started today at 95k (not triggering daily loss)
        peak_value=105000,
        consecutive_losses=0,
        trade_history=[]
    )

    if should_trip and "drawdown" in reason:
        print(f"✓ Circuit breaker triggered: {reason}")
        return True
    else:
        print(f"✗ Circuit breaker should have triggered on 10.95% drawdown")
        return False


def test_consecutive_losses():
    """Test 3: Consecutive losses trigger."""
    print("\n" + "=" * 80)
    print("TEST 3: Consecutive Losses Trigger (5)")
    print("=" * 80)

    cb = CircuitBreaker(max_consecutive_losses=5)

    should_trip, reason = cb.check(
        portfolio_value=98000,
        daily_start_value=100000,
        peak_value=105000,
        consecutive_losses=5,
        trade_history=[]
    )

    if should_trip and "Consecutive losses" in reason:
        print(f"✓ Circuit breaker triggered: {reason}")
        return True
    else:
        print(f"✗ Circuit breaker should have triggered on 5 consecutive losses")
        return False


def test_trades_per_hour():
    """Test 4: Trades per hour limit trigger."""
    print("\n" + "=" * 80)
    print("TEST 4: Trades Per Hour Limit Trigger (10)")
    print("=" * 80)

    cb = CircuitBreaker(max_trades_per_hour=10)

    # Create 12 recent trades
    now = datetime.now()
    trade_history = [
        {'timestamp': now - timedelta(minutes=i*5), 'symbol': 'SPY', 'profit': 100}
        for i in range(12)
    ]

    should_trip, reason = cb.check(
        portfolio_value=100000,
        daily_start_value=100000,
        peak_value=105000,
        consecutive_losses=0,
        trade_history=trade_history
    )

    if should_trip and "trades per hour" in reason:
        print(f"✓ Circuit breaker triggered: {reason}")
        return True
    else:
        print(f"✗ Circuit breaker should have triggered on 12 trades/hour")
        return False


def test_no_trigger():
    """Test 5: Normal conditions (no trigger)."""
    print("\n" + "=" * 80)
    print("TEST 5: Normal Conditions (No Trigger)")
    print("=" * 80)

    cb = CircuitBreaker()

    # Scenario: Portfolio up 2%, 4 consecutive losses (below limit)
    now = datetime.now()
    trade_history = [
        {'timestamp': now - timedelta(minutes=i*10), 'symbol': 'SPY', 'profit': 100}
        for i in range(5)
    ]

    should_trip, reason = cb.check(
        portfolio_value=102000,
        daily_start_value=100000,
        peak_value=105000,
        consecutive_losses=4,
        trade_history=trade_history
    )

    if not should_trip:
        print("✓ Circuit breaker correctly NOT triggered (normal conditions)")
        return True
    else:
        print(f"✗ Circuit breaker should NOT trigger: {reason}")
        return False


def test_trip_and_reset():
    """Test 6: Trip and reset functionality."""
    print("\n" + "=" * 80)
    print("TEST 6: Trip and Reset Functionality")
    print("=" * 80)

    cb = CircuitBreaker()

    # Trip the circuit breaker
    cb.trip("Test trip reason")

    if cb.is_tripped and cb.trip_reason == "Test trip reason":
        print(f"✓ Circuit breaker tripped successfully")
    else:
        print("✗ Circuit breaker trip() failed")
        return False

    # Reset the circuit breaker
    cb.reset()

    if not cb.is_tripped and cb.trip_reason is None:
        print("✓ Circuit breaker reset successfully")
        return True
    else:
        print("✗ Circuit breaker reset() failed")
        return False


def test_auto_reset():
    """Test 7: Auto-reset on new trading day."""
    print("\n" + "=" * 80)
    print("TEST 7: Auto-Reset on New Trading Day")
    print("=" * 80)

    cb = CircuitBreaker()

    # Trip the circuit breaker with a timestamp from yesterday
    cb.is_tripped = True
    cb.trip_reason = "Test trip from yesterday"
    cb.trip_timestamp = datetime.now() - timedelta(days=1)

    # Check if should auto-reset
    if cb.should_auto_reset():
        print(f"✓ Circuit breaker correctly identified need for auto-reset")
        cb.reset()

        if not cb.is_tripped:
            print("✓ Auto-reset successful")
            return True
        else:
            print("✗ Auto-reset failed")
            return False
    else:
        print("✗ Circuit breaker should auto-reset (trip was yesterday)")
        return False


if __name__ == "__main__":
    print("\nRunning test suite...\n")

    # Run tests
    test1_passed = test_daily_loss_limit()
    test2_passed = test_drawdown_limit()
    test3_passed = test_consecutive_losses()
    test4_passed = test_trades_per_hour()
    test5_passed = test_no_trigger()
    test6_passed = test_trip_and_reset()
    test7_passed = test_auto_reset()

    # Summary
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Test 1 - Daily Loss Limit: {'✓ PASS' if test1_passed else '✗ FAIL'}")
    print(f"Test 2 - Drawdown Limit: {'✓ PASS' if test2_passed else '✗ FAIL'}")
    print(f"Test 3 - Consecutive Losses: {'✓ PASS' if test3_passed else '✗ FAIL'}")
    print(f"Test 4 - Trades Per Hour: {'✓ PASS' if test4_passed else '✗ FAIL'}")
    print(f"Test 5 - Normal Conditions: {'✓ PASS' if test5_passed else '✗ FAIL'}")
    print(f"Test 6 - Trip and Reset: {'✓ PASS' if test6_passed else '✗ FAIL'}")
    print(f"Test 7 - Auto-Reset: {'✓ PASS' if test7_passed else '✗ FAIL'}")
    print("=" * 80)

    all_passed = all([
        test1_passed, test2_passed, test3_passed, test4_passed,
        test5_passed, test6_passed, test7_passed
    ])

    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nValidation:")
        print("  - Daily loss limit (3%) triggers circuit breaker")
        print("  - Max drawdown (10%) triggers circuit breaker")
        print("  - Consecutive losses (5) triggers circuit breaker")
        print("  - Trades per hour limit (10) triggers circuit breaker")
        print("  - Normal conditions do NOT trigger circuit breaker")
        print("  - Trip and reset functionality works correctly")
        print("  - Auto-reset on new trading day works correctly")
    else:
        print("✗ SOME TESTS FAILED")

    print("=" * 80)

    import sys
    sys.exit(0 if all_passed else 1)
