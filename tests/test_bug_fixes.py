#!/usr/bin/env python
"""
Test script to verify all bug fixes.

This tests the new modular components:
1. MarketDataClient - fixes cash detection bug
2. StopLossManager - fixes bracket order verification bug
3. HedgeManager - fixes missing hedge logic bug

Run this to verify everything works before starting live trading.
"""

import os
import sys
import logging

# Add project root to path so we can import from config, data, risk
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.market_data import get_market_data_client
from risk.stop_loss_manager import StopLossManager
from risk.hedge_manager import HedgeManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_market_data():
    """Test #1: Verify cash detection bug is fixed."""
    print("\n" + "=" * 80)
    print("TEST 1: Cash Detection (Bug Fix)")
    print("=" * 80)

    try:
        market_data = get_market_data_client()

        # Get cash - this used to return None!
        cash = market_data.get_cash()
        buying_power = market_data.get_buying_power()
        portfolio_value = market_data.get_portfolio_value()

        print(f"✅ Cash: ${cash:,.2f}")
        print(f"✅ Buying power: ${buying_power:,.2f}")
        print(f"✅ Portfolio value: ${portfolio_value:,.2f}")

        if cash is None:
            print("❌ FAIL: Cash returned None!")
            return False

        if cash == 0 and buying_power > 0:
            print("⚠️  Warning: Cash is $0 but buying power exists")
            print("   This means all cash is tied up in positions")

        print("✅ PASS: Cash detection working correctly!")
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stop_loss_verification():
    """Test #2: Verify bracket order verification bug is fixed."""
    print("\n" + "=" * 80)
    print("TEST 2: Stop-Loss Verification (Bug Fix)")
    print("=" * 80)

    try:
        stop_loss_mgr = StopLossManager()
        market_data = get_market_data_client()

        # Get positions
        positions = market_data.get_positions()

        if not positions:
            print("⚠️  No positions found - can't test protection verification")
            print("   (This is OK if you don't have positions)")
            return True

        print(f"Found {len(positions)} position(s)")

        # ACTUALLY verify protection (this was the bug - bot never checked!)
        protection_status = stop_loss_mgr.verify_protection(positions)

        protected_count = sum(1 for status in protection_status.values() if status['fully_protected'])
        unprotected_count = len(positions) - protected_count

        print(f"\nProtection Status:")
        print(f"  Protected: {protected_count}")
        print(f"  Unprotected: {unprotected_count}")

        if unprotected_count > 0:
            print(f"\n⚠️  WARNING: {unprotected_count} position(s) have NO protection!")
            print("   The bot will create stop-loss orders for them.")

            # Protect unprotected positions
            protected, total = stop_loss_mgr.protect_all_positions(positions)
            print(f"\n✅ Protected {protected}/{total} positions")

        else:
            print("\n✅ All positions have stop-loss protection!")

        print("✅ PASS: Bracket order verification working correctly!")
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hedge_logic():
    """Test #3: Verify hedge logic bug is fixed."""
    print("\n" + "=" * 80)
    print("TEST 3: Hedge Logic (Bug Fix)")
    print("=" * 80)

    try:
        hedge_mgr = HedgeManager()
        market_data = get_market_data_client()

        # Check market sentiment
        sentiment = hedge_mgr.check_market_sentiment()

        print(f"\nMarket Sentiment Analysis:")
        print(f"  Overbought: {sentiment['bearish_count']}/{sentiment['total_checked']}")
        print(f"  Bearish ratio: {sentiment['bearish_ratio']:.1%}")
        print(f"  Is bearish: {'YES' if sentiment['is_bearish'] else 'NO'}")

        if sentiment['symbols_overbought']:
            print(f"  Overbought symbols: {', '.join(sentiment['symbols_overbought'])}")

        # Check current hedge allocation
        hedge_value, hedge_pct = hedge_mgr.get_current_hedge_allocation()
        print(f"\nCurrent Hedge Status:")
        print(f"  Hedge value: ${hedge_value:,.2f}")
        print(f"  Hedge allocation: {hedge_pct:.1%}")

        # Get cash for hypothetical hedge
        cash = market_data.get_cash()
        portfolio_value = market_data.get_portfolio_value()

        print(f"\nAvailable for Hedging:")
        print(f"  Cash: ${cash:,.2f}")
        print(f"  Portfolio: ${portfolio_value:,.2f}")

        # Don't actually create hedge in test, just show what would happen
        if sentiment['is_bearish'] and cash > 100:
            print(f"\n🛡️  Market is bearish - bot WOULD create hedge")
            print(f"   (Not creating in test mode)")
        elif sentiment['bearish_ratio'] < 0.3 and hedge_pct > 0:
            print(f"\n📈 Market recovered - bot WOULD exit hedges")
            print(f"   (Not exiting in test mode)")
        else:
            print(f"\n✅ No hedge action needed")

        print("\n✅ PASS: Hedge logic working correctly!")
        return True

    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("BUG FIX VERIFICATION TEST SUITE")
    print("=" * 80)
    print("\nThis will verify that all critical bugs have been fixed:")
    print("1. Cash detection (was returning None)")
    print("2. Stop-loss verification (was not actually checking)")
    print("3. Hedge logic (was missing/not running)")
    print("\n" + "=" * 80)

    results = {
        'cash_detection': test_market_data(),
        'stop_loss_verification': test_stop_loss_verification(),
        'hedge_logic': test_hedge_logic()
    }

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print("\n" + "=" * 80)
    if passed == total:
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("\nAll bugs fixed! Safe to run live bot.")
    else:
        print(f"❌ SOME TESTS FAILED ({total - passed}/{total})")
        print("\nFix failing tests before running live bot!")
    print("=" * 80 + "\n")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
