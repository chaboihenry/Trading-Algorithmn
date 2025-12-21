"""
Fix Existing Positions - Cancel old stop-loss orders and create OCO orders.

This script fixes the "shares held by orders" problem by:
1. Canceling all existing stop-loss orders (which hold shares)
2. Creating new protection using separate stop-loss and take-profit orders
   that DON'T conflict with each other

NOTE: We use stop-MARKET orders instead of stop-LIMIT to avoid holding shares.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, StopOrderRequest, LimitOrderRequest
from alpaca.trading.enums import (
    OrderSide, OrderType, TimeInForce, QueryOrderStatus
)
from data.market_data import get_market_data_client
from config.settings import (
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    ALPACA_PAPER,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    ENABLE_EXTENDED_HOURS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cancel_all_stop_orders():
    """Cancel all existing stop-loss orders."""

    client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=ALPACA_PAPER)

    # Get all open orders
    request = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100)
    orders = client.get_orders(request)

    # Filter for stop orders
    stop_orders = [o for o in orders if o.type in [OrderType.STOP, OrderType.STOP_LIMIT]]

    logger.info(f"Found {len(stop_orders)} stop-loss orders to cancel")

    for order in stop_orders:
        try:
            client.cancel_order_by_id(order.id)
            logger.info(f"  ✅ Canceled stop-loss for {order.symbol} (Order #{order.id})")
        except Exception as e:
            logger.error(f"  ❌ Failed to cancel order {order.id}: {e}")

    return len(stop_orders)


def create_protection_for_position(client, position):
    """
    Create stop-loss and take-profit for a position.

    Uses stop-MARKET (not stop-limit) to avoid holding shares.
    """
    symbol = position.symbol
    quantity = float(position.qty)
    entry_price = float(position.avg_entry_price)
    current_price = float(position.current_price)

    logger.info(f"\n📊 Protecting {symbol}:")
    logger.info(f"   Quantity: {quantity:.2f} shares")
    logger.info(f"   Entry: ${entry_price:.2f}, Current: ${current_price:.2f}")

    # Calculate prices
    stop_price = round(entry_price * (1 - STOP_LOSS_PCT), 2)
    tp_price = round(entry_price * (1 + TAKE_PROFIT_PCT), 2)

    logger.info(f"   Stop-loss: ${stop_price:.2f} (-{STOP_LOSS_PCT:.1%})")
    logger.info(f"   Take-profit: ${tp_price:.2f} (+{TAKE_PROFIT_PCT:.1%})")

    # Check market status
    try:
        clock = client.get_clock()
        market_is_open = clock.is_open
    except:
        market_is_open = False

    success_count = 0

    # Create stop-MARKET order (doesn't hold shares like stop-limit does)
    try:
        stop_order = StopOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC if market_is_open else TimeInForce.DAY,
            stop_price=stop_price
        )

        result = client.submit_order(stop_order)
        logger.info(f"   ✅ Stop-loss created (Order #{result.id})")
        success_count += 1
    except Exception as e:
        logger.error(f"   ❌ Failed to create stop-loss: {e}")

    # Create take-profit limit order
    try:
        if market_is_open and ENABLE_EXTENDED_HOURS:
            time_in_force = TimeInForce.GTC
            extended_hours = True
        else:
            time_in_force = TimeInForce.DAY
            extended_hours = False

        tp_order = LimitOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=OrderSide.SELL,
            time_in_force=time_in_force,
            limit_price=tp_price,
            extended_hours=extended_hours
        )

        result = client.submit_order(tp_order)
        logger.info(f"   ✅ Take-profit created (Order #{result.id})")
        success_count += 1
    except Exception as e:
        logger.error(f"   ❌ Failed to create take-profit: {e}")

    return success_count == 2


def main():
    """Main function to fix all existing positions."""

    logger.info("=" * 80)
    logger.info("FIX EXISTING POSITIONS - Remove old orders and create new protection")
    logger.info("=" * 80)

    # Step 1: Cancel all existing stop-loss orders
    logger.info("\nStep 1: Canceling existing stop-loss orders...")
    canceled_count = cancel_all_stop_orders()

    if canceled_count > 0:
        logger.info(f"\n✅ Canceled {canceled_count} stop-loss orders")
        logger.info("   (Shares are now free to be protected with new orders)")

    # Step 2: Get all positions
    logger.info("\nStep 2: Getting all positions...")
    market_data = get_market_data_client()
    positions = market_data.get_positions()
    logger.info(f"Found {len(positions)} positions to protect")

    if not positions:
        logger.info("No positions found!")
        return

    # Step 3: Create new protection for each position
    logger.info("\nStep 3: Creating new protection (stop-market + take-profit)...")

    client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=ALPACA_PAPER)

    fully_protected = 0
    for position in positions:
        if create_protection_for_position(client, position):
            fully_protected += 1

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total positions: {len(positions)}")
    logger.info(f"Fully protected: {fully_protected}/{len(positions)}")

    if fully_protected == len(positions):
        logger.info("\n✅ ALL POSITIONS FULLY PROTECTED!")
        logger.info("   Each position now has:")
        logger.info("   - Stop-MARKET order (doesn't hold shares)")
        logger.info("   - Take-profit LIMIT order")
    else:
        logger.warning(f"\n⚠️  {len(positions) - fully_protected} positions failed")
        logger.warning("   Check error messages above for details")

    logger.info("\n💡 Run 'python monitoring/dashboard.py' to verify protection status")


if __name__ == "__main__":
    main()
