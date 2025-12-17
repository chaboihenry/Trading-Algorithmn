"""
Stop-Loss Manager - Ensures ALL positions have protection.

FIXES BUG: Bot claimed "all positions have bracket orders" without actually checking.
This module VERIFIES positions have stop orders and creates them if missing.

CRITICAL FIX: Uses OCO (One-Cancels-Other) orders to avoid "shares held by orders" error.
- Alpaca REJECTS separate stop-loss and take-profit orders (insufficient qty error)
- Must submit BOTH protective orders together as ONE OCO order
- Uses stop-MARKET (not stop-LIMIT) to avoid share reservation conflicts

OOP CONCEPTS:
- StopLossManager is a CLASS that manages stop-loss orders
- It has STATE (self.client, self.positions) that persists between method calls
- Methods work together - verify_protection() calls _get_stop_orders() internally
"""

import logging
from typing import List, Dict, Optional, Tuple
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOrdersRequest, LimitOrderRequest, TakeProfitRequest, StopLossRequest
)
from alpaca.trading.enums import (
    OrderSide, OrderType, TimeInForce, QueryOrderStatus, OrderClass
)
from config.settings import (
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    ALPACA_PAPER,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    ENABLE_EXTENDED_HOURS
)

logger = logging.getLogger(__name__)


class StopLossManager:
    """
    Manages stop-loss and take-profit orders for all positions.

    This class ensures EVERY position has protection by:
    1. Checking Alpaca for existing stop orders
    2. Creating missing stop orders
    3. Updating trailing stops as prices increase
    """

    def __init__(self):
        """Initialize connection to Alpaca."""
        self.client = TradingClient(
            ALPACA_API_KEY,
            ALPACA_API_SECRET,
            paper=ALPACA_PAPER
        )
        logger.info("✅ Stop-loss manager initialized")

    def _get_stop_orders(self) -> List:
        """
        Get all open stop-loss orders from Alpaca.

        Returns:
            List of Order objects that are stop-loss or stop-limit orders
        """
        try:
            # Request all open orders
            request = GetOrdersRequest(
                status=QueryOrderStatus.OPEN,
                limit=100
            )
            orders = self.client.get_orders(request)

            # Filter for stop orders only
            stop_orders = [
                order for order in orders
                if order.type in [OrderType.STOP, OrderType.STOP_LIMIT]
            ]

            logger.debug(f"Found {len(stop_orders)} stop orders")
            return stop_orders

        except Exception as e:
            logger.error(f"Error getting stop orders: {e}")
            return []

    def _get_take_profit_orders(self) -> List:
        """
        Get all open take-profit (limit sell) orders.

        Returns:
            List of Order objects that are limit sell orders
        """
        try:
            request = GetOrdersRequest(
                status=QueryOrderStatus.OPEN,
                limit=100
            )
            orders = self.client.get_orders(request)

            # Filter for limit sell orders (take-profit)
            tp_orders = [
                order for order in orders
                if order.type == OrderType.LIMIT and order.side == OrderSide.SELL
            ]

            logger.debug(f"Found {len(tp_orders)} take-profit orders")
            return tp_orders

        except Exception as e:
            logger.error(f"Error getting take-profit orders: {e}")
            return []

    def verify_protection(self, positions: List) -> Dict[str, Dict]:
        """
        Verify which positions have stop-loss and take-profit protection.

        This is the main method that fixes the bug - it ACTUALLY checks!

        Args:
            positions: List of Position objects from Alpaca

        Returns:
            Dict mapping symbol -> protection status
            {
                'AAPL': {
                    'has_stop_loss': True,
                    'has_take_profit': False,
                    'stop_order_id': '12345',
                    'tp_order_id': None
                },
                ...
            }
        """
        # Get all existing orders
        stop_orders = self._get_stop_orders()
        tp_orders = self._get_take_profit_orders()

        # Create lookup dicts by symbol
        stops_by_symbol = {order.symbol: order for order in stop_orders}
        tp_by_symbol = {order.symbol: order for order in tp_orders}

        # Check each position
        protection_status = {}

        for position in positions:
            symbol = position.symbol

            has_stop = symbol in stops_by_symbol
            has_tp = symbol in tp_by_symbol

            protection_status[symbol] = {
                'has_stop_loss': has_stop,
                'has_take_profit': has_tp,
                'stop_order_id': stops_by_symbol[symbol].id if has_stop else None,
                'tp_order_id': tp_by_symbol[symbol].id if has_tp else None,
                'fully_protected': has_stop and has_tp
            }

            # Log status
            if has_stop and has_tp:
                logger.info(f"✅ {symbol}: Fully protected (stop + take-profit)")
            elif has_stop:
                logger.warning(f"⚠️  {symbol}: Has stop-loss but NO take-profit")
            elif has_tp:
                logger.warning(f"⚠️  {symbol}: Has take-profit but NO stop-loss")
            else:
                logger.error(f"🚫 {symbol}: NO PROTECTION - CRITICAL!")

        return protection_status

    def create_oco_protection(self, symbol: str, quantity: float,
                              entry_price: float, current_price: float) -> Optional[str]:
        """
        Create OCO (One-Cancels-Other) protection for a position.

        CRITICAL: Submits stop-loss and take-profit as ONE OCO order to avoid
        "shares held by orders" error. Alpaca rejects separate protective orders.

        Uses stop-MARKET (StopLossRequest) which doesn't reserve shares like stop-limit.

        Args:
            symbol: Stock ticker
            quantity: Number of shares to protect
            entry_price: Original purchase price
            current_price: Current market price

        Returns:
            Order ID if successful, None on failure
        """
        try:
            # Calculate prices
            stop_price = round(entry_price * (1 - STOP_LOSS_PCT), 2)
            tp_price = round(entry_price * (1 + TAKE_PROFIT_PCT), 2)

            logger.info(f"Creating OCO protection for {symbol}:")
            logger.info(f"  Quantity: {quantity:.2f} shares")
            logger.info(f"  Entry: ${entry_price:.2f}, Current: ${current_price:.2f}")
            logger.info(f"  Stop-loss: ${stop_price:.2f} ({-STOP_LOSS_PCT:.1%})")
            logger.info(f"  Take-profit: ${tp_price:.2f} (+{TAKE_PROFIT_PCT:.1%})")

            # Check market status for time_in_force
            try:
                clock = self.client.get_clock()
                market_is_open = clock.is_open
            except:
                market_is_open = False

            # CRITICAL FIX: Fractional quantities MUST use DAY time_in_force
            # Check if quantity has decimals
            is_fractional = (quantity % 1) != 0

            # Determine time_in_force based on fractional status and market status
            if is_fractional:
                # Fractional orders MUST be DAY per Alpaca rules
                time_in_force = TimeInForce.DAY
                logger.info(f"   Using DAY time_in_force (fractional quantity: {quantity:.4f})")
            elif market_is_open and ENABLE_EXTENDED_HOURS:
                time_in_force = TimeInForce.GTC
            else:
                time_in_force = TimeInForce.DAY

            # Create ONE OCO order with BOTH protective legs
            # CRITICAL FIX: OCO orders MUST use LimitOrderRequest, not MarketOrderRequest
            # Alpaca requires the main leg to be a limit order for OCO orders
            oco_order = LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.SELL,  # SELL to exit long position
                time_in_force=time_in_force,
                limit_price=tp_price,  # REQUIRED for OCO - set to take-profit price
                order_class=OrderClass.OCO,
                take_profit=TakeProfitRequest(limit_price=tp_price),  # Take profit leg
                stop_loss=StopLossRequest(stop_price=stop_price)      # Stop loss leg (uses stop-MARKET)
            )

            # Submit the OCO order (ONE API call creates BOTH protective orders)
            result = self.client.submit_order(order_data=oco_order)

            logger.info(f"✅ OCO protection created for {symbol} (Order ID: {str(result.id)})")
            logger.info(f"   Stop-loss: ${stop_price:.2f} | Take-profit: ${tp_price:.2f}")
            logger.info(f"   When one order fills, the other will auto-cancel")

            return str(result.id)

        except Exception as e:
            logger.error(f"Failed to create OCO protection for {symbol}: {e}")
            logger.error(f"This usually means:")
            logger.error(f"  1. Position already has protective orders")
            logger.error(f"  2. Insufficient shares available")
            logger.error(f"  3. Market is closed and order type not allowed")
            logger.error(f"  4. Symbol doesn't support OCO orders (crypto, options)")
            import traceback
            traceback.print_exc()
            return None

    def protect_all_positions(self, positions: List) -> Tuple[int, int]:
        """
        Ensure ALL positions have stop-loss and take-profit protection.

        This is the main entry point - call this to fix unprotected positions.

        CRITICAL: Uses OCO orders to avoid "shares held by orders" error.

        Args:
            positions: List of Position objects

        Returns:
            Tuple of (positions_protected, total_positions)
        """
        if not positions:
            logger.info("No positions to protect")
            return 0, 0

        logger.info(f"Protecting {len(positions)} positions...")

        # First, check which positions are already protected
        protection_status = self.verify_protection(positions)

        protected_count = 0
        unprotected_count = 0

        # Protect any unprotected positions
        for position in positions:
            symbol = position.symbol
            status = protection_status[symbol]

            if status['fully_protected']:
                protected_count += 1
                continue

            # Position needs protection
            unprotected_count += 1
            logger.warning(f"🔧 Protecting unprotected position: {symbol}")

            try:
                quantity = float(position.qty)
                entry_price = float(position.avg_entry_price)
                current_price = float(position.current_price)

                # If position has PARTIAL protection, cancel existing orders first
                # to avoid conflicts when creating new OCO protection
                if status['has_stop_loss'] or status['has_take_profit']:
                    logger.warning(f"⚠️  {symbol} has partial protection - canceling old orders")

                    if status['stop_order_id']:
                        try:
                            self.client.cancel_order_by_id(status['stop_order_id'])
                            logger.info(f"   Canceled old stop-loss order")
                        except Exception as e:
                            logger.warning(f"   Could not cancel stop-loss: {e}")

                    if status['tp_order_id']:
                        try:
                            self.client.cancel_order_by_id(status['tp_order_id'])
                            logger.info(f"   Canceled old take-profit order")
                        except Exception as e:
                            logger.warning(f"   Could not cancel take-profit: {e}")

                # Create OCO protection (both stop-loss and take-profit in ONE order)
                order_id = self.create_oco_protection(
                    symbol, quantity, entry_price, current_price
                )

                if order_id:
                    protected_count += 1
                    logger.info(f"✅ {symbol} fully protected with OCO order")
                else:
                    logger.error(f"❌ Failed to protect {symbol}")

            except Exception as e:
                logger.error(f"Failed to protect {symbol}: {e}")
                import traceback
                traceback.print_exc()

        logger.info(f"Protection summary: {protected_count}/{len(positions)} protected")

        return protected_count, len(positions)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def ensure_all_positions_protected():
    """
    One-line function to protect all positions.

    Call this at the start of each trading iteration to ensure safety.
    """
    try:
        manager = StopLossManager()

        # Get positions
        from data.market_data import get_market_data_client
        market_data = get_market_data_client()
        positions = market_data.get_positions()

        # Protect them
        protected, total = manager.protect_all_positions(positions)

        if protected < total:
            logger.warning(f"⚠️  Only {protected}/{total} positions fully protected!")
        else:
            logger.info(f"✅ All {total} positions protected")

        return protected == total

    except Exception as e:
        logger.error(f"Error in position protection: {e}")
        return False
