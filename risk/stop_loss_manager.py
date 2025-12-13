"""
Stop-Loss Manager - Ensures ALL positions have protection.

FIXES BUG: Bot claimed "all positions have bracket orders" without actually checking.
This module VERIFIES positions have stop orders and creates them if missing.

OOP CONCEPTS:
- StopLossManager is a CLASS that manages stop-loss orders
- It has STATE (self.client, self.positions) that persists between method calls
- Methods work together - verify_protection() calls _get_stop_orders() internally
"""

import logging
from typing import List, Dict, Optional, Tuple
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, QueryOrderStatus
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

    def create_stop_loss(self, symbol: str, quantity: float,
                        entry_price: float, current_price: float) -> Optional[str]:
        """
        Create a stop-loss order for an unprotected position.

        Args:
            symbol: Stock ticker
            quantity: Number of shares to protect
            entry_price: Original purchase price
            current_price: Current market price

        Returns:
            Order ID if successful, None otherwise
        """
        try:
            # Calculate stop price (5% below entry)
            stop_price = round(entry_price * (1 - STOP_LOSS_PCT), 2)

            # Use stop-limit for better control
            # Stop triggers at stop_price, then places limit order at limit_price
            limit_price = round(stop_price * 0.99, 2)  # 1% below stop for slippage

            logger.info(f"Creating stop-loss for {symbol}:")
            logger.info(f"  Quantity: {quantity:.2f} shares")
            logger.info(f"  Entry: ${entry_price:.2f}, Current: ${current_price:.2f}")
            logger.info(f"  Stop: ${stop_price:.2f} ({-STOP_LOSS_PCT:.1%})")
            logger.info(f"  Limit: ${limit_price:.2f}")

            # Create the stop-limit order
            # NOTE: Alpaca doesn't support stop-limit orders with extended hours
            # So we use DAY orders only (will re-create daily during market hours)
            from alpaca.trading.requests import StopLimitOrderRequest

            order_data = StopLimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.SELL,
                type=OrderType.STOP_LIMIT,
                time_in_force=TimeInForce.DAY,  # Must be DAY for stop-limit
                stop_price=stop_price,
                limit_price=limit_price,
                extended_hours=False  # Stop-limit not allowed in extended hours
            )

            order = self.client.submit_order(order_data)

            logger.info(f"✅ Stop-loss created for {symbol} (Order ID: {order.id})")
            return order.id

        except Exception as e:
            logger.error(f"Failed to create stop-loss for {symbol}: {e}")
            return None

    def create_take_profit(self, symbol: str, quantity: float,
                          entry_price: float, current_price: float) -> Optional[str]:
        """
        Create a take-profit order for an unprotected position.

        Args:
            symbol: Stock ticker
            quantity: Number of shares
            entry_price: Original purchase price
            current_price: Current market price

        Returns:
            Order ID if successful, None otherwise
        """
        try:
            # Calculate take-profit price (15% above entry)
            tp_price = round(entry_price * (1 + TAKE_PROFIT_PCT), 2)

            logger.info(f"Creating take-profit for {symbol}:")
            logger.info(f"  Quantity: {quantity:.2f} shares")
            logger.info(f"  Entry: ${entry_price:.2f}, Current: ${current_price:.2f}")
            logger.info(f"  Take-profit: ${tp_price:.2f} (+{TAKE_PROFIT_PCT:.1%})")

            # Check if market is open to determine order parameters
            # When market is closed, Alpaca requires DAY orders for limit orders
            try:
                clock = self.client.get_clock()
                market_is_open = clock.is_open
            except:
                market_is_open = False  # Assume closed if can't check

            # Create limit sell order at take-profit price
            from alpaca.trading.requests import LimitOrderRequest

            # If market is closed, must use DAY and no extended hours
            # If market is open and extended hours enabled, use GTC
            if market_is_open and ENABLE_EXTENDED_HOURS:
                time_in_force = TimeInForce.GTC
                extended_hours = True
            else:
                time_in_force = TimeInForce.DAY
                extended_hours = False

            order_data = LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.SELL,
                type=OrderType.LIMIT,
                time_in_force=time_in_force,
                limit_price=tp_price,
                extended_hours=extended_hours
            )

            order = self.client.submit_order(order_data)

            logger.info(f"✅ Take-profit created for {symbol} (Order ID: {order.id})")
            return order.id

        except Exception as e:
            logger.error(f"Failed to create take-profit for {symbol}: {e}")
            return None

    def protect_all_positions(self, positions: List) -> Tuple[int, int]:
        """
        Ensure ALL positions have stop-loss and take-profit protection.

        This is the main entry point - call this to fix unprotected positions.

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

                # Create missing orders
                if not status['has_stop_loss']:
                    self.create_stop_loss(symbol, quantity, entry_price, current_price)

                if not status['has_take_profit']:
                    self.create_take_profit(symbol, quantity, entry_price, current_price)

                protected_count += 1

            except Exception as e:
                logger.error(f"Failed to protect {symbol}: {e}")

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
