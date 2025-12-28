"""Stop-Loss Manager - Protects positions with stop-loss and take-profit orders."""

import logging
from typing import List, Dict, Optional, Tuple
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOrdersRequest, LimitOrderRequest, StopOrderRequest,
    TakeProfitRequest, StopLossRequest
)
from alpaca.trading.enums import (
    OrderSide, OrderType, TimeInForce, QueryOrderStatus, OrderClass
)
from backup.settings import (
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    ALPACA_PAPER,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    ENABLE_EXTENDED_HOURS
)

logger = logging.getLogger(__name__)


class StopLossManager:
    """Manages stop-loss and take-profit orders for all positions."""

    def __init__(self):
        self.client = TradingClient(
            ALPACA_API_KEY,
            ALPACA_API_SECRET,
            paper=ALPACA_PAPER
        )
        logger.info("✅ Stop-loss manager initialized")

    def _get_stop_orders(self) -> List:
        """Get all open stop-loss orders."""
        try:
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100)
            orders = self.client.get_orders(request)
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
        """Get all open take-profit (limit sell) orders."""
        try:
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100)
            orders = self.client.get_orders(request)
            tp_orders = [
                order for order in orders
                if order.type == OrderType.LIMIT and order.side == OrderSide.SELL
            ]
            logger.debug(f"Found {len(tp_orders)} take-profit orders")
            return tp_orders
        except Exception as e:
            logger.error(f"Error getting take-profit orders: {e}")
            return []

    def _cancel_all_orders_for_symbol(self, symbol: str) -> None:
        """Cancel all open orders for a symbol to free held shares.

        When Alpaca holds shares for existing orders, you can't create new orders
        for the same shares. This method cancels ALL orders for a symbol, then waits
        0.5 seconds for Alpaca to release the shares.

        Args:
            symbol: Stock ticker to cancel orders for
        """
        try:
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100)
            orders = self.client.get_orders(request)

            symbol_orders = [order for order in orders if order.symbol == symbol]

            if symbol_orders:
                logger.info(f"Canceling {len(symbol_orders)} existing orders for {symbol}")
                for order in symbol_orders:
                    try:
                        self.client.cancel_order_by_id(order.id)
                    except Exception as e:
                        logger.warning(f"Could not cancel order {order.id}: {e}")

                # Wait for Alpaca to release shares
                import time
                time.sleep(0.5)
                logger.debug(f"Shares released for {symbol}")

        except Exception as e:
            logger.error(f"Error canceling orders for {symbol}: {e}")

    def verify_protection(self, positions: List) -> Dict[str, Dict]:
        """Verify which positions have stop-loss and take-profit protection."""
        stop_orders = self._get_stop_orders()
        tp_orders = self._get_take_profit_orders()

        stops_by_symbol = {order.symbol: order for order in stop_orders}
        tp_by_symbol = {order.symbol: order for order in tp_orders}

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

            if has_stop and has_tp:
                logger.info(f"✅ {symbol}: Fully protected")
            elif has_stop:
                logger.warning(f"⚠️  {symbol}: Has stop-loss but NO take-profit")
            elif has_tp:
                logger.warning(f"⚠️  {symbol}: Has take-profit but NO stop-loss")
            else:
                logger.error(f"🚫 {symbol}: NO PROTECTION")

        return protection_status

    def create_oco_protection(self, symbol: str, quantity: float,
                              entry_price: float, current_price: float) -> Optional[str]:
        """Create stop-loss protection for a position.

        CRITICAL FIX: Alpaca holds shares when orders exist, blocking new orders.
        Solution: Cancel ALL existing orders first, wait for shares to release.

        For fractional shares: Creates STOP-LOSS ONLY (Alpaca can't handle take-profit on fractional)
        For whole shares: Creates OCO order (both stop-loss and take-profit)
        """
        try:
            # STEP 1: Cancel all existing orders to free shares
            self._cancel_all_orders_for_symbol(symbol)

            stop_price = round(entry_price * (1 - STOP_LOSS_PCT), 2)
            tp_price = round(entry_price * (1 + TAKE_PROFIT_PCT), 2)

            logger.info(f"Creating protection for {symbol}:")
            logger.info(f"  Quantity: {quantity:.4f} shares")
            logger.info(f"  Stop-loss: ${stop_price:.2f} ({-STOP_LOSS_PCT:.1%})")

            is_fractional = (quantity % 1) != 0

            if is_fractional:
                # Fractional shares: STOP-LOSS ONLY
                # Alpaca limitation: Can't hold fractional shares with multiple orders
                logger.info(f"  Fractional quantity - creating stop-loss only")

                stop_order = StopOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    stop_price=stop_price
                )
                stop_result = self.client.submit_order(order_data=stop_order)

                logger.info(f"✅ Stop-loss created (ID: {stop_result.id})")
                logger.info(f"   Note: No take-profit for fractional shares (Alpaca limitation)")
                return str(stop_result.id)

            else:
                # Whole shares: OCO order (both stop-loss and take-profit)
                try:
                    clock = self.client.get_clock()
                    market_is_open = clock.is_open
                except:
                    market_is_open = False

                time_in_force = TimeInForce.GTC if (market_is_open and ENABLE_EXTENDED_HOURS) else TimeInForce.DAY

                logger.info(f"  Take-profit: ${tp_price:.2f} (+{TAKE_PROFIT_PCT:.1%})")

                oco_order = LimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=OrderSide.SELL,
                    time_in_force=time_in_force,
                    limit_price=tp_price,
                    order_class=OrderClass.OCO,
                    take_profit=TakeProfitRequest(limit_price=tp_price),
                    stop_loss=StopLossRequest(stop_price=stop_price)
                )

                result = self.client.submit_order(order_data=oco_order)
                logger.info(f"✅ OCO protection created (ID: {result.id})")
                return str(result.id)

        except Exception as e:
            logger.error(f"Failed to create protection for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def protect_all_positions(self, positions: List) -> Tuple[int, int]:
        """Ensure all positions have stop-loss and take-profit protection."""
        if not positions:
            logger.info("No positions to protect")
            return 0, 0

        logger.info(f"Protecting {len(positions)} positions...")
        protection_status = self.verify_protection(positions)
        protected_count = 0

        for position in positions:
            symbol = position.symbol
            status = protection_status[symbol]

            if status['fully_protected']:
                protected_count += 1
                continue

            logger.warning(f"🔧 Protecting: {symbol}")

            try:
                quantity = float(position.qty)
                entry_price = float(position.avg_entry_price)
                current_price = float(position.current_price)

                # create_oco_protection() handles canceling existing orders
                order_id = self.create_oco_protection(
                    symbol, quantity, entry_price, current_price
                )

                if order_id:
                    protected_count += 1
                else:
                    logger.error(f"❌ Failed to protect {symbol}")

            except Exception as e:
                logger.error(f"Failed to protect {symbol}: {e}")

        logger.info(f"Protection: {protected_count}/{len(positions)} protected")
        return protected_count, len(positions)


def ensure_all_positions_protected():
    """Protect all open positions with stop-loss and take-profit orders."""
    try:
        manager = StopLossManager()
        from backup.market_data import get_market_data_client
        market_data = get_market_data_client()
        positions = market_data.get_positions()
        protected, total = manager.protect_all_positions(positions)

        if protected < total:
            logger.warning(f"⚠️  Only {protected}/{total} positions protected")
        else:
            logger.info(f"✅ All {total} positions protected")

        return protected == total
    except Exception as e:
        logger.error(f"Error in position protection: {e}")
        return False
