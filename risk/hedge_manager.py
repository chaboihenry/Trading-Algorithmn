"""
Hedge Manager - Profit when the market drops using inverse ETFs.

FIXES BUG: Hedge logic was in combined_strategy but never ran due to no cash.
This module makes hedging a standalone, testable component.

WHY HEDGING MATTERS:
When the market crashes -10%, a traditional portfolio loses $10K on $100K.
With inverse ETFs, you PROFIT from the crash, offsetting losses.

INVERSE ETFs EXPLAINED:
- Regular ETF: SPY tracks S&P 500. SPY up 1% when market up 1%.
- Inverse ETF: SH is -1x S&P 500. SH up 1% when market DOWN 1%.
- Leveraged inverse: SPXS is -3x S&P 500. SPXS up 3% when market down 1%.

OOP CONCEPTS:
- HedgeManager is a CLASS with methods for detecting bearish markets
- It maintains STATE (self.current_hedge_value) between calls
- Uses COMPOSITION - contains a MarketDataClient instance
"""

import logging
from typing import Dict, List, Optional, Tuple
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from data.market_data import get_market_data_client, get_latest_price
from config.settings import (
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    ALPACA_PAPER,
    INVERSE_ETFS,
    DEFAULT_INVERSE_ETF,
    MAX_INVERSE_ALLOCATION,
    MARKET_SENTIMENT_SYMBOLS,
    BEARISH_MARKET_THRESHOLD,
    RSI_OVERBOUGHT,
    ENABLE_EXTENDED_HOURS
)

logger = logging.getLogger(__name__)


class HedgeManager:
    """
    Manages inverse ETF positions to profit from market downturns.

    STRATEGY:
    1. Sample major stocks (AAPL, MSFT, GOOGL, NVDA, TSLA)
    2. Calculate % that are overbought (RSI > 70)
    3. If ≥60% overbought → market is topping out → hedge
    4. Buy inverse ETF (SH) to profit when market crashes
    5. Exit hedge when market recovers
    """

    def __init__(self):
        """Initialize hedge manager with Alpaca connection."""
        self.trading_client = TradingClient(
            ALPACA_API_KEY,
            ALPACA_API_SECRET,
            paper=ALPACA_PAPER
        )
        self.market_data = get_market_data_client()
        logger.info("✅ Hedge manager initialized")

    def _get_rsi_from_db(self, symbol: str) -> Optional[float]:
        """
        Get RSI for a symbol from the database.

        Args:
            symbol: Stock ticker

        Returns:
            RSI value (0-100) or None if not available
        """
        try:
            import sqlite3
            from config.settings import DB_PATH

            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            query = """
                SELECT rsi_14
                FROM technical_indicators
                WHERE symbol_ticker = ?
                ORDER BY indicator_date DESC
                LIMIT 1
            """

            cursor.execute(query, (symbol,))
            result = cursor.fetchone()
            conn.close()

            if result and result[0] is not None:
                rsi = float(result[0])
                logger.debug(f"{symbol} RSI: {rsi:.1f}")
                return rsi

            return None

        except Exception as e:
            logger.error(f"Error getting RSI for {symbol}: {e}")
            return None

    def check_market_sentiment(self) -> Dict[str, any]:
        """
        Analyze overall market sentiment by sampling major stocks.

        Returns:
            Dict with sentiment analysis:
            {
                'bearish_count': int,
                'total_checked': int,
                'bearish_ratio': float,
                'is_bearish': bool,
                'symbols_overbought': List[str]
            }
        """
        bearish_count = 0
        total_checked = 0
        symbols_overbought = []

        logger.info("Checking market sentiment...")

        for symbol in MARKET_SENTIMENT_SYMBOLS:
            rsi = self._get_rsi_from_db(symbol)

            if rsi is not None:
                total_checked += 1

                if rsi > RSI_OVERBOUGHT:
                    bearish_count += 1
                    symbols_overbought.append(symbol)
                    logger.info(f"  📊 {symbol}: RSI {rsi:.1f} (overbought)")
                else:
                    logger.info(f"  📊 {symbol}: RSI {rsi:.1f}")

        if total_checked == 0:
            logger.warning("⚠️  No RSI data available for sentiment analysis")
            return {
                'bearish_count': 0,
                'total_checked': 0,
                'bearish_ratio': 0.0,
                'is_bearish': False,
                'symbols_overbought': []
            }

        bearish_ratio = bearish_count / total_checked
        is_bearish = bearish_ratio >= BEARISH_MARKET_THRESHOLD

        logger.info(f"Market sentiment: {bearish_count}/{total_checked} overbought ({bearish_ratio:.1%})")

        if is_bearish:
            logger.warning(f"🔴 BEARISH MARKET DETECTED ({bearish_ratio:.1%} overbought)")
        else:
            logger.info(f"✅ Market is healthy ({bearish_ratio:.1%} overbought)")

        return {
            'bearish_count': bearish_count,
            'total_checked': total_checked,
            'bearish_ratio': bearish_ratio,
            'is_bearish': is_bearish,
            'symbols_overbought': symbols_overbought
        }

    def get_current_hedge_allocation(self) -> Tuple[float, float]:
        """
        Calculate current allocation to inverse ETFs.

        Returns:
            Tuple of (hedge_value, hedge_pct)
        """
        try:
            positions = self.market_data.get_positions()
            portfolio_value = self.market_data.get_portfolio_value()

            hedge_value = 0.0

            for position in positions:
                if position.symbol in INVERSE_ETFS.values():
                    pos_value = float(position.qty) * float(position.current_price)
                    hedge_value += pos_value
                    logger.debug(f"  Hedge: {position.symbol} = ${pos_value:,.2f}")

            hedge_pct = hedge_value / portfolio_value if portfolio_value > 0 else 0.0

            logger.info(f"Current hedge allocation: ${hedge_value:,.2f} ({hedge_pct:.1%})")
            return hedge_value, hedge_pct

        except Exception as e:
            logger.error(f"Error calculating hedge allocation: {e}")
            return 0.0, 0.0

    def create_hedge(self, cash_available: float, portfolio_value: float) -> bool:
        """
        Create a hedge position using inverse ETF.

        Args:
            cash_available: Cash available for trading
            portfolio_value: Total portfolio value

        Returns:
            True if hedge created successfully
        """
        try:
            # Get current hedge allocation
            hedge_value, hedge_pct = self.get_current_hedge_allocation()

            # Check if we're already at max allocation
            if hedge_pct >= MAX_INVERSE_ALLOCATION:
                logger.info(f"Already at max hedge allocation ({hedge_pct:.1%})")
                return False

            # Calculate how much more we can allocate
            max_hedge_value = portfolio_value * MAX_INVERSE_ALLOCATION
            room_for_hedge = max_hedge_value - hedge_value

            # Allocate up to 10% of portfolio per hedge trade
            target_value = min(
                portfolio_value * 0.10,  # 10% of portfolio
                room_for_hedge,           # Remaining room
                cash_available * 0.5      # Max 50% of available cash
            )

            if target_value < 100:
                logger.warning("Not enough cash/room for hedge position")
                return False

            # Use default inverse ETF (SH = 1x inverse S&P 500)
            hedge_symbol = INVERSE_ETFS[DEFAULT_INVERSE_ETF]

            # Get current price
            price = get_latest_price(self.market_data, hedge_symbol)
            if not price:
                logger.error(f"Could not get price for {hedge_symbol}")
                return False

            # Calculate quantity
            quantity = int(target_value / price)

            if quantity < 1:
                logger.warning(f"Quantity too small: {quantity}")
                return False

            logger.warning(f"🛡️  Creating hedge position:")
            logger.warning(f"  Symbol: {hedge_symbol}")
            logger.warning(f"  Quantity: {quantity} shares")
            logger.warning(f"  Price: ${price:.2f}")
            logger.warning(f"  Value: ${quantity * price:,.2f}")
            logger.warning(f"  This will PROFIT if market drops!")

            # Create market order to buy inverse ETF
            order_data = MarketOrderRequest(
                symbol=hedge_symbol,
                qty=quantity,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )

            order = self.trading_client.submit_order(order_data)

            logger.info(f"✅ Hedge order submitted (Order ID: {order.id})")
            return True

        except Exception as e:
            logger.error(f"Failed to create hedge: {e}")
            return False

    def exit_hedges(self) -> bool:
        """
        Exit all inverse ETF positions (market is bullish again).

        Returns:
            True if all hedges exited successfully
        """
        try:
            positions = self.market_data.get_positions()
            hedge_positions = [p for p in positions if p.symbol in INVERSE_ETFS.values()]

            if not hedge_positions:
                logger.info("No hedge positions to exit")
                return True

            logger.info(f"📈 Market bullish - exiting {len(hedge_positions)} hedge position(s)")

            success = True
            for position in hedge_positions:
                try:
                    symbol = position.symbol
                    quantity = float(position.qty)

                    logger.info(f"Exiting hedge: {symbol} ({quantity:.2f} shares)")

                    # Create market sell order
                    order_data = MarketOrderRequest(
                        symbol=symbol,
                        qty=quantity,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )

                    order = self.trading_client.submit_order(order_data)
                    logger.info(f"✅ Hedge exit order submitted (Order ID: {order.id})")

                except Exception as e:
                    logger.error(f"Failed to exit hedge {symbol}: {e}")
                    success = False

            return success

        except Exception as e:
            logger.error(f"Error exiting hedges: {e}")
            return False

    def manage_hedges(self, cash_available: float, portfolio_value: float) -> None:
        """
        Main entry point - check market and manage hedges accordingly.

        Call this every trading iteration to maintain appropriate hedging.

        Args:
            cash_available: Cash available for trading
            portfolio_value: Total portfolio value
        """
        logger.info("=" * 60)
        logger.info("HEDGE MANAGEMENT")
        logger.info("=" * 60)

        # Check market sentiment
        sentiment = self.check_market_sentiment()

        # Get current hedge status
        hedge_value, hedge_pct = self.get_current_hedge_allocation()

        # Decision logic
        if sentiment['is_bearish']:
            # Market is overbought - hedge if possible
            if hedge_pct < MAX_INVERSE_ALLOCATION:
                logger.warning(f"🔴 Bearish market - creating/increasing hedge")
                self.create_hedge(cash_available, portfolio_value)
            else:
                logger.info(f"Already fully hedged ({hedge_pct:.1%})")

        elif sentiment['bearish_ratio'] < 0.3:
            # Market is healthy - exit hedges if we have any
            if hedge_pct > 0:
                logger.info(f"📈 Market recovered - exiting hedges")
                self.exit_hedges()
            else:
                logger.info("No hedges, market healthy - no action needed")

        else:
            # Market is neutral - maintain current hedges
            logger.info(f"Market neutral ({sentiment['bearish_ratio']:.1%} overbought) - maintaining hedges")

        logger.info("=" * 60)


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def check_and_hedge():
    """
    One-line function to manage hedges.

    Call this at each trading iteration.
    """
    try:
        manager = HedgeManager()

        # Get portfolio info
        market_data = get_market_data_client()
        cash = market_data.get_cash()
        portfolio_value = market_data.get_portfolio_value()

        # Manage hedges
        manager.manage_hedges(cash, portfolio_value)

    except Exception as e:
        logger.error(f"Error in hedge management: {e}")
