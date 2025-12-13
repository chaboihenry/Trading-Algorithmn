"""
Reliable market data fetching using Alpaca API directly.

WHY THIS EXISTS:
Lumibot's get_cash() method sometimes returns None, which breaks the bot.
This module bypasses Lumibot and talks to Alpaca directly for reliable data.

OOP CONCEPTS:
- This is a CLASS that encapsulates (groups together) all market data operations
- The class maintains a connection to Alpaca (self.client)
- Methods (functions in the class) can access this connection via 'self'
"""

import logging
from typing import Optional, Dict, List
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass
from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_PAPER

logger = logging.getLogger(__name__)


class MarketDataClient:
    """
    Fetches portfolio and market data directly from Alpaca.

    This is a CLASS - a blueprint for creating market data objects.
    When you create an instance of this class, it connects to Alpaca.

    Example usage:
        market_data = MarketDataClient()  # Create instance
        cash = market_data.get_cash()     # Call method
        positions = market_data.get_positions()
    """

    def __init__(self):
        """
        Constructor - runs when you create a new MarketDataClient instance.

        'self' refers to the specific instance being created.
        We store the Alpaca client in self.client so all methods can use it.
        """
        try:
            self.client = TradingClient(
                ALPACA_API_KEY,
                ALPACA_API_SECRET,
                paper=ALPACA_PAPER
            )
            logger.info("✅ Market data client connected to Alpaca")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            raise

    def get_cash(self) -> float:
        """
        Get available cash balance.

        Returns:
            float: Cash available for trading (NOT buying power, which includes margin)

        This fixes the bug where Lumibot's get_cash() returned None!
        """
        try:
            account = self.client.get_account()

            # Alpaca returns cash as a string, convert to float
            cash = float(account.cash)

            logger.debug(f"Cash: ${cash:,.2f}")
            return cash

        except Exception as e:
            logger.error(f"Error getting cash balance: {e}")
            return 0.0  # Return 0 instead of None to prevent crashes

    def get_buying_power(self) -> float:
        """
        Get buying power (cash + margin).

        For paper trading, this is usually 2x your cash (Alpaca gives 2:1 margin).
        For real money, we'll use cash only to avoid margin calls.
        """
        try:
            account = self.client.get_account()
            buying_power = float(account.buying_power)

            logger.debug(f"Buying power: ${buying_power:,.2f}")
            return buying_power

        except Exception as e:
            logger.error(f"Error getting buying power: {e}")
            return 0.0

    def get_portfolio_value(self) -> float:
        """
        Get total portfolio value (cash + positions).

        Returns:
            float: Total account equity
        """
        try:
            account = self.client.get_account()
            portfolio_value = float(account.equity)

            logger.debug(f"Portfolio value: ${portfolio_value:,.2f}")
            return portfolio_value

        except Exception as e:
            logger.error(f"Error getting portfolio value: {e}")
            return 0.0

    def get_positions(self) -> List:
        """
        Get all open positions.

        Returns:
            List of Position objects from Alpaca
        """
        try:
            positions = self.client.get_all_positions()
            logger.debug(f"Found {len(positions)} positions")
            return positions

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[object]:
        """
        Get position for a specific symbol.

        Args:
            symbol: Stock ticker (e.g., "AAPL")

        Returns:
            Position object if exists, None otherwise
        """
        try:
            position = self.client.get_open_position(symbol)
            return position
        except Exception as e:
            # Position doesn't exist - not an error
            logger.debug(f"No position for {symbol}")
            return None

    def get_account_info(self) -> Dict[str, any]:
        """
        Get comprehensive account information.

        Returns:
            Dict with all account details for debugging
        """
        try:
            account = self.client.get_account()

            return {
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.equity),
                'day_trade_count': int(account.daytrade_count),
                'pattern_day_trader': account.pattern_day_trader,
                'account_blocked': account.account_blocked,
                'trade_suspended': getattr(account, 'trade_suspended', False),
                'trading_blocked': getattr(account, 'trading_blocked', False)
            }

        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}

    def verify_can_trade(self) -> bool:
        """
        Verify account is allowed to trade.

        Returns:
            bool: True if account can trade, False otherwise
        """
        try:
            account = self.client.get_account()

            if account.account_blocked:
                logger.error("🚫 Account is blocked!")
                return False

            if getattr(account, 'trade_suspended', False):
                logger.error("🚫 Trading is suspended!")
                return False

            if getattr(account, 'trading_blocked', False):
                logger.error("🚫 Trading is blocked!")
                return False

            logger.info("✅ Account verified - ready to trade")
            return True

        except Exception as e:
            logger.error(f"Error verifying account: {e}")
            return False


# =============================================================================
# HELPER FUNCTIONS (not part of class)
# =============================================================================

def get_latest_price(client: MarketDataClient, symbol: str) -> Optional[float]:
    """
    Get the latest price for a symbol.

    This is a standalone FUNCTION (not a method in the class).
    You can call it without creating a MarketDataClient instance.

    Args:
        client: MarketDataClient instance
        symbol: Stock ticker

    Returns:
        Latest price or None
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest

        data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
        request = StockLatestTradeRequest(symbol_or_symbols=symbol)
        trade = data_client.get_stock_latest_trade(request)

        if symbol in trade:
            price = float(trade[symbol].price)
            logger.debug(f"{symbol}: ${price:.2f}")
            return price

        return None

    except Exception as e:
        logger.error(f"Error getting price for {symbol}: {e}")
        return None


# =============================================================================
# MODULE-LEVEL INSTANCE (Singleton pattern)
# =============================================================================

# Create one global instance that all code can share
# This is more efficient than creating multiple connections
_market_data_client = None

def get_market_data_client() -> MarketDataClient:
    """
    Get the global MarketDataClient instance (creates it if needed).

    This is the SINGLETON PATTERN - ensures only one instance exists.
    Multiple calls return the same instance instead of creating new ones.
    """
    global _market_data_client

    if _market_data_client is None:
        _market_data_client = MarketDataClient()

    return _market_data_client
