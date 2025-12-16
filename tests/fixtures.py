"""
Test Fixtures for Trading Bot Unit Tests

FIXED (Problem 15): Reusable test data for all unit tests.

Provides:
- Sample market data (bars, prices)
- Sample orders
- Sample positions
- Mock API responses
"""

from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd


class SampleData:
    """Sample data for testing."""

    @staticmethod
    def get_sample_bars(symbol: str = "AAPL", days: int = 30) -> List[Dict]:
        """
        Get sample price bars for testing.

        Args:
            symbol: Stock symbol
            days: Number of days of data

        Returns:
            List of bar dictionaries
        """
        bars = []
        base_price = 150.0
        base_date = datetime.now() - timedelta(days=days)

        for i in range(days):
            # Simulate realistic price movement
            price = base_price + (i * 0.5) + ((-1) ** i * 2)  # Slight uptrend with volatility

            bar = {
                'symbol': symbol,
                'timestamp': base_date + timedelta(days=i),
                'open': price - 1,
                'high': price + 2,
                'low': price - 2,
                'close': price,
                'volume': 1000000 + (i * 10000)
            }
            bars.append(bar)

        return bars

    @staticmethod
    def get_sample_position(symbol: str = "AAPL", qty: int = 100, avg_price: float = 150.0):
        """Get sample position for testing."""
        return {
            'symbol': symbol,
            'qty': qty,
            'side': 'long' if qty > 0 else 'short',
            'market_value': qty * 155.0,  # Current market value
            'avg_entry_price': avg_price,
            'unrealized_pl': qty * (155.0 - avg_price),
            'unrealized_plpc': ((155.0 - avg_price) / avg_price) * 100
        }

    @staticmethod
    def get_sample_account(equity: float = 100000.0, cash: float = 50000.0):
        """Get sample account data for testing."""
        return {
            'equity': equity,
            'cash': cash,
            'buying_power': cash * 2,  # 2x margin
            'portfolio_value': equity,
            'long_market_value': equity - cash,
            'short_market_value': 0.0,
            'initial_margin': (equity - cash) * 0.5,
            'maintenance_margin': (equity - cash) * 0.25
        }

    @staticmethod
    def get_sample_order(symbol: str = "AAPL", qty: int = 10, side: str = "buy"):
        """Get sample order for testing."""
        return {
            'id': 'order_12345',
            'symbol': symbol,
            'qty': qty,
            'side': side,
            'type': 'market',
            'time_in_force': 'day',
            'status': 'filled',
            'filled_qty': qty,
            'filled_avg_price': 150.0,
            'submitted_at': datetime.now(),
            'filled_at': datetime.now()
        }


class MockResponses:
    """Mock API responses for testing."""

    @staticmethod
    def mock_bars_response(symbol: str = "AAPL", days: int = 30):
        """Mock response for get_stock_bars."""
        bars = SampleData.get_sample_bars(symbol, days)

        # Convert to mock Bar objects
        class MockBar:
            def __init__(self, data):
                self.symbol = data['symbol']
                self.timestamp = data['timestamp']
                self.open = data['open']
                self.high = data['high']
                self.low = data['low']
                self.close = data['close']
                self.volume = data['volume']

        return [MockBar(bar) for bar in bars]

    @staticmethod
    def mock_account_response():
        """Mock response for get_account."""
        account_data = SampleData.get_sample_account()

        class MockAccount:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
                self.status = 'ACTIVE'
                self.account_number = 'TEST123'

        return MockAccount(account_data)

    @staticmethod
    def mock_position_response(symbol: str = "AAPL"):
        """Mock response for get_position."""
        pos_data = SampleData.get_sample_position(symbol)

        class MockPosition:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)

        return MockPosition(pos_data)

    @staticmethod
    def mock_order_response(symbol: str = "AAPL"):
        """Mock response for submit_order."""
        order_data = SampleData.get_sample_order(symbol)

        class MockOrder:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)

        return MockOrder(order_data)


# Constants for testing
TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
TEST_PORTFOLIO_VALUE = 100000.0
TEST_CASH = 50000.0
TEST_MAX_POSITION_SIZE = 5000.0
TEST_STOP_LOSS_PCT = 0.05
TEST_TAKE_PROFIT_PCT = 0.15
