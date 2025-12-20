"""
Custom Exception Classes for Trading Bot

FIXED (Problem 14): Trading-specific exception hierarchy for better error handling.

Exception Hierarchy:
    TradingException (base)
    ├── MarketDataError (market data fetching issues)
    ├── OrderExecutionError (order placement/management issues)
    ├── RiskManagementError (risk limit violations)
    ├── StrategyError (strategy execution issues)
    └── ConfigurationError (configuration/setup issues)

Usage:
    from utils.exceptions import OrderExecutionError

    if order_failed:
        raise OrderExecutionError(
            f"Failed to place order for {symbol}",
            symbol=symbol,
            order_id=order_id,
            reason=failure_reason
        )
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class TradingException(Exception):
    """
    Base exception for all trading-specific errors.

    FIXED (Problem 14): Custom exception with context logging.

    Attributes:
        message: Human-readable error message
        context: Additional context (symbol, order_id, etc.)
    """

    def __init__(self, message: str, **context):
        """
        Initialize trading exception.

        Args:
            message: Error message
            **context: Additional context (logged for debugging)
        """
        super().__init__(message)
        self.message = message
        self.context = context

        # Log exception with full context
        logger.error(f"{self.__class__.__name__}: {message}")
        if context:
            logger.error(f"  Context: {context}")

    def __str__(self):
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} ({ctx_str})"
        return self.message


class MarketDataError(TradingException):
    """
    Exception for market data fetching or processing errors.

    Use when:
    - API calls to data provider fail
    - Data parsing fails
    - Data quality issues (missing bars, bad prices)

    Example:
        raise MarketDataError(
            "Failed to fetch bars for AAPL",
            symbol="AAPL",
            days=7,
            error=str(original_error)
        )
    """
    pass


class OrderExecutionError(TradingException):
    """
    Exception for order placement or execution errors.

    Use when:
    - Order placement fails
    - Order cancellation fails
    - Order status unclear
    - Insufficient buying power

    Example:
        raise OrderExecutionError(
            "Insufficient buying power for order",
            symbol="TSLA",
            quantity=100,
            required_cash=50000,
            available_cash=45000
        )
    """
    pass


class RiskManagementError(TradingException):
    """
    Exception for risk limit violations.

    Use when:
    - Position size exceeds limits
    - Daily loss limit hit
    - Too many positions open
    - Risk checks fail

    Example:
        raise RiskManagementError(
            "Daily loss limit exceeded",
            current_loss=-3.2,
            max_loss=-3.0,
            action="blocking_all_trades"
        )
    """
    pass


class StrategyError(TradingException):
    """
    Exception for strategy execution errors.

    Use when:
    - Strategy logic fails
    - Signal generation fails
    - Model prediction fails
    - Invalid strategy state

    Example:
        raise StrategyError(
            "Meta-model prediction failed",
            model_type="KerasMLP",
            feature_count=40,
            error=str(original_error)
        )
    """
    pass


class ConfigurationError(TradingException):
    """
    Exception for configuration or setup errors.

    Use when:
    - Missing API keys
    - Invalid configuration values
    - Database connection fails
    - Required files missing

    Example:
        raise ConfigurationError(
            "Database file not found",
            db_path="/path/to/db.sqlite",
            action="check_config"
        )
    """
    pass


# Exception handling decorators for common patterns

def handle_market_data_errors(func):
    """
    Decorator to handle market data errors gracefully.

    FIXED (Problem 14): Standardized error handling for market data calls.

    Converts API errors to MarketDataError with full context.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except MarketDataError:
            # Already a MarketDataError, re-raise
            raise
        except Exception as e:
            # Convert to MarketDataError with context
            raise MarketDataError(
                f"Market data operation failed in {func.__name__}",
                function=func.__name__,
                error_type=type(e).__name__,
                error_message=str(e)
            ) from e

    return wrapper


def handle_order_errors(func):
    """
    Decorator to handle order execution errors gracefully.

    FIXED (Problem 14): Standardized error handling for order operations.

    Converts order errors to OrderExecutionError with full context.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except OrderExecutionError:
            # Already an OrderExecutionError, re-raise
            raise
        except Exception as e:
            # Convert to OrderExecutionError with context
            raise OrderExecutionError(
                f"Order operation failed in {func.__name__}",
                function=func.__name__,
                error_type=type(e).__name__,
                error_message=str(e)
            ) from e

    return wrapper
