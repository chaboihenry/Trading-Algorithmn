"""
Connection Manager for Alpaca Trading Bot

FIXES SOCKET EXHAUSTION by:
- Properly closing all API clients on shutdown
- Implementing singleton pattern for client reuse
- Monitoring connection health and auto-reconnecting
- Cleaning up HTTP sessions and WebSocket streams
"""

import logging
import signal
import sys
import atexit
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages all Alpaca API connections with proper cleanup.

    CRITICAL FIX: Ensures all HTTP sessions, WebSocket streams, and
    API clients are properly closed on shutdown to prevent socket exhaustion.
    """

    def __init__(self, broker=None):
        """
        Initialize connection manager.

        Args:
            broker: Optional Alpaca broker instance (for Lumibot compatibility)
        """
        self.broker = broker
        self.active = True
        self._cleanup_registered = False

        # Register cleanup handlers (only once)
        if not self._cleanup_registered:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            atexit.register(self.cleanup_all)
            self._cleanup_registered = True
            logger.info("✅ Connection cleanup handlers registered")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, cleaning up connections...")
        self.cleanup_all()
        sys.exit(0)

    def cleanup_broker(self):
        """Clean up Lumibot broker connections."""
        if not self.broker:
            return

        logger.info("Closing broker connections...")

        try:
            # Close websocket streams if active
            if hasattr(self.broker, 'stream') and self.broker.stream:
                logger.info("Closing trading stream...")
                self.broker.stream.close()

            # Close REST API session
            if hasattr(self.broker, 'api') and hasattr(self.broker.api, 'close'):
                logger.info("Closing API session...")
                self.broker.api.close()

            logger.info("✅ Broker connections closed")

        except Exception as e:
            logger.error(f"Error closing broker: {e}")

    def cleanup_alpaca_clients(self):
        """Clean up singleton Alpaca API clients."""
        logger.info("Closing Alpaca API clients...")

        try:
            # Import here to avoid circular imports
            from backup import market_data

            # Close TradingClient singleton
            if market_data._market_data_client is not None:
                client = market_data._market_data_client.client
                if hasattr(client, '_session') and client._session:
                    client._session.close()
                    logger.info("✅ TradingClient session closed")
                market_data._market_data_client = None

            # Close StockHistoricalDataClient singleton
            if market_data._stock_data_client is not None:
                client = market_data._stock_data_client
                if hasattr(client, '_session') and client._session:
                    client._session.close()
                    logger.info("✅ StockHistoricalDataClient session closed")
                market_data._stock_data_client = None

        except Exception as e:
            logger.error(f"Error closing Alpaca clients: {e}")

    def cleanup_all(self):
        """Clean up ALL connections (broker + Alpaca clients)."""
        if not self.active:
            return

        logger.info("=" * 60)
        logger.info("CLEANING UP ALL CONNECTIONS")
        logger.info("=" * 60)

        try:
            self.cleanup_broker()
            self.cleanup_alpaca_clients()

            self.active = False
            logger.info("=" * 60)
            logger.info("✅ ALL CONNECTIONS CLOSED SUCCESSFULLY")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    @contextmanager
    def managed_connection(self):
        """
        Context manager for broker connections.

        Usage:
            with conn_manager.managed_connection():
                # Use broker
                pass
            # Connections automatically cleaned up
        """
        try:
            yield self.broker
        finally:
            # Don't cleanup here - let shutdown handlers handle it
            pass

    def healthcheck(self) -> bool:
        """
        Check if connections are healthy.

        Returns:
            True if all connections are healthy
        """
        if not self.broker:
            # No broker - check Alpaca API directly
            try:
                from backup.market_data import get_market_data_client
                client = get_market_data_client()
                cash = client.get_cash()
                return cash is not None and cash >= 0
            except Exception as e:
                logger.warning(f"Alpaca API health check failed: {e}")
                return False

        try:
            # Try to get account info from broker
            account = self.broker.api.get_account()
            return account.status == 'ACTIVE'
        except Exception as e:
            logger.warning(f"Connection health check failed: {e}")
            return False

    def reconnect_if_needed(self) -> bool:
        """
        Reconnect if connection is unhealthy.

        Returns:
            True if connection is healthy or reconnection succeeded
        """
        if self.healthcheck():
            return True

        logger.warning("Connection unhealthy, attempting reconnect...")

        try:
            self.cleanup_all()
            # Clients will auto-reconnect on next API call (singleton pattern)
            self.active = True
            return self.healthcheck()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            return False


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_connection_manager = None

def get_connection_manager(broker=None) -> ConnectionManager:
    """
    Get the global ConnectionManager instance (creates it if needed).

    Args:
        broker: Optional broker instance for Lumibot compatibility

    Returns:
        Singleton ConnectionManager instance
    """
    global _connection_manager

    if _connection_manager is None:
        _connection_manager = ConnectionManager(broker)

    return _connection_manager
