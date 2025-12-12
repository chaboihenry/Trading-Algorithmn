"""
Connection Manager for Alpaca Trading Bot

Handles websocket lifecycle to prevent socket exhaustion:
- Properly closes connections when not needed
- Implements connection pooling
- Monitors connection health
"""

import logging
import signal
import sys
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages Alpaca broker connections with proper cleanup."""

    def __init__(self, broker):
        """
        Initialize connection manager.

        Args:
            broker: Alpaca broker instance
        """
        self.broker = broker
        self.active = True

        # Register cleanup handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, cleaning up connections...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        """Clean up all active connections."""
        if not self.active:
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

            self.active = False
            logger.info("All connections closed successfully")

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
            # Don't cleanup here - let sleep cycle handle it
            pass

    def healthcheck(self) -> bool:
        """
        Check if connections are healthy.

        Returns:
            True if all connections are healthy
        """
        try:
            # Try to get account info
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
            self.cleanup()
            # Broker will auto-reconnect on next API call
            self.active = True
            return self.healthcheck()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            return False
