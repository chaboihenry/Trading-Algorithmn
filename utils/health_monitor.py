"""
Health Monitor for Trading Bot

Tracks bot health metrics and provides health endpoints for watchdog monitoring.

Features:
- Memory usage tracking
- Iteration counting
- Last successful trade timestamp
- Error rate monitoring
- Health status reporting
"""

import os
import time
import psutil
import logging
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class HealthMonitor:
    """
    Monitors bot health and provides status reporting.

    Tracks:
    - Memory usage and growth
    - CPU usage
    - Iteration count
    - Last successful operations
    - Error counts
    - Uptime
    """

    def __init__(self):
        """Initialize health monitor."""
        self.start_time = datetime.now()
        self.process = psutil.Process(os.getpid())

        # Metrics
        self.iteration_count = 0
        self.error_count = 0
        self.last_successful_iteration: Optional[datetime] = None
        self.last_trade_time: Optional[datetime] = None

        # Memory tracking
        self.initial_memory_mb = self._get_memory_mb()
        self.peak_memory_mb = self.initial_memory_mb

        # Health status file for watchdog
        self.health_file = Path('/tmp/trading_bot_health.txt')

        logger.info(f"✅ Health monitor initialized (PID: {os.getpid()})")
        logger.info(f"Initial memory: {self.initial_memory_mb:.1f}MB")

    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def _get_cpu_percent(self) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent(interval=0.1)

    def record_iteration(self):
        """Record a successful iteration."""
        self.iteration_count += 1
        self.last_successful_iteration = datetime.now()
        self._update_health_file()

    def record_error(self):
        """Record an error occurrence."""
        self.error_count += 1
        self._update_health_file()

    def record_trade(self):
        """Record a successful trade."""
        self.last_trade_time = datetime.now()
        self._update_health_file()

    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get memory usage statistics.

        Returns:
            Dict with current, peak, and growth metrics
        """
        current_mb = self._get_memory_mb()

        # Update peak
        if current_mb > self.peak_memory_mb:
            self.peak_memory_mb = current_mb

        growth_mb = current_mb - self.initial_memory_mb
        growth_percent = (growth_mb / self.initial_memory_mb) * 100 if self.initial_memory_mb > 0 else 0

        return {
            'current_mb': current_mb,
            'initial_mb': self.initial_memory_mb,
            'peak_mb': self.peak_memory_mb,
            'growth_mb': growth_mb,
            'growth_percent': growth_percent
        }

    def check_memory_leak(self, threshold_mb: float = 100) -> bool:
        """
        Check if memory growth indicates a leak.

        Args:
            threshold_mb: Memory growth threshold in MB

        Returns:
            True if potential leak detected
        """
        stats = self.get_memory_stats()

        if stats['growth_mb'] > threshold_mb:
            logger.warning(f"⚠️ POTENTIAL MEMORY LEAK DETECTED")
            logger.warning(f"Memory growth: {stats['growth_mb']:.1f}MB (+{stats['growth_percent']:.1f}%)")
            return True

        return False

    def get_health_status(self) -> Dict[str, any]:
        """
        Get comprehensive health status.

        Returns:
            Dict with all health metrics
        """
        uptime = (datetime.now() - self.start_time).total_seconds()
        memory_stats = self.get_memory_stats()

        status = {
            'healthy': True,
            'pid': os.getpid(),
            'uptime_seconds': uptime,
            'uptime_hours': uptime / 3600,
            'iteration_count': self.iteration_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.iteration_count, 1),
            'memory_mb': memory_stats['current_mb'],
            'memory_growth_mb': memory_stats['growth_mb'],
            'memory_growth_percent': memory_stats['growth_percent'],
            'cpu_percent': self._get_cpu_percent(),
            'last_iteration': self.last_successful_iteration.isoformat() if self.last_successful_iteration else None,
            'last_trade': self.last_trade_time.isoformat() if self.last_trade_time else None,
        }

        # Determine if unhealthy
        if memory_stats['current_mb'] > 1024:  # Over 1GB
            status['healthy'] = False
            status['reason'] = 'High memory usage'
        elif memory_stats['growth_percent'] > 50:  # 50% growth
            status['healthy'] = False
            status['reason'] = 'Memory leak suspected'
        elif self.error_count > 10 and status['error_rate'] > 0.5:
            status['healthy'] = False
            status['reason'] = 'High error rate'

        return status

    def _update_health_file(self):
        """Write health status to file for watchdog monitoring."""
        try:
            status = self.get_health_status()

            # Write simple status file
            with open(self.health_file, 'w') as f:
                f.write(f"healthy={status['healthy']}\n")
                f.write(f"pid={status['pid']}\n")
                f.write(f"uptime_hours={status['uptime_hours']:.2f}\n")
                f.write(f"iterations={status['iteration_count']}\n")
                f.write(f"errors={status['error_count']}\n")
                f.write(f"memory_mb={status['memory_mb']:.1f}\n")
                f.write(f"timestamp={datetime.now().isoformat()}\n")

        except Exception as e:
            logger.debug(f"Could not write health file: {e}")

    def log_health_summary(self):
        """Log a health summary."""
        status = self.get_health_status()

        logger.info("=" * 60)
        logger.info("HEALTH SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Status: {'✅ HEALTHY' if status['healthy'] else '🔴 UNHEALTHY'}")
        logger.info(f"Uptime: {status['uptime_hours']:.2f} hours")
        logger.info(f"Iterations: {status['iteration_count']}")
        logger.info(f"Errors: {status['error_count']} (rate: {status['error_rate']:.1%})")
        logger.info(f"Memory: {status['memory_mb']:.1f}MB (+{status['memory_growth_mb']:.1f}MB)")
        logger.info(f"CPU: {status['cpu_percent']:.1f}%")

        if status['last_iteration']:
            logger.info(f"Last iteration: {status['last_iteration']}")
        if status['last_trade']:
            logger.info(f"Last trade: {status['last_trade']}")

        if not status['healthy']:
            logger.warning(f"Unhealthy reason: {status.get('reason', 'Unknown')}")

        logger.info("=" * 60)


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_health_monitor = None

def get_health_monitor() -> HealthMonitor:
    """
    Get the global HealthMonitor instance (creates it if needed).

    Returns:
        Singleton HealthMonitor instance
    """
    global _health_monitor

    if _health_monitor is None:
        _health_monitor = HealthMonitor()

    return _health_monitor
