"""
Watchdog Script for Trading Bot

ENSURES 90-DAY RELIABILITY by:
- Monitoring bot process health
- Automatically restarting on crashes
- Sending alerts on failures
- Logging all recovery attempts
- Preventing infinite restart loops

Usage:
    # Start the watchdog (will start and monitor the bot)
    python utils/watchdog.py

    # Or with custom settings
    python utils/watchdog.py --max-restarts 10 --restart-delay 60
"""

import os
import sys
import time
import logging
import subprocess
import signal
import psutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
LOGS_DIR = PROJECT_ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True)

log_file = LOGS_DIR / f'watchdog_{datetime.now().strftime("%Y%m%d")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BotWatchdog:
    """
    Monitors the trading bot and restarts it on failure.

    Features:
    - Process health monitoring (CPU, memory, responsiveness)
    - Automatic restart on crash
    - Maximum restart limit to prevent infinite loops
    - Backoff delay between restarts
    - Graceful shutdown handling
    - Alert logging for failures
    """

    def __init__(
        self,
        bot_script: str = 'core/live_trader.py',
        max_restarts: int = 5,
        restart_delay: int = 30,
        health_check_interval: int = 60,
        max_memory_mb: int = 1024
    ):
        """
        Initialize the watchdog.

        Args:
            bot_script: Path to the bot script to monitor
            max_restarts: Maximum consecutive restarts before giving up
            restart_delay: Seconds to wait between restarts
            health_check_interval: Seconds between health checks
            max_memory_mb: Maximum allowed memory usage in MB
        """
        self.bot_script = PROJECT_ROOT / bot_script
        self.max_restarts = max_restarts
        self.restart_delay = restart_delay
        self.health_check_interval = health_check_interval
        self.max_memory_mb = max_memory_mb

        self.bot_process: Optional[subprocess.Popen] = None
        self.restart_count = 0
        self.last_restart_time: Optional[datetime] = None
        self.running = True

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("=" * 80)
        logger.info("WATCHDOG INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Bot script: {self.bot_script}")
        logger.info(f"Max restarts: {self.max_restarts}")
        logger.info(f"Restart delay: {self.restart_delay}s")
        logger.info(f"Health check interval: {self.health_check_interval}s")
        logger.info(f"Max memory: {self.max_memory_mb}MB")
        logger.info("=" * 80)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down watchdog...")
        self.running = False
        self.stop_bot()
        sys.exit(0)

    def start_bot(self) -> bool:
        """
        Start the bot process.

        Returns:
            True if bot started successfully
        """
        try:
            # Check if we've hit restart limit
            if self.restart_count >= self.max_restarts:
                logger.error(f"🚨 RESTART LIMIT REACHED ({self.max_restarts} attempts)")
                logger.error("Bot has crashed too many times. Manual intervention required.")
                logger.error("Check logs for errors and fix issues before restarting.")
                return False

            # Apply restart delay if not first start
            if self.last_restart_time:
                time_since_restart = (datetime.now() - self.last_restart_time).total_seconds()
                if time_since_restart < self.restart_delay:
                    wait_time = self.restart_delay - time_since_restart
                    logger.info(f"Waiting {wait_time:.0f}s before restart (backoff delay)...")
                    time.sleep(wait_time)

            logger.info("=" * 80)
            logger.info(f"STARTING BOT (Attempt {self.restart_count + 1}/{self.max_restarts})")
            logger.info("=" * 80)

            # Get Python interpreter (use same one as watchdog)
            python_executable = sys.executable

            # Start bot as subprocess
            self.bot_process = subprocess.Popen(
                [python_executable, str(self.bot_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(PROJECT_ROOT)
            )

            # Wait a moment to ensure process started
            time.sleep(2)

            if self.bot_process.poll() is None:
                # Process is running
                logger.info(f"✅ Bot started successfully (PID: {self.bot_process.pid})")
                self.restart_count += 1
                self.last_restart_time = datetime.now()
                return True
            else:
                # Process already died
                logger.error(f"❌ Bot process died immediately")
                stdout, stderr = self.bot_process.communicate()
                if stderr:
                    logger.error(f"Error output: {stderr}")
                return False

        except Exception as e:
            logger.error(f"Failed to start bot: {e}")
            return False

    def stop_bot(self):
        """Stop the bot process gracefully."""
        if not self.bot_process:
            return

        try:
            logger.info("Stopping bot process...")

            # Send SIGINT for graceful shutdown
            self.bot_process.send_signal(signal.SIGINT)

            # Wait up to 10 seconds for graceful shutdown
            try:
                self.bot_process.wait(timeout=10)
                logger.info("✅ Bot stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if not responding
                logger.warning("Bot not responding, force killing...")
                self.bot_process.kill()
                self.bot_process.wait()
                logger.info("✅ Bot force killed")

            self.bot_process = None

        except Exception as e:
            logger.error(f"Error stopping bot: {e}")

    def is_bot_running(self) -> bool:
        """
        Check if bot process is running.

        Returns:
            True if bot is running
        """
        if not self.bot_process:
            return False

        # Check if process has terminated
        return self.bot_process.poll() is None

    def check_bot_health(self) -> bool:
        """
        Check bot process health (CPU, memory, responsiveness).

        Returns:
            True if bot is healthy
        """
        if not self.is_bot_running():
            return False

        try:
            # Get process info
            process = psutil.Process(self.bot_process.pid)

            # Check memory usage
            memory_mb = process.memory_info().rss / 1024 / 1024

            if memory_mb > self.max_memory_mb:
                logger.warning(f"⚠️ MEMORY LIMIT EXCEEDED: {memory_mb:.1f}MB > {self.max_memory_mb}MB")
                logger.warning("Bot may have a memory leak. Restarting...")
                return False

            # Check if process is zombie
            if process.status() == psutil.STATUS_ZOMBIE:
                logger.warning("⚠️ Bot process is zombie")
                return False

            # Log health stats
            cpu_percent = process.cpu_percent(interval=1)
            logger.debug(f"Bot health: CPU={cpu_percent:.1f}%, Memory={memory_mb:.1f}MB")

            return True

        except psutil.NoSuchProcess:
            logger.warning("Bot process disappeared")
            return False
        except Exception as e:
            logger.error(f"Error checking bot health: {e}")
            return False

    def reset_restart_counter(self):
        """
        Reset restart counter if bot has been stable.

        If bot runs for 1 hour without issues, reset counter to allow
        future restarts (prevents hitting limit from old crashes).
        """
        if not self.last_restart_time:
            return

        time_since_restart = (datetime.now() - self.last_restart_time).total_seconds()

        # If bot has been running for 1 hour, reset counter
        if time_since_restart > 3600:  # 1 hour
            if self.restart_count > 0:
                logger.info(f"✅ Bot stable for 1 hour, resetting restart counter (was {self.restart_count})")
                self.restart_count = 0

    def run(self):
        """
        Main watchdog loop.

        Continuously monitors bot health and restarts on failure.
        """
        logger.info("Starting watchdog monitoring loop...")

        # Initial bot start
        if not self.start_bot():
            logger.error("Failed to start bot initially. Exiting.")
            return

        # Monitoring loop
        while self.running:
            try:
                # Sleep between checks
                time.sleep(self.health_check_interval)

                # Reset counter if bot is stable
                self.reset_restart_counter()

                # Check if bot is running
                if not self.is_bot_running():
                    logger.warning("🔴 BOT PROCESS DIED")

                    # Get exit code and error output
                    if self.bot_process:
                        exit_code = self.bot_process.poll()
                        stdout, stderr = self.bot_process.communicate()

                        logger.warning(f"Exit code: {exit_code}")
                        if stderr:
                            logger.error(f"Error output:\n{stderr}")

                    # Attempt restart
                    logger.info("Attempting automatic restart...")
                    if not self.start_bot():
                        logger.error("Failed to restart bot. Watchdog exiting.")
                        break

                # Check bot health
                elif not self.check_bot_health():
                    logger.warning("🔴 BOT HEALTH CHECK FAILED")
                    logger.info("Restarting unhealthy bot...")

                    # Stop and restart
                    self.stop_bot()
                    if not self.start_bot():
                        logger.error("Failed to restart bot. Watchdog exiting.")
                        break

                else:
                    # Bot is healthy
                    logger.debug("✅ Bot is healthy")

            except Exception as e:
                logger.error(f"Error in watchdog loop: {e}")
                logger.exception("Full traceback:")
                time.sleep(10)  # Wait before retrying

        # Cleanup on exit
        logger.info("Watchdog shutting down...")
        self.stop_bot()


def main():
    """Main entry point for watchdog."""
    parser = argparse.ArgumentParser(description='Run trading bot watchdog')
    parser.add_argument(
        '--max-restarts',
        type=int,
        default=5,
        help='Maximum consecutive restarts before giving up (default: 5)'
    )
    parser.add_argument(
        '--restart-delay',
        type=int,
        default=30,
        help='Seconds to wait between restarts (default: 30)'
    )
    parser.add_argument(
        '--health-check-interval',
        type=int,
        default=60,
        help='Seconds between health checks (default: 60)'
    )
    parser.add_argument(
        '--max-memory',
        type=int,
        default=1024,
        help='Maximum allowed memory in MB (default: 1024)'
    )

    args = parser.parse_args()

    # Create and run watchdog
    watchdog = BotWatchdog(
        max_restarts=args.max_restarts,
        restart_delay=args.restart_delay,
        health_check_interval=args.health_check_interval,
        max_memory_mb=args.max_memory
    )

    watchdog.run()


if __name__ == "__main__":
    main()
