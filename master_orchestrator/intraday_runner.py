#!/Users/henry/miniconda3/envs/trading/bin/python
"""
Intraday Runner for Trading Data Pipeline

Executes high-frequency tasks during market hours.
- Every 1 minute: Price data
- Every 5 minutes: ML features, trading signals
"""

import subprocess
import sqlite3
import yaml
import time
import signal
import sys
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from zoneinfo import ZoneInfo

from health_monitor import HealthMonitor


# Python interpreter for trading environment
PYTHON_BIN = "/Users/henry/miniconda3/envs/trading/bin/python"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/intraday_runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IntradayRunner:
    """Execute intraday data collection tasks with scheduling"""

    def __init__(self, config_path: Optional[str] = None,
                 db_path: str = "/Volumes/Vault/85_assets_prediction.db",
                 dry_run: bool = False):
        """
        Initialize intraday runner

        Args:
            config_path: Path to dependency_graph.yaml (auto-detected if None)
            db_path: Path to SQLite database
            dry_run: If True, don't actually run scripts (for testing)
        """
        self.db_path = db_path
        self.dry_run = dry_run
        self.running = True

        # Auto-detect config path
        if config_path is None:
            script_dir = Path(__file__).parent
            config_path = script_dir / "dependency_graph.yaml"

        self.config_path = config_path
        self.config = self._load_config()

        # Get project root (parent of master_orchestrator)
        self.project_root = Path(__file__).parent.parent

        # Initialize health monitor
        self.health_monitor = HealthMonitor(db_path=db_path, config_path=config_path)

        # Track last execution times
        self.last_execution: Dict[str, datetime] = {}

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info("\n📴 Shutdown signal received. Finishing current tasks...")
        self.running = False

    def _load_config(self) -> dict:
        """Load configuration from YAML"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            sys.exit(1)

    def is_market_hours(self) -> bool:
        """
        Check if current time is during market hours

        Returns:
            True if market is open, False otherwise
        """
        market_config = self.config.get('market_hours', {}).get('regular', {})
        timezone_str = market_config.get('timezone', 'America/New_York')

        try:
            tz = ZoneInfo(timezone_str)
            now = datetime.now(tz)
            current_time = now.time()

            # Parse market hours
            start_str = market_config.get('start', '09:30')
            end_str = market_config.get('end', '16:00')

            start_hour, start_min = map(int, start_str.split(':'))
            end_hour, end_min = map(int, end_str.split(':'))

            market_start = dt_time(start_hour, start_min)
            market_end = dt_time(end_hour, end_min)

            # Check if it's a weekday (Monday=0, Sunday=6)
            is_weekday = now.weekday() < 5

            # Check if current time is within market hours
            is_open = market_start <= current_time <= market_end

            return is_weekday and is_open

        except Exception as e:
            logger.warning(f"Error checking market hours: {e}. Assuming market is closed.")
            return False

    def should_run_task(self, task: Dict) -> bool:
        """
        Check if a task should run based on its interval and last execution

        Args:
            task: Task configuration dictionary

        Returns:
            True if task should run now
        """
        task_name = task['name']
        interval_minutes = task.get('interval_minutes', 5)
        market_hours_only = task.get('market_hours_only', True)

        # Check market hours requirement
        if market_hours_only and not self.is_market_hours():
            return False

        # Check if enough time has passed since last execution
        if task_name in self.last_execution:
            time_since_last = (datetime.now() - self.last_execution[task_name]).total_seconds()
            interval_seconds = interval_minutes * 60

            if time_since_last < interval_seconds:
                return False

        return True

    def run_script(self, task: Dict) -> Tuple[bool, str, int, float]:
        """
        Run a data collection script

        Args:
            task: Task configuration dictionary

        Returns:
            (success, error_message, records_processed, runtime_seconds)
        """
        script_path = self.project_root / task['script']
        task_name = task['name']
        max_runtime = task.get('max_runtime_seconds', 120)
        args = task.get('args', [])

        logger.info(f"▶️  Running: {task_name}")

        if self.dry_run:
            logger.info(f"   [DRY RUN] Would execute: {PYTHON_BIN} {script_path} {' '.join(args)}")
            return True, "", 0, 0.0

        # Mark task as started
        self.health_monitor.mark_task_started(task_name)

        try:
            start_time = time.time()

            # Build command
            cmd = [PYTHON_BIN, str(script_path)] + args

            # Execute script with timeout
            # Don't capture output - let it stream to console for visibility
            result = subprocess.run(
                cmd,
                timeout=max_runtime,
                cwd=str(self.project_root)
            )

            runtime_seconds = time.time() - start_time

            if result.returncode == 0:
                # Success
                logger.info(f"   ✅ {task_name} completed in {runtime_seconds:.1f}s")
                return True, "", 0, runtime_seconds
            else:
                # Script failed
                logger.warning(f"   ❌ {task_name} failed with exit code {result.returncode}")
                return False, f"Exit code {result.returncode}", 0, runtime_seconds

        except subprocess.TimeoutExpired:
            runtime_seconds = time.time() - start_time
            error_msg = f"Script timed out after {max_runtime}s"
            logger.error(f"   ⏱️  {task_name} timed out after {max_runtime}s")
            return False, error_msg, 0, runtime_seconds

        except Exception as e:
            runtime_seconds = time.time() - start_time
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"   💥 {task_name} crashed: {e}")
            return False, error_msg, 0, runtime_seconds

    def _extract_records_processed(self, output: str) -> int:
        """
        Try to extract number of records processed from script output

        Looks for patterns like:
        - "Processed 123 records"
        - "Inserted 456 rows"
        - "Updated 789 assets"
        """
        import re

        patterns = [
            r'Processed (\d+) records',
            r'Inserted (\d+) rows',
            r'Updated (\d+) assets',
            r'Collected (\d+) items',
            r'Added (\d+) new',
        ]

        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return 0

    def check_dependencies(self, task: Dict) -> bool:
        """
        Check if all dependencies for a task have been executed recently

        For intraday tasks, we don't enforce strict dependency ordering like
        the daily runner. We just check that dependencies have run at some point.

        Args:
            task: Task configuration dictionary

        Returns:
            True if all dependencies met, False otherwise
        """
        dependencies = task.get('dependencies', [])

        if not dependencies:
            return True  # No dependencies

        for dep_name in dependencies:
            # Check if dependency has been executed
            if dep_name not in self.last_execution:
                # Check if there's a task in the schedule with this name
                intraday_schedule = self.config.get('intraday_schedule', [])
                dep_task = next((t for t in intraday_schedule if t['name'] == dep_name), None)

                if dep_task:
                    logger.info(f"   ℹ️  Running dependency first: {dep_name}")
                    # Run the dependency first
                    success, error, records, runtime = self.run_script(dep_task)
                    self.last_execution[dep_name] = datetime.now()

                    # Update health monitor
                    status = 'SUCCESS' if success else 'FAILED'
                    self.health_monitor.update_task_status(
                        task_name=dep_name,
                        status=status,
                        success=success,
                        error_message=error,
                        records_processed=records,
                        runtime_seconds=runtime
                    )

                    if not success:
                        logger.warning(f"   ⚠️  Dependency '{dep_name}' failed")
                        return False
                else:
                    logger.warning(f"   ⚠️  Dependency '{dep_name}' not found in schedule")
                    return False

        return True

    def run_intraday_cycle(self):
        """Execute one cycle of intraday tasks"""
        intraday_schedule = self.config.get('intraday_schedule', [])

        if not intraday_schedule:
            logger.warning("⚠️  No intraday schedule found in config!")
            return

        # Sort tasks by tier
        tasks_by_tier: Dict[int, List[Dict]] = {}
        for task in intraday_schedule:
            tier = task.get('tier', 0)
            if tier not in tasks_by_tier:
                tasks_by_tier[tier] = []
            tasks_by_tier[tier].append(task)

        # Execute tasks by tier
        for tier in sorted(tasks_by_tier.keys()):
            tier_tasks = tasks_by_tier[tier]

            for task in tier_tasks:
                if not self.running:
                    logger.info("🛑 Shutdown requested, stopping task execution")
                    return

                task_name = task['name']

                # Check if task should run
                if not self.should_run_task(task):
                    continue

                # Check dependencies
                if not self.check_dependencies(task):
                    logger.warning(f"⏭️  Skipping {task_name} - dependencies not met")
                    continue

                # Run the task
                success, error_msg, records, runtime = self.run_script(task)

                # Update last execution time
                self.last_execution[task_name] = datetime.now()

                # Update health monitor
                status = 'SUCCESS' if success else 'FAILED'
                self.health_monitor.update_task_status(
                    task_name=task_name,
                    status=status,
                    success=success,
                    error_message=error_msg,
                    records_processed=records,
                    runtime_seconds=runtime
                )

    def run_continuous(self, check_interval_seconds: int = 30):
        """
        Run intraday pipeline continuously

        Args:
            check_interval_seconds: How often to check if tasks should run (default 30s)
        """
        logger.info("\n" + "="*80)
        logger.info(f"🔄 INTRADAY RUNNER STARTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80)
        logger.info(f"Check Interval: Every {check_interval_seconds}s")
        logger.info(f"Market Hours: {self.config.get('market_hours', {}).get('regular', {})}")
        logger.info("="*80 + "\n")

        if self.dry_run:
            logger.info("🧪 DRY RUN MODE - No scripts will actually execute\n")

        cycle_count = 0

        while self.running:
            try:
                cycle_count += 1
                logger.info(f"\n🔁 Cycle {cycle_count} - {datetime.now().strftime('%H:%M:%S')}")

                # Check if market is open
                is_market_open = self.is_market_hours()
                logger.info(f"   Market Status: {'🟢 OPEN' if is_market_open else '🔴 CLOSED'}")

                # Run task cycle
                self.run_intraday_cycle()

                # Sleep until next check
                logger.info(f"   💤 Sleeping for {check_interval_seconds}s...")
                time.sleep(check_interval_seconds)

            except KeyboardInterrupt:
                logger.info("\n⌨️  Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"\n💥 Error in main loop: {e}")
                logger.info(f"   Continuing after 60s...")
                time.sleep(60)

        logger.info("\n" + "="*80)
        logger.info("🏁 INTRADAY RUNNER STOPPED")
        logger.info("="*80 + "\n")

    def run_once(self):
        """Execute intraday pipeline once (useful for testing)"""
        logger.info("\n" + "="*80)
        logger.info(f"🔄 INTRADAY RUNNER (ONE-TIME) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80 + "\n")

        if self.dry_run:
            logger.info("🧪 DRY RUN MODE - No scripts will actually execute\n")

        # Check if market is open
        is_market_open = self.is_market_hours()
        logger.info(f"Market Status: {'🟢 OPEN' if is_market_open else '🔴 CLOSED'}\n")

        # Run one cycle
        self.run_intraday_cycle()

        logger.info("\n" + "="*80)
        logger.info("✅ INTRADAY RUNNER COMPLETED")
        logger.info("="*80 + "\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Run intraday data collection pipeline')
    parser.add_argument('--config', type=str, help='Path to dependency_graph.yaml')
    parser.add_argument('--db', type=str, default='/Volumes/Vault/85_assets_prediction.db',
                       help='Path to database')
    parser.add_argument('--dry-run', action='store_true',
                       help='Simulate execution without running scripts')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit (instead of continuous loop)')
    parser.add_argument('--interval', type=int, default=30,
                       help='Check interval in seconds (default: 30)')

    args = parser.parse_args()

    runner = IntradayRunner(
        config_path=args.config,
        db_path=args.db,
        dry_run=args.dry_run
    )

    if args.once:
        runner.run_once()
    else:
        runner.run_continuous(check_interval_seconds=args.interval)


if __name__ == "__main__":
    main()
