#!/Users/henry/miniconda3/envs/trading/bin/python
"""
Daily Runner for Trading Data Pipeline

Executes daily scheduled tasks with dependency management.
Runs at 5:00 PM EST (after market close) for complete OHLCV data and full pipeline execution.
"""

import subprocess
import sqlite3
import yaml
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import logging

from health_monitor import HealthMonitor


# Python interpreter for trading environment
PYTHON_BIN = "/Users/henry/miniconda3/envs/trading/bin/python"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/daily_runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DailyRunner:
    """Execute daily data collection tasks with dependency management"""

    def __init__(self, config_path: Optional[str] = None,
                 db_path: str = "/Volumes/Vault/85_assets_prediction.db",
                 dry_run: bool = False):
        """
        Initialize daily runner

        Args:
            config_path: Path to dependency_graph.yaml (auto-detected if None)
            db_path: Path to SQLite database
            dry_run: If True, don't actually run scripts (for testing)
        """
        self.db_path = db_path
        self.dry_run = dry_run

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

        # Track execution results
        self.results: Dict[str, Dict] = {}

    def _load_config(self) -> dict:
        """Load configuration from YAML"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            sys.exit(1)

    def run_script(self, task: Dict, retry_policy: Dict) -> Tuple[bool, str, int, float]:
        """
        Run a data collection script with retry logic

        Args:
            task: Task configuration dictionary
            retry_policy: Retry configuration

        Returns:
            (success, error_message, records_processed, runtime_seconds)
        """
        script_path = self.project_root / task['script']
        task_name = task['name']
        max_runtime = task.get('max_runtime_seconds', 300)

        logger.info(f"▶️  Running: {task_name} ({script_path})")

        if self.dry_run:
            logger.info(f"   [DRY RUN] Would execute: {PYTHON_BIN} {script_path}")
            return True, "", 0, 0.0

        # Mark task as started
        self.health_monitor.mark_task_started(task_name)

        # Retry logic
        max_attempts = retry_policy.get('max_attempts', 3)
        backoff_seconds = retry_policy.get('backoff_seconds', [10, 30, 60])

        for attempt in range(1, max_attempts + 1):
            try:
                start_time = time.time()

                # Execute script with timeout and capture output
                result = subprocess.run(
                    [PYTHON_BIN, str(script_path)],
                    timeout=max_runtime,
                    cwd=str(self.project_root),
                    capture_output=True,
                    text=True
                )

                runtime_seconds = time.time() - start_time
                records_processed = self._extract_records_processed(result.stdout)

                if result.returncode == 0:
                    # Success
                    logger.info(f"   ✅ {task_name} completed in {runtime_seconds:.1f}s ({records_processed} records)")
                    if result.stdout:
                        logger.debug(f"   Output:\n{result.stdout.strip()}")
                    return True, "", records_processed, runtime_seconds
                else:
                    # Script failed
                    error_message = f"Exit code {result.returncode}"
                    logger.warning(f"   ❌ {task_name} failed with {error_message} (attempt {attempt}/{max_attempts})")
                    if result.stderr:
                        logger.error(f"   Stderr:\n{result.stderr.strip()}")
                    if result.stdout:
                        logger.warning(f"   Stdout:\n{result.stdout.strip()}")

                    # Retry if not last attempt
                    if attempt < max_attempts:
                        wait_time = backoff_seconds[min(attempt - 1, len(backoff_seconds) - 1)]
                        logger.info(f"      Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return False, error_message, records_processed, runtime_seconds

            except subprocess.TimeoutExpired as e:
                runtime_seconds = time.time() - start_time
                error_msg = f"Script timed out after {max_runtime}s"
                logger.error(f"   ⏱️  {task_name} timed out after {max_runtime}s")
                # Log any output captured before the timeout
                if e.stdout:
                    logger.warning(f"   Stdout (partial):\n{e.stdout.strip()}")
                if e.stderr:
                    logger.error(f"   Stderr (partial):\n{e.stderr.strip()}")

                if attempt < max_attempts:
                    wait_time = backoff_seconds[min(attempt - 1, len(backoff_seconds) - 1)]
                    logger.info(f"      Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    return False, error_msg, 0, runtime_seconds

            except Exception as e:
                runtime_seconds = time.time() - start_time
                error_msg = f"Unexpected error: {str(e)}"
                logger.error(f"   💥 {task_name} crashed: {e}")

                if attempt < max_attempts:
                    wait_time = backoff_seconds[min(attempt - 1, len(backoff_seconds) - 1)]
                    logger.info(f"      Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    return False, error_msg, 0, runtime_seconds

        # Should never reach here, but just in case
        return False, "Max retries exceeded", 0, 0.0

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
        Check if all dependencies for a task have completed successfully

        Args:
            task: Task configuration dictionary

        Returns:
            True if all dependencies met, False otherwise
        """
        dependencies = task.get('dependencies', [])

        if not dependencies:
            return True  # No dependencies

        for dep_name in dependencies:
            if dep_name not in self.results:
                logger.warning(f"   ⚠️  Dependency '{dep_name}' has not run yet")
                return False

            if not self.results[dep_name]['success']:
                logger.warning(f"   ⚠️  Dependency '{dep_name}' failed")
                return False

        return True

    def run_tier(self, tier: int, tasks: List[Dict], retry_policy: Dict) -> Dict[str, bool]:
        """
        Run all tasks in a specific tier

        Args:
            tier: Tier number
            tasks: List of tasks in this tier
            retry_policy: Retry configuration

        Returns:
            Dictionary mapping task names to success status
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🔄 TIER {tier} - {len(tasks)} task(s)")
        logger.info(f"{'='*80}")

        tier_results = {}

        for task in tasks:
            task_name = task['name']
            is_critical = task.get('critical', False)

            # Check dependencies
            if not self.check_dependencies(task):
                logger.warning(f"⏭️  Skipping {task_name} - dependencies not met")
                tier_results[task_name] = False
                self.results[task_name] = {
                    'success': False,
                    'skipped': True,
                    'error': 'Dependencies not met'
                }
                continue

            # Run the task
            success, error_msg, records, runtime = self.run_script(task, retry_policy)

            # Store results
            tier_results[task_name] = success
            self.results[task_name] = {
                'success': success,
                'skipped': False,
                'error': error_msg,
                'records_processed': records,
                'runtime_seconds': runtime
            }

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

            # Handle critical task failure
            if not success and is_critical:
                logger.error(f"\n{'='*80}")
                logger.error(f"🚨 CRITICAL TASK FAILED: {task_name}")
                logger.error(f"{'='*80}")
                logger.error(f"Error: {error_msg}")
                logger.error(f"\nAborting pipeline due to critical failure.")
                logger.error(f"{'='*80}\n")
                return tier_results

        return tier_results

    def run_daily_pipeline(self):
        """Execute the complete daily pipeline"""
        logger.info("\n" + "="*80)
        logger.info(f"🌅 DAILY PIPELINE STARTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80)

        if self.dry_run:
            logger.info("🧪 DRY RUN MODE - No scripts will actually execute")

        start_time = time.time()

        # Get daily schedule and retry policy
        daily_schedule = self.config.get('daily_schedule', [])
        retry_policy = self.config.get('retry_policy', {})

        if not daily_schedule:
            logger.error("❌ No daily schedule found in config!")
            sys.exit(1)

        # Group tasks by tier
        tiers: Dict[int, List[Dict]] = {}
        for task in daily_schedule:
            tier = task.get('tier', 0)
            if tier not in tiers:
                tiers[tier] = []
            tiers[tier].append(task)

        # Execute tiers in order
        all_success = True
        for tier in sorted(tiers.keys()):
            tier_tasks = tiers[tier]
            tier_results = self.run_tier(tier, tier_tasks, retry_policy)

            # Check if any critical task failed
            for task in tier_tasks:
                if task.get('critical', False) and not tier_results.get(task['name'], False):
                    all_success = False
                    logger.error("\n🛑 STOPPING PIPELINE - Critical task failed in tier {tier}")
                    break

            if not all_success:
                break

        # Print summary
        total_runtime = time.time() - start_time
        self._print_summary(total_runtime)

        # Return appropriate exit code
        if all_success:
            logger.info("\n✅ Daily pipeline completed successfully!")
            sys.exit(0)
        else:
            logger.error("\n❌ Daily pipeline completed with errors!")
            sys.exit(1)

    def _print_summary(self, total_runtime: float):
        """Print execution summary"""
        logger.info("\n" + "="*80)
        logger.info("📊 EXECUTION SUMMARY")
        logger.info("="*80)

        successful = [name for name, result in self.results.items()
                     if result['success']]
        failed = [name for name, result in self.results.items()
                 if not result['success'] and not result.get('skipped', False)]
        skipped = [name for name, result in self.results.items()
                  if result.get('skipped', False)]

        total_tasks = len(self.results)
        total_records = sum(result.get('records_processed', 0)
                           for result in self.results.values())

        logger.info(f"\n📈 Tasks: {len(successful)} succeeded, {len(failed)} failed, "
                   f"{len(skipped)} skipped (Total: {total_tasks})")

        if successful:
            logger.info(f"\n✅ Successful Tasks ({len(successful)}):")
            for task_name in successful:
                result = self.results[task_name]
                logger.info(f"   • {task_name} - {result['runtime_seconds']:.1f}s, "
                           f"{result['records_processed']} records")

        if failed:
            logger.info(f"\n❌ Failed Tasks ({len(failed)}):")
            for task_name in failed:
                result = self.results[task_name]
                logger.info(f"   • {task_name} - {result['error'][:80]}")

        if skipped:
            logger.info(f"\n⏭️  Skipped Tasks ({len(skipped)}):")
            for task_name in skipped:
                logger.info(f"   • {task_name}")

        logger.info(f"\n⏱️  Total Runtime: {total_runtime:.1f}s")
        logger.info(f"📦 Total Records Processed: {total_records:,}")
        logger.info("="*80 + "\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Run daily data collection pipeline')
    parser.add_argument('--config', type=str, help='Path to dependency_graph.yaml')
    parser.add_argument('--db', type=str, default='/Volumes/Vault/85_assets_prediction.db',
                       help='Path to database')
    parser.add_argument('--dry-run', action='store_true',
                       help='Simulate execution without running scripts')

    args = parser.parse_args()

    runner = DailyRunner(
        config_path=args.config,
        db_path=args.db,
        dry_run=args.dry_run
    )

    runner.run_daily_pipeline()


if __name__ == "__main__":
    main()
