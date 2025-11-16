#!/Users/henry/miniconda3/envs/trading/bin/python
"""
Master Orchestrator for Trading Data Pipeline

Central controller that coordinates daily and intraday runners.
Provides unified interface for managing the entire data pipeline.
"""

import subprocess
import sys
import time
import signal
import argparse
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Optional
import logging

from health_monitor import HealthMonitor


# Python interpreter for trading environment
PYTHON_BIN = "/Users/henry/miniconda3/envs/trading/bin/python"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MasterOrchestrator:
    """Master controller for the entire data pipeline"""

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db",
                 config_path: Optional[str] = None,
                 dry_run: bool = False):
        """
        Initialize master orchestrator

        Args:
            db_path: Path to SQLite database
            config_path: Path to dependency_graph.yaml (auto-detected if None)
            dry_run: If True, don't actually run anything
        """
        self.db_path = db_path
        self.config_path = config_path
        self.dry_run = dry_run

        # Get script directory
        self.script_dir = Path(__file__).parent

        # Initialize health monitor
        self.health_monitor = HealthMonitor(db_path=db_path, config_path=config_path)

        # Track running processes
        self.processes = []
        self.running = True

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info("\n📴 Shutdown signal received. Stopping all processes...")
        self.running = False
        self._cleanup_processes()
        sys.exit(0)

    def _cleanup_processes(self):
        """Terminate all running subprocess"""
        for process in self.processes:
            if process.poll() is None:  # Process is still running
                logger.info(f"   Terminating subprocess {process.pid}...")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning(f"   Force killing subprocess {process.pid}...")
                    process.kill()

    def run_daily_pipeline(self, blocking: bool = True) -> int:
        """
        Execute the daily data collection pipeline

        Args:
            blocking: If True, wait for completion. If False, run in background.

        Returns:
            Exit code (0 = success, non-zero = failure)
        """
        logger.info("\n" + "="*80)
        logger.info("🌅 Starting Daily Pipeline")
        logger.info("="*80 + "\n")

        cmd = [PYTHON_BIN, str(self.script_dir / 'daily_runner.py')]

        if self.config_path:
            cmd.extend(['--config', str(self.config_path)])
        if self.db_path:
            cmd.extend(['--db', self.db_path])
        if self.dry_run:
            cmd.append('--dry-run')

        if blocking:
            # Run and wait for completion
            result = subprocess.run(cmd)
            return result.returncode
        else:
            # Run in background
            process = subprocess.Popen(cmd)
            self.processes.append(process)
            logger.info(f"   Daily pipeline started (PID: {process.pid})")
            return 0

    def run_intraday_pipeline(self, blocking: bool = False,
                             run_once: bool = False,
                             interval: int = 30) -> int:
        """
        Execute the intraday data collection pipeline

        Args:
            blocking: If True, wait for completion. If False, run in background.
            run_once: If True, run once and exit. If False, run continuously.
            interval: Check interval in seconds (default: 30)

        Returns:
            Exit code (0 = success, non-zero = failure)
        """
        logger.info("\n" + "="*80)
        logger.info("🔄 Starting Intraday Pipeline")
        logger.info("="*80 + "\n")

        cmd = [PYTHON_BIN, str(self.script_dir / 'intraday_runner.py')]

        if self.config_path:
            cmd.extend(['--config', str(self.config_path)])
        if self.db_path:
            cmd.extend(['--db', self.db_path])
        if self.dry_run:
            cmd.append('--dry-run')
        if run_once:
            cmd.append('--once')
        cmd.extend(['--interval', str(interval)])

        if blocking:
            # Run and wait for completion
            result = subprocess.run(cmd)
            return result.returncode
        else:
            # Run in background
            process = subprocess.Popen(cmd)
            self.processes.append(process)
            logger.info(f"   Intraday pipeline started (PID: {process.pid})")
            return 0

    def run_health_check(self) -> int:
        """
        Run health check on all data sources

        Returns:
            Exit code (0 = healthy, 1 = unhealthy)
        """
        logger.info("\n" + "="*80)
        logger.info("🏥 Running Health Check")
        logger.info("="*80 + "\n")

        # Print both dashboards
        self.health_monitor.print_health_dashboard()
        self.health_monitor.print_task_status_dashboard()

        # Return appropriate exit code
        if self.health_monitor.is_healthy():
            logger.info("✅ All systems healthy!")
            return 0
        else:
            logger.warning("⚠️  Some data sources are stale!")
            return 1

    def update_stale_data(self, min_staleness: str = 'yellow') -> int:
        """
        Update data sources that are stale

        Args:
            min_staleness: Minimum staleness level ('yellow' or 'red')

        Returns:
            Exit code
        """
        logger.info("\n" + "="*80)
        logger.info(f"🔄 Updating Stale Data (minimum: {min_staleness})")
        logger.info("="*80 + "\n")

        from health_monitor import HealthStatus

        # Get stale sources
        min_status = HealthStatus.RED if min_staleness == 'red' else HealthStatus.YELLOW
        stale_sources = self.health_monitor.get_stale_sources(min_status=min_status)

        if not stale_sources:
            logger.info("✅ No stale data sources found!")
            return 0

        logger.info(f"Found {len(stale_sources)} stale sources:")
        for source in stale_sources:
            logger.info(f"   • {source}")

        logger.info("\nRunning daily pipeline to refresh data...\n")

        # Run daily pipeline to refresh everything
        return self.run_daily_pipeline(blocking=True)

    def run_full_system(self, intraday_interval: int = 30):
        """
        Run the complete system (daily + continuous intraday)

        This is the main operational mode that would be used in production.

        Args:
            intraday_interval: Check interval for intraday tasks (seconds)
        """
        logger.info("\n" + "="*80)
        logger.info("🚀 STARTING FULL TRADING SYSTEM")
        logger.info("="*80)
        logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Database: {self.db_path}")
        logger.info(f"Intraday Interval: {intraday_interval}s")
        if self.dry_run:
            logger.info("🧪 DRY RUN MODE")
        logger.info("="*80 + "\n")

        try:
            # Step 1: Run daily pipeline
            logger.info("📅 Step 1: Running daily pipeline...")
            exit_code = self.run_daily_pipeline(blocking=True)

            if exit_code != 0:
                logger.error("\n❌ Daily pipeline failed! Check logs for details.")
                logger.error("   Aborting system startup.\n")
                return exit_code

            logger.info("\n✅ Daily pipeline completed successfully!\n")

            # Step 2: Start intraday pipeline (continuous)
            logger.info("⏰ Step 2: Starting continuous intraday pipeline...")
            self.run_intraday_pipeline(
                blocking=False,  # Run in background
                run_once=False,  # Continuous mode
                interval=intraday_interval
            )

            # Wait for intraday process
            logger.info("\n" + "="*80)
            logger.info("✅ SYSTEM RUNNING")
            logger.info("="*80)
            logger.info("Daily pipeline: ✅ Complete")
            logger.info("Intraday pipeline: 🔄 Running continuously")
            logger.info("\nPress Ctrl+C to stop...\n")

            # Keep main process alive
            while self.running:
                time.sleep(5)

                # Check if intraday process is still alive
                if self.processes:
                    if self.processes[0].poll() is not None:
                        logger.error("\n⚠️  Intraday pipeline stopped unexpectedly!")
                        logger.info("   Restarting intraday pipeline...")
                        self.processes.pop(0)
                        self.run_intraday_pipeline(
                            blocking=False,
                            run_once=False,
                            interval=intraday_interval
                        )

        except KeyboardInterrupt:
            logger.info("\n⌨️  Keyboard interrupt received")
        except Exception as e:
            logger.error(f"\n💥 Unexpected error: {e}")
            return 1
        finally:
            self._cleanup_processes()

        logger.info("\n" + "="*80)
        logger.info("🏁 SYSTEM SHUTDOWN COMPLETE")
        logger.info("="*80 + "\n")

        return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Master Orchestrator for Trading Data Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full system (daily + continuous intraday)
  python orchestrator.py --full

  # Run daily pipeline only
  python orchestrator.py --daily

  # Run intraday pipeline continuously
  python orchestrator.py --intraday

  # Run intraday pipeline once
  python orchestrator.py --intraday --once

  # Check health of all data sources
  python orchestrator.py --health

  # Update stale data
  python orchestrator.py --update-stale

  # Dry run (simulate without executing)
  python orchestrator.py --daily --dry-run
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--full', action='store_true',
                           help='Run full system (daily + continuous intraday)')
    mode_group.add_argument('--daily', action='store_true',
                           help='Run daily pipeline only')
    mode_group.add_argument('--intraday', action='store_true',
                           help='Run intraday pipeline')
    mode_group.add_argument('--health', action='store_true',
                           help='Check health of all data sources')
    mode_group.add_argument('--update-stale', action='store_true',
                           help='Update stale data sources')

    # Options
    parser.add_argument('--config', type=str, help='Path to dependency_graph.yaml')
    parser.add_argument('--db', type=str, default='/Volumes/Vault/85_assets_prediction.db',
                       help='Path to database')
    parser.add_argument('--dry-run', action='store_true',
                       help='Simulate execution without running scripts')
    parser.add_argument('--once', action='store_true',
                       help='Run intraday pipeline once (not continuous)')
    parser.add_argument('--interval', type=int, default=30,
                       help='Intraday check interval in seconds (default: 30)')
    parser.add_argument('--min-staleness', type=str, choices=['yellow', 'red'],
                       default='yellow', help='Minimum staleness for update (default: yellow)')

    args = parser.parse_args()

    # Create orchestrator
    orchestrator = MasterOrchestrator(
        db_path=args.db,
        config_path=args.config,
        dry_run=args.dry_run
    )

    # Execute requested mode
    if args.full:
        exit_code = orchestrator.run_full_system(intraday_interval=args.interval)
    elif args.daily:
        exit_code = orchestrator.run_daily_pipeline(blocking=True)
    elif args.intraday:
        exit_code = orchestrator.run_intraday_pipeline(
            blocking=True,
            run_once=args.once,
            interval=args.interval
        )
    elif args.health:
        exit_code = orchestrator.run_health_check()
    elif args.update_stale:
        exit_code = orchestrator.update_stale_data(min_staleness=args.min_staleness)
    else:
        parser.print_help()
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
