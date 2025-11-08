"""
Automated Daily Scheduler for Pairs Paper Trading
================================================

This script runs the pairs paper trading system automatically on a daily schedule.
It executes trades, monitors positions, and updates the portfolio dashboard.

Features:
- Runs trading cycle once per day at specified time
- Automatic position monitoring and management
- Error handling and logging
- Can run continuously or as a scheduled task

Usage:
    python run_daily_trading.py [--time HH:MM] [--run-once]

    --time HH:MM    : Time to run daily (default: 09:30)
    --run-once      : Run immediately once and exit (for cron/scheduled tasks)
"""

import argparse
import logging
import schedule
import time
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from paper_trading.pairs_paper_trading import PairsPaperTrader

# Configure logging
log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(exist_ok=True)
log_file = log_dir / 'daily_trading.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DailyTradingScheduler:
    """Automated scheduler for daily pairs trading execution"""

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db"):
        """
        Initialize the daily trading scheduler

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.trader = None

    def run_trading_cycle(self):
        """Execute one complete trading cycle"""
        try:
            logger.info("=" * 80)
            logger.info(f"Starting scheduled trading cycle - {datetime.now()}")
            logger.info("=" * 80)

            # Initialize trader
            self.trader = PairsPaperTrader(db_path=self.db_path)

            # Run the trading cycle
            self.trader.run_trading_cycle()

            # Get portfolio summary
            portfolio = self.trader.get_portfolio_summary()

            logger.info("\n" + "=" * 80)
            logger.info("DAILY TRADING CYCLE COMPLETED")
            logger.info("=" * 80)
            logger.info(f"Total Capital: ${portfolio['total_capital']:,.2f}")
            logger.info(f"Cumulative P&L: ${portfolio['cumulative_pnl']:,.2f} ({portfolio['return_pct']:.2f}%)")
            logger.info(f"Open Positions: {portfolio['open_positions']}")
            logger.info(f"Closed Positions: {portfolio['closed_positions']}")
            logger.info("=" * 80)

            return True

        except Exception as e:
            logger.error(f"Error during trading cycle: {str(e)}", exc_info=True)
            return False

    def run_once(self):
        """Run trading cycle once and exit"""
        logger.info("Running trading cycle once (immediate execution)...")
        success = self.run_trading_cycle()
        if success:
            logger.info("Trading cycle completed successfully")
            return 0
        else:
            logger.error("Trading cycle failed")
            return 1

    def run_scheduled(self, run_time: str = "09:30"):
        """
        Run trading cycle on a daily schedule

        Args:
            run_time: Time to run daily in HH:MM format (default: 09:30)
        """
        logger.info(f"Starting automated daily trading scheduler")
        logger.info(f"Scheduled time: {run_time} daily")
        logger.info("Press Ctrl+C to stop")

        # Schedule the daily job
        schedule.every().day.at(run_time).do(self.run_trading_cycle)

        # Keep the scheduler running
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute

        except KeyboardInterrupt:
            logger.info("\nScheduler stopped by user")
            sys.exit(0)


def main():
    """Main entry point for the scheduler"""
    parser = argparse.ArgumentParser(
        description='Automated Daily Pairs Trading Scheduler',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run continuously, execute at 9:30 AM daily
  python run_daily_trading.py --time 09:30

  # Run once immediately (for cron jobs)
  python run_daily_trading.py --run-once

  # Run at market open (9:30 AM Eastern)
  python run_daily_trading.py --time 09:30

  # Run at market close (4:00 PM Eastern)
  python run_daily_trading.py --time 16:00
        """
    )

    parser.add_argument(
        '--time',
        type=str,
        default='09:30',
        help='Time to run daily in HH:MM format (default: 09:30)'
    )

    parser.add_argument(
        '--run-once',
        action='store_true',
        help='Run immediately once and exit (for cron/scheduled tasks)'
    )

    parser.add_argument(
        '--db-path',
        type=str,
        default='/Volumes/Vault/85_assets_prediction.db',
        help='Path to database (default: /Volumes/Vault/85_assets_prediction.db)'
    )

    args = parser.parse_args()

    # Initialize scheduler
    scheduler = DailyTradingScheduler(db_path=args.db_path)

    # Run once or continuously
    if args.run_once:
        sys.exit(scheduler.run_once())
    else:
        scheduler.run_scheduled(run_time=args.time)


if __name__ == "__main__":
    main()