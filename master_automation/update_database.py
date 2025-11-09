#!/usr/bin/env python3
"""
Database Update Script for Trading Algorithm
=============================================

This script updates all data in the SQLite database by running all data collectors
and preprocessing scripts in the correct order.

Features:
- Checks if external database drive is connected
- Runs all data collection scripts sequentially
- Runs all preprocessing scripts
- Generates trading signals
- Comprehensive error handling and logging
- Progress tracking

Usage:
    python update_database.py [--wait-for-drive]

    --wait-for-drive: Wait for external drive to be connected (default: fail immediately)
"""

import subprocess
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
import os

# Configure logging
log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(exist_ok=True)
log_file = log_dir / 'database_update.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
PYTHON_PATH = "/Users/henry/miniconda3/envs/trading/bin/python"
DB_PATH = "/Volumes/Vault/85_assets_prediction.db"
PROJECT_DIR = Path(__file__).parent

# Data collection scripts in order
DATA_COLLECTORS = [
    "data_collectors/01_collect_assets.py",
    "data_collectors/02_collect_price_data.py",
    "data_collectors/03_collect_fundamentals.py",
    "data_collectors/03b_backfill_fundamentals_quarterly.py",
    "data_collectors/04_collect_economic_indicators.py",
    "data_collectors/05_collect_sentiment.py",
    "data_collectors/06_collect_earnings.py",
    "data_collectors/07_collect_insider_trades.py",
    "data_collectors/08_collect_analyst_ratings.py",
    "data_collectors/09_collect_options_data.py",
    "data_collectors/10_collect_news_events.py",
]

# Preprocessing scripts in order
PREPROCESSING_SCRIPTS = [
    "data_preprocessing/01_calculate_correlations.py",
    "data_preprocessing/02_calculate_technical_indicators.py",
    "data_preprocessing/03_calculate_volatility.py",
    "data_preprocessing/04_analyze_pairs.py",
    "data_preprocessing/05_ml_features_aggregator.py",
]

# Signal generation scripts
SIGNAL_GENERATION_SCRIPTS = [
    "paper_trading/01_signal_generator.py",
]


def check_database_available(wait: bool = False, max_wait_minutes: int = 60) -> bool:
    """
    Check if database is accessible

    Args:
        wait: If True, wait for database to become available
        max_wait_minutes: Maximum time to wait in minutes

    Returns:
        True if database is accessible, False otherwise
    """
    db_path = Path(DB_PATH)

    if db_path.exists():
        logger.info(f"✓ Database found at {DB_PATH}")
        return True

    if not wait:
        logger.error(f"✗ Database not found at {DB_PATH}")
        logger.error("External drive 'Vault' may not be connected")
        return False

    # Wait for database
    logger.warning(f"⏳ Database not found. Waiting for external drive to connect...")
    logger.warning(f"   Will wait up to {max_wait_minutes} minutes")

    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    check_interval = 10  # Check every 10 seconds

    while time.time() - start_time < max_wait_seconds:
        if db_path.exists():
            logger.info(f"✓ Database connected after {int(time.time() - start_time)} seconds")
            return True

        elapsed = int(time.time() - start_time)
        remaining = int(max_wait_seconds - (time.time() - start_time))
        logger.info(f"   Still waiting... ({elapsed}s elapsed, {remaining}s remaining)")
        time.sleep(check_interval)

    logger.error(f"✗ Database not available after {max_wait_minutes} minutes")
    return False


def run_script(script_path: str, description: str) -> bool:
    """
    Run a Python script and log output

    Args:
        script_path: Path to script relative to project directory
        description: Description of what the script does

    Returns:
        True if successful, False otherwise
    """
    full_path = PROJECT_DIR / script_path

    if not full_path.exists():
        logger.warning(f"⚠ Script not found: {script_path} - Skipping")
        return True  # Don't fail the whole process

    logger.info(f"\n{'='*80}")
    logger.info(f"Running: {description}")
    logger.info(f"Script: {script_path}")
    logger.info(f"{'='*80}")

    try:
        result = subprocess.run(
            [PYTHON_PATH, str(full_path)],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per script
        )

        if result.stdout:
            logger.info(f"Output:\n{result.stdout}")

        if result.returncode == 0:
            logger.info(f"✓ {description} completed successfully")
            return True
        else:
            logger.error(f"✗ {description} failed with exit code {result.returncode}")
            if result.stderr:
                logger.error(f"Error output:\n{result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"✗ {description} timed out after 10 minutes")
        return False
    except Exception as e:
        logger.error(f"✗ {description} failed with error: {str(e)}")
        return False


def update_database(wait_for_drive: bool = False) -> bool:
    """
    Update all database tables with latest data

    Args:
        wait_for_drive: Wait for external drive if not connected

    Returns:
        True if all updates successful, False otherwise
    """
    logger.info("\n" + "="*80)
    logger.info("DATABASE UPDATE STARTED")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80 + "\n")

    # Check database availability
    if not check_database_available(wait=wait_for_drive, max_wait_minutes=60):
        logger.error("Database update failed: Database not accessible")
        return False

    total_scripts = len(DATA_COLLECTORS) + len(PREPROCESSING_SCRIPTS) + len(SIGNAL_GENERATION_SCRIPTS)
    completed = 0
    failed = 0

    # Run data collectors
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: DATA COLLECTION")
    logger.info("="*80 + "\n")

    for script in DATA_COLLECTORS:
        script_name = Path(script).stem
        description = f"Data Collection: {script_name}"

        if run_script(script, description):
            completed += 1
        else:
            failed += 1
            logger.warning(f"⚠ Continuing despite failure in {script_name}")

    # Run preprocessing
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: DATA PREPROCESSING")
    logger.info("="*80 + "\n")

    for script in PREPROCESSING_SCRIPTS:
        script_name = Path(script).stem
        description = f"Preprocessing: {script_name}"

        if run_script(script, description):
            completed += 1
        else:
            failed += 1
            logger.warning(f"⚠ Continuing despite failure in {script_name}")

    # Run signal generation
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: SIGNAL GENERATION")
    logger.info("="*80 + "\n")

    for script in SIGNAL_GENERATION_SCRIPTS:
        script_name = Path(script).stem
        description = f"Signal Generation: {script_name}"

        if run_script(script, description):
            completed += 1
        else:
            failed += 1
            logger.warning(f"⚠ Continuing despite failure in {script_name}")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("DATABASE UPDATE COMPLETED")
    logger.info("="*80)
    logger.info(f"Total scripts: {total_scripts}")
    logger.info(f"Completed: {completed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {(completed/total_scripts)*100:.1f}%")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80 + "\n")

    return failed == 0


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Update trading algorithm database with latest data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update immediately (fail if drive not connected)
  python update_database.py

  # Wait for drive to be connected (up to 60 minutes)
  python update_database.py --wait-for-drive

  # For automated execution via cron/LaunchAgent
  python update_database.py --wait-for-drive
        """
    )

    parser.add_argument(
        '--wait-for-drive',
        action='store_true',
        help='Wait for external drive to be connected (max 60 minutes)'
    )

    args = parser.parse_args()

    success = update_database(wait_for_drive=args.wait_for_drive)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
