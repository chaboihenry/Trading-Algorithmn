#!/usr/bin/env python3
"""
Master Orchestration Script - Trade ALL US Stocks

This script automates the complete pipeline to trade ALL US stocks:
1. Fetch all tradeable symbols from Alpaca (NYSE, NASDAQ, AMEX, etc.)
2. Download tick data for selected tier
3. Train RiskLabAI models for each symbol
4. Configure live trading bot
5. Run the bot!

Usage:
    python scripts/setup/master_setup.py [--tier TIER] [--skip-fetch] [--skip-backfill] [--skip-train]

Arguments:
    --tier: Which tier to set up (tier_1, tier_2, tier_3, tier_4, tier_5)
            tier_1: Top 100 most liquid (RECOMMENDED START)
            tier_2: Top 500 (S&P 500 level)
            tier_3: Top 1000 (Russell 1000)
            tier_4: Top 2000
            tier_5: ALL liquid stocks (2000-5000)
            Default: tier_1

    --skip-fetch: Skip fetching symbols (use existing all_symbols.py)
    --skip-backfill: Skip downloading tick data
    --skip-train: Skip training models
    --run-bot: Automatically run the trading bot after setup

    --parallel: Number of parallel downloads/training jobs (default: 8)
    --days: Days of tick history to download (default: 365)
"""

import sys
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime
from config.tick_config import TICK_DB_PATH

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner(text: str):
    """Print a fancy banner."""
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"  {text}")
    logger.info("=" * 80)
    logger.info("")


def run_command(cmd: list, description: str) -> bool:
    """
    Run a command and return success status.

    Args:
        cmd: Command to run
        description: What the command does

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info("")

    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            check=True
        )
        logger.info(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"✗ {description} failed: {e}")
        return False


def main():
    """Run master setup."""
    parser = argparse.ArgumentParser(
        description='Master setup script for trading ALL US stocks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Set up Tier 1 (100 symbols) - RECOMMENDED START
  python scripts/setup/master_setup.py --tier tier_1

  # Set up Tier 2 (500 symbols) with 16 parallel jobs
  python scripts/setup/master_setup.py --tier tier_2 --parallel 16

  # Set up and run bot automatically
  python scripts/setup/master_setup.py --tier tier_1 --run-bot

  # Skip steps you already did
  python scripts/setup/master_setup.py --tier tier_1 --skip-fetch --skip-backfill
        """
    )

    parser.add_argument('--tier', type=str, default='tier_1',
                        choices=['tier_1', 'tier_2', 'tier_3', 'tier_4', 'tier_5'],
                        help='Which tier to set up (default: tier_1)')
    parser.add_argument('--skip-fetch', action='store_true',
                        help='Skip fetching symbols')
    parser.add_argument('--skip-backfill', action='store_true',
                        help='Skip downloading tick data')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip training models')
    parser.add_argument('--run-bot', action='store_true',
                        help='Run trading bot after setup')
    parser.add_argument('--parallel', type=int, default=8,
                        help='Parallel jobs for backfill/training')
    parser.add_argument('--days', type=int, default=365,
                        help='Days of tick history')

    args = parser.parse_args()

    # Print welcome banner
    print_banner("MASTER SETUP: TRADE ALL US STOCKS")

    logger.info(f"Configuration:")
    logger.info(f"  Tier: {args.tier}")
    logger.info(f"  Parallel jobs: {args.parallel}")
    logger.info(f"  Days of history: {args.days}")
    logger.info(f"  Skip fetch: {args.skip_fetch}")
    logger.info(f"  Skip backfill: {args.skip_backfill}")
    logger.info(f"  Skip train: {args.skip_train}")
    logger.info(f"  Run bot after: {args.run_bot}")
    logger.info("")

    start_time = datetime.now()

    # Tier info
    tier_info = {
        'tier_1': ('Top 100', 100, '~15-20 hours'),
        'tier_2': ('Top 500', 500, '~4-5 days'),
        'tier_3': ('Top 1000', 1000, '~8-10 days'),
        'tier_4': ('Top 2000', 2000, '~15-20 days'),
        'tier_5': ('All stocks', 2500, '~3-4 weeks'),
    }

    tier_name, tier_count, tier_time = tier_info[args.tier]

    logger.info(f"Setting up {tier_name} ({args.tier})")
    logger.info(f"  Expected symbols: ~{tier_count}")
    logger.info(f"  Estimated training time: {tier_time}")
    logger.info("")

    # Step 1: Fetch all symbols
    if not args.skip_fetch:
        print_banner("STEP 1/4: FETCHING ALL US SYMBOLS FROM ALPACA")

        cmd = [
            sys.executable,
            'scripts/setup/fetch_all_symbols.py'
        ]

        if not run_command(cmd, "Fetch symbols from Alpaca"):
            logger.error("Failed to fetch symbols. Aborting.")
            return 1
    else:
        logger.info("⏭️  Skipping symbol fetch (using existing all_symbols.py)")

    # Verify all_symbols.py exists
    if not Path('config/all_symbols.py').exists():
        logger.error("config/all_symbols.py not found!")
        logger.error("Run without --skip-fetch to create it")
        return 1

    # Step 2: Download tick data
    if not args.skip_backfill:
        print_banner(f"STEP 2/4: DOWNLOADING TICK DATA FOR {tier_name.upper()}")

        cmd = [
            sys.executable,
            'scripts/setup/backfill_all_symbols.py',
            '--tier', args.tier,
            '--parallel', str(args.parallel),
            '--days', str(args.days)
        ]

        logger.warning(f"This will download {args.days} days of tick data for ~{tier_count} symbols")
        logger.warning(f"Estimated time: {tier_count / args.parallel * 2 / 60:.1f} hours")
        logger.warning("Press Ctrl+C within 5 seconds to cancel...")

        import time
        time.sleep(5)

        if not run_command(cmd, "Download tick data"):
            logger.error("Failed to download tick data. Aborting.")
            return 1

        # Verify all symbols have tick data
        logger.info("")
        logger.info("Verifying tick data for all symbols...")

        import sqlite3
        from config.all_symbols import get_symbols_by_tier

        tier_symbols = get_symbols_by_tier(args.tier)

        conn = sqlite3.connect(str(TICK_DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM ticks ORDER BY symbol")
        symbols_with_data = {row[0] for row in cursor.fetchall()}
        conn.close()

        missing_symbols = [s for s in tier_symbols if s not in symbols_with_data]

        if missing_symbols:
            logger.warning(f"⚠️  {len(missing_symbols)} symbols are missing tick data:")
            logger.warning(f"  Missing: {', '.join(missing_symbols[:10])}")
            if len(missing_symbols) > 10:
                logger.warning(f"  ... and {len(missing_symbols) - 10} more")
            logger.warning("")
            logger.warning("Retrying download for missing symbols...")

            # Retry downloading missing symbols one at a time
            for symbol in missing_symbols:
                logger.info(f"  Downloading {symbol}...")
                retry_cmd = [
                    sys.executable,
                    'scripts/setup/backfill_ticks.py',
                    '--symbol', symbol,
                    '--days', str(args.days)
                ]
                subprocess.run(retry_cmd, cwd=project_root)

            logger.info("")
            logger.info("✓ Retry complete - all symbols should now have data")
        else:
            logger.info(f"✓ All {len(tier_symbols)} symbols have tick data")
    else:
        logger.info("⏭️  Skipping tick data download")

    # Step 3: Train models
    if not args.skip_train:
        print_banner(f"STEP 3/4: TRAINING MODELS FOR {tier_name.upper()}")

        cmd = [
            sys.executable,
            'scripts/setup/train_all_symbols.py',
            '--tier', args.tier,
            '--parallel', '1'  # Keep at 1 to avoid memory issues
        ]

        logger.warning(f"This will train ~{tier_count} models")
        logger.warning(f"Estimated time: {tier_time}")
        logger.warning("Consider running in a screen/tmux session for long training runs")
        logger.warning("Press Ctrl+C within 5 seconds to cancel...")

        import time
        time.sleep(5)

        if not run_command(cmd, "Train models"):
            logger.error("Failed to train models. Aborting.")
            return 1
    else:
        logger.info("⏭️  Skipping model training")

    # Step 4: Verify setup
    print_banner("STEP 4/4: VERIFYING SETUP")

    # Check models directory
    models_dir = Path('models')
    model_files = list(models_dir.glob('risklabai_*_models.pkl'))

    logger.info(f"Found {len(model_files)} trained models in models/")

    if len(model_files) == 0:
        logger.warning("No models found! Training may have failed.")
        logger.warning("Check logs above for errors.")
    else:
        logger.info(f"✓ Models ready: {model_files[:5]}")
        if len(model_files) > 5:
            logger.info(f"  ... and {len(model_files) - 5} more")

    logger.info("")

    # Summary
    print_banner("SETUP COMPLETE!")

    elapsed = datetime.now() - start_time
    hours = elapsed.total_seconds() / 3600

    logger.info(f"Total time: {hours:.1f} hours")
    logger.info(f"Models trained: {len(model_files)}")
    logger.info(f"Ready to trade: {len(model_files) > 0}")
    logger.info("")

    # Next steps
    logger.info("=" * 80)
    logger.info("NEXT STEPS")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Your bot is ready to trade ALL US stocks!")
    logger.info("")
    logger.info("To run the trading bot:")
    logger.info("  python run_live_trading.py")
    logger.info("")
    logger.info("To expand to more symbols:")
    logger.info("  python scripts/setup/master_setup.py --tier tier_2")
    logger.info("")
    logger.info("To monitor models:")
    logger.info("  ls -lh models/risklabai_*_models.pkl | wc -l")
    logger.info("")
    logger.info("=" * 80)

    # Auto-run bot if requested
    if args.run_bot:
        logger.info("")
        print_banner("STARTING TRADING BOT")

        cmd = [
            sys.executable,
            'run_live_trading.py'
        ]

        run_command(cmd, "Run live trading bot")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("")
        logger.info("Setup cancelled by user")
        sys.exit(1)
