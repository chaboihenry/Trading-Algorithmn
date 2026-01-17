import sys
import os
import signal
import logging
from datetime import datetime
from pathlib import Path

# Path Setup
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

from lumibot.brokers import Alpaca
from lumibot.traders import Trader
from strategies.risklabai_bot import RiskLabAIStrategy
from config.logging_config import setup_logging
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, DB_PATH
from config.all_symbols import SYMBOLS

# Configuration
IS_PAPER_TRADING = True

# Logging
logger = setup_logging(script_name="live_trading", log_dir="logs")

shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    if shutdown_requested:
        logger.warning("Force exiting...")
        sys.exit(1)
    logger.info("Shutdown requested. Finishing up...")
    shutdown_requested = True
    raise KeyboardInterrupt()

def check_requirements():
    db_file = Path(DB_PATH)
    if not db_file.exists() or db_file.stat().st_size == 0:
        logger.error(f"Tick database missing or empty at: {DB_PATH}")
        logger.error("Please run 'data/backfill_ticks.py' first.")
        return False
    return True

def main():
    logger.info("=" * 60)
    logger.info(" RISKLABAI LIVE TRADING BOT")
    logger.info("=" * 60)
    logger.info(f" Mode: {'PAPER' if IS_PAPER_TRADING else 'LIVE (REAL MONEY)'}")
    logger.info(f" Database: {DB_PATH}")
    logger.info(f" Symbols: {len(SYMBOLS)} loaded")
    logger.info("-" * 60)

    if not check_requirements():
        return

    # Configure Broker
    alpaca_config = {
        "API_KEY": ALPACA_API_KEY,
        "API_SECRET": ALPACA_SECRET_KEY,
        "PAPER": IS_PAPER_TRADING
    }
    broker = Alpaca(alpaca_config)

    # Initialize Trader & Strategy
    trader = Trader()
    strategy = RiskLabAIStrategy(broker=broker)
    trader.add_strategy(strategy)

    # Run
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("Starting strategy execution...")
    try:
        trader.run_all()
    except KeyboardInterrupt:
        logger.info("User stopped trading.")
    except Exception as e:
        logger.exception(f"Unexpected crash: {e}")
        raise
    finally:
        logger.info("Trading session ended.")

if __name__ == "__main__":
    main()