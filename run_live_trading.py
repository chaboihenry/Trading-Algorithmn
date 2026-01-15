import sys
import os
import logging
import signal
from datetime import datetime
from pathlib import Path

# --- PATH SETUP ---
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

# --- IMPORTS ---
from lumibot.brokers import Alpaca
from lumibot.traders import Trader
from strategies.risklabai_bot import RiskLabAICombined

# New Config Imports
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, DB_PATH
from config.all_symbols import SYMBOLS

# --- CONFIGURATION ---
# Trading Mode
IS_PAPER_TRADING = True

# Strategy Parameters
PROFIT_TARGET = 2.5        # 2.5x volatility
STOP_LOSS = 2.5            # 2.5x volatility
MAX_HOLDING_BARS = 10      # Max bars to hold a position
META_THRESHOLD = 0.001     # Meta-model confidence threshold
PROB_THRESHOLD = 0.015     # Primary model probability threshold

# Risk Management
DAILY_LOSS_LIMIT = 0.03    # 3% max daily loss
MAX_DRAWDOWN = 0.10        # 10% hard stop
KELLY_FRACTION = 0.5       # Half-Kelly size

# Logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f'live_trading_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Graceful Shutdown Globals
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
    """Verifies that the database and models exist."""
    db_file = Path(DB_PATH)
    if not db_file.exists() or db_file.stat().st_size == 0:
        logger.error(f"Tick database missing or empty at: {DB_PATH}")
        logger.error("Please run 'data/backfill_ticks.py' first.")
        return False
    return True

def main():
    print("=" * 60)
    print(" RISKLABAI LIVE TRADING BOT")
    print("=" * 60)
    print(f" Mode: {'PAPER' if IS_PAPER_TRADING else 'LIVE (REAL MONEY)'}")
    print(f" Database: {DB_PATH}")
    print(f" Symbols: {len(SYMBOLS)} loaded")
    print("-" * 60)

    if not check_requirements():
        return

    # 1. Configure Broker
    alpaca_config = {
        "API_KEY": ALPACA_API_KEY,
        "API_SECRET": ALPACA_SECRET_KEY,
        "PAPER": IS_PAPER_TRADING
    }
    broker = Alpaca(alpaca_config)

    # 2. Configure Strategy
    # We inject these parameters directly into the class
    RiskLabAICombined.parameters = {
        'symbols': SYMBOLS,
        'profit_taking': PROFIT_TARGET,
        'stop_loss': STOP_LOSS,
        'max_holding': MAX_HOLDING_BARS,
        'd': None, # Auto-calculated
        
        # Thresholds
        'meta_threshold': META_THRESHOLD,
        'prob_threshold': PROB_THRESHOLD,
        
        # Risk Settings
        'daily_loss_limit_pct': DAILY_LOSS_LIMIT,
        'max_drawdown_pct': MAX_DRAWDOWN,
        'use_kelly_sizing': True,
        'kelly_fraction': KELLY_FRACTION,
        
        # Misc
        'enable_profitability_tracking': True,
        'min_training_bars': 100,
        'retrain_days': 30
    }

    # 3. Initialize Trader & Strategy
    trader = Trader()
    strategy = RiskLabAICombined(broker=broker)
    trader.add_strategy(strategy)

    # 4. Run
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("Starting strategy execution...")
    try:
        trader.run_all()
    except KeyboardInterrupt:
        logger.info("User stopped trading.")
    except Exception as e:
        logger.error(f"Unexpected crash: {e}")
        raise
    finally:
        logger.info("Trading session ended.")

if __name__ == "__main__":
    main()