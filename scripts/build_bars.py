"""
Build Bars Utility (Debug/Verify)

Use this script to verify that your raw ticks can successfully be converted
into Imbalance Bars. It helps you tune the 'threshold' parameter.

Usage:
    python scripts/build_bars.py --symbol AAPL
    python scripts/build_bars.py --symbol AAPL --threshold 5000
    python scripts/build_bars.py --all
"""

import argparse
import logging
import sys
import pandas as pd
from pathlib import Path

# --- PATH SETUP ---
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- IMPORTS ---
from config.settings import DB_PATH
from config.all_symbols import SYMBOLS
from data.tick_storage import TickStorage
from data.tick_to_bars import generate_bars_from_ticks

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# --- DEFAULTS ---
DEFAULT_THRESHOLD = 3000

def parse_args():
    parser = argparse.ArgumentParser(description="Build Bars (Test)")
    parser.add_argument("--symbol", help="Symbol to process")
    parser.add_argument("--all", action="store_true", help="Process all symbols")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Imbalance threshold")
    return parser.parse_args()

def process_symbol(symbol, threshold):
    logger.info(f"--- Processing {symbol} ---")
    
    # 1. Load Ticks
    try:
        storage = TickStorage(DB_PATH)
        ticks = storage.load_ticks(symbol)
        storage.close()
    except Exception as e:
        logger.error(f"Failed to load DB: {e}")
        return

    if not ticks:
        logger.warning(f"[{symbol}] No ticks found in database.")
        return

    # 2. Generate Bars
    logger.info(f"[{symbol}] Loaded {len(ticks):,} ticks. Generating bars (Threshold={threshold})...")
    
    try:
        bars = generate_bars_from_ticks(ticks, threshold=threshold)
    except Exception as e:
        logger.error(f"[{symbol}] Bar generation failed: {e}")
        return

    if not bars:
        logger.warning(f"[{symbol}] Result: 0 bars generated. (Try lowering threshold?)")
        return

    # 3. Analysis
    df = pd.DataFrame(bars)
    
    # Check time range
    start_date = df['bar_start'].iloc[0] if 'bar_start' in df else "N/A"
    end_date = df['bar_end'].iloc[-1] if 'bar_end' in df else "N/A"
    
    avg_ticks = df['tick_count'].mean()
    
    logger.info(f"[{symbol}] SUCCESS:")
    logger.info(f"   Total Bars: {len(df):,}")
    logger.info(f"   Time Range: {start_date} -> {end_date}")
    logger.info(f"   Avg Ticks/Bar: {avg_ticks:.1f}")
    
    # Alert if data is too thin
    if len(df) < 50:
        logger.warning(f"   ⚠️  Low bar count ({len(df)}). Model training might fail.")

def main():
    args = parse_args()
    
    targets = []
    if args.symbol:
        targets = [args.symbol.upper()]
    elif args.all:
        targets = SYMBOLS
    else:
        logger.error("Please specify --symbol <TICKER> or --all")
        return

    for sym in targets:
        process_symbol(sym, args.threshold)

if __name__ == "__main__":
    main()