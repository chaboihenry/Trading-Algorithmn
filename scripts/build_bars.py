import argparse
import logging
import sys
import pandas as pd
from pathlib import Path

file_path = Path(__file__).resolve()
project_root = file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.settings import DB_PATH
from config.all_symbols import SYMBOLS
from config.logging_config import setup_logging
from data.tick_storage import TickStorage
from data.tick_to_bars import ImbalanceBarGenerator

logger = setup_logging(script_name="build_bars")

DEFAULT_THRESHOLD = 3000

def parse_args():
    parser = argparse.ArgumentParser(description="Build Bars (Test)")
    parser.add_argument("--symbol", help="Symbol to process")
    parser.add_argument("--all", action="store_true", help="Process all symbols")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Imbalance threshold")
    return parser.parse_args()

def process_symbol(symbol, threshold):
    logger.info(f"--- Processing {symbol} ---")
    
    try:
        storage = TickStorage(DB_PATH)
        ticks = storage.load_ticks(symbol)
        storage.close()
    except Exception as e:
        logger.error(f"Failed to load DB: {e}")
        return

    if ticks.empty:
        logger.warning(f"[{symbol}] No ticks found in database.")
        return

    logger.info(f"[{symbol}] Loaded {len(ticks):,} ticks. Generating bars (Threshold={threshold})...")
    
    try:
        bars = ImbalanceBarGenerator.process_ticks(ticks, threshold=threshold)
    except Exception as e:
        logger.error(f"[{symbol}] Bar generation failed: {e}")
        return

    if bars.empty:
        logger.warning(f"[{symbol}] Result: 0 bars generated.")
        return

    avg_ticks = bars['tick_count'].mean()
    start_date = bars.index[0]
    end_date = bars.index[-1]
    
    logger.info(f"[{symbol}] SUCCESS:")
    logger.info(f"   Total Bars: {len(bars):,}")
    logger.info(f"   Time Range: {start_date} -> {end_date}")
    logger.info(f"   Avg Ticks/Bar: {avg_ticks:.1f}")

def main():
    args = parse_args()
    targets = []
    if args.symbol: targets = [args.symbol.upper()]
    elif args.all: targets = SYMBOLS
    else:
        logger.error("Please specify --symbol <TICKER> or --all")
        return

    for sym in targets:
        process_symbol(sym, args.threshold)

if __name__ == "__main__":
    main()