import argparse
import logging
import os
import sys
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

file_path = Path(__file__).resolve()
project_root = file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from dotenv import load_dotenv
from config.settings import DB_PATH
from config.all_symbols import SYMBOLS
from config.logging_config import setup_logging
from data.tick_storage import TickStorage
from data.model_storage import ModelStorage
from strategies.risklabai_bot import RiskLabAIModel

load_dotenv(project_root / "config" / ".env", override=True)

logger = setup_logging(script_name="train_models")

ROLLING_WINDOW_DAYS = 90

def parse_args():
    parser = argparse.ArgumentParser(description="Train RiskLabAI Models")
    parser.add_argument("--symbol", help="Train a specific symbol")
    parser.add_argument("--all", action="store_true", help="Train all symbols")
    parser.add_argument("--no-s3", action="store_true", help="Skip S3 upload")
    return parser.parse_args()

def prune_old_data(symbol: str):
    cutoff_date = (datetime.now() - timedelta(days=ROLLING_WINDOW_DAYS)).date()
    cutoff_iso = cutoff_date.isoformat()
    logger.info(f"[{symbol}] Pruning data older than {cutoff_iso}...")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        safe_symbol = symbol.replace('.', '_').replace('-', '_')
        table_name = f"ticks_{safe_symbol}"
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if not cursor.fetchone(): return
        query = f"DELETE FROM {table_name} WHERE timestamp < ?"
        cursor.execute(query, (cutoff_iso,))
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        if deleted_count > 0: logger.info(f"[{symbol}] Deleted {deleted_count} old ticks.")
        else: logger.info(f"[{symbol}] No old ticks to prune.")
    except Exception as e: logger.warning(f"[{symbol}] Pruning failed: {e}")

def train_and_save(symbol: str, upload_to_s3: bool = True):
    logger.info("=" * 60)
    logger.info(f"PROCESSING: {symbol}")
    logger.info("=" * 60)
    
    prune_old_data(symbol)
    
    logger.info(f"[{symbol}] Training model on last {ROLLING_WINDOW_DAYS} days...")
    model = RiskLabAIModel()
    
    try:
        results = model.train_from_ticks(symbol, min_samples=100)
    except Exception as e:
        logger.error(f"[{symbol}] Training crashed: {e}")
        return False
        
    if not results.get('success'):
        logger.warning(f"[{symbol}] Training failed: {results.get('reason')}")
        return False
        
    logger.info(f"[{symbol}] Training Success!")
    
    payload = {
        "primary_model": model.primary_model,
        "meta_model": model.meta_model,
        "scaler": model.scaler,
        "label_encoder": model.label_encoder,
        "feature_names": model.feature_names,
        "training_metrics": results,
        "trained_at": datetime.now().isoformat(),
        "window_days": ROLLING_WINDOW_DAYS
    }
    
    local_models_path = project_root / "models"
    storage = ModelStorage(local_dir=local_models_path)
    
    try:
        storage.save_model(symbol, payload, upload_to_s3=upload_to_s3)
        return True
    except Exception as e:
        logger.error(f"[{symbol}] Save failed: {e}")
        return False

def main():
    args = parse_args()
    if args.symbol: symbols_to_train = [args.symbol.upper()]
    elif args.all: symbols_to_train = SYMBOLS
    else:
        logger.error("Please specify --symbol <TICKER> or --all")
        return 1
        
    success_count = 0
    fail_count = 0
    for symbol in symbols_to_train:
        if train_and_save(symbol, upload_to_s3=not args.no_s3): success_count += 1
        else: fail_count += 1
        
    logger.info("=" * 60)
    logger.info(f"BATCH COMPLETE: {success_count} Succeeded, {fail_count} Failed")
    logger.info("=" * 60)

if __name__ == "__main__":
    sys.exit(main())