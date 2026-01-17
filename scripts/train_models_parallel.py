import logging
import sys
import os
import concurrent.futures
from pathlib import Path

# Path Setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

from config.all_symbols import SYMBOLS
from scripts.train_models import train_and_save

# Logging setup for the main process
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

MAX_WORKERS = os.cpu_count()  # Uses all available CPU cores

def train_worker(symbol):
    """
    Worker function to train a single symbol.
    """
    try:
        # We assume train_and_save handles its own logging/errors
        success = train_and_save(symbol, upload_to_s3=True)
        return symbol, success
    except Exception as e:
        return symbol, False

def run_parallel_training():
    logger.info(f"🚀 Starting Parallel Training with {MAX_WORKERS} cores...")
    
    success_count = 0
    fail_count = 0
    
    # ProcessPoolExecutor for CPU-bound tasks (Training)
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(train_worker, sym): sym for sym in SYMBOLS}
        
        for future in concurrent.futures.as_completed(futures):
            sym = futures[future]
            try:
                symbol, status = future.result()
                if status:
                    logger.info(f"✅ {symbol} Completed Successfully")
                    success_count += 1
                else:
                    logger.error(f"❌ {symbol} Failed")
                    fail_count += 1
            except Exception as e:
                logger.error(f"❌ {sym} Crashed: {e}")
                fail_count += 1

    logger.info("=" * 40)
    logger.info(f"PARALLEL BATCH COMPLETE: {success_count} Succeeded, {fail_count} Failed")
    logger.info("=" * 40)

if __name__ == "__main__":
    run_parallel_training()

    