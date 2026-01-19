import sys
import os
import time
import sqlite3
import logging
import argparse
import threading
import queue
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, wait

from dotenv import load_dotenv
load_dotenv()

import requests
import requests.adapters
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockTradesRequest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from data.tick_storage import TickStorage
from utils.market_calendar import is_trading_day
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, DB_PATH
from config.all_symbols import SYMBOLS
from config.logging_config import setup_logging

logger = setup_logging(script_name="backfill_ticks")

# --- CONFIG ---
TARGET_WINDOW_DAYS = 365
# Increased Batch Size for faster throughput
BATCH_SIZE = 50000 
MAX_WORKERS = 4 

db_write_queue = queue.Queue()

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS, max_retries=3)
session.mount('https://', adapter)

client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
print_lock = threading.Lock()

class TradeDataAdapter:
    def __init__(self, trade):
        self.t = trade.timestamp
        self.p = trade.price
        self.s = trade.size
        self.c = trade.conditions
        self.x = trade.exchange

def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA journal_mode=WAL;") 
        # SPEED HACK: Don't wait for disk flush
        conn.execute("PRAGMA synchronous=OFF;") 
        conn.execute("PRAGMA cache_size=-64000;") # 64MB Cache
        conn.execute("""
            CREATE TABLE IF NOT EXISTS backfill_status (
                symbol TEXT PRIMARY KEY,
                status TEXT,
                last_backfilled_date TEXT
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"DB Init Warning: {e}")

def db_writer_worker():
    storage = TickStorage(DB_PATH)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=OFF;") # Critical for speed
    
    writes_since_checkpoint = 0
    
    while True:
        try:
            item = db_write_queue.get()
            if item is None: break
            
            task_type, payload = item
            
            if task_type == 'TICKS':
                symbol, batch = payload
                storage.save_ticks(symbol, batch)
                writes_since_checkpoint += 1
                
            elif task_type == 'STATUS':
                symbol, date_str = payload
                with conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO backfill_status (symbol, status, last_backfilled_date) VALUES (?, ?, ?)",
                        (symbol, 'active', date_str)
                    )
            
            # Less frequent checkpointing (every 50 batches instead of 20)
            if writes_since_checkpoint >= 50:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
                writes_since_checkpoint = 0
            
            db_write_queue.task_done()
            
        except Exception as e:
            with print_lock:
                print(f"❌ DB Writer Error: {e}")
    
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);") 
    conn.close()
    print("💾 DB Writer Thread Finished.")

def check_date_exists(conn, symbol, date_obj):
    safe_symbol = symbol.replace('.', '_').replace('-', '_')
    table_name = f"ticks_{safe_symbol}"
    start_ts = date_obj.strftime("%Y-%m-%d")
    next_day = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if not cursor.fetchone(): return False
        query = f"SELECT 1 FROM {table_name} WHERE timestamp >= ? AND timestamp < ? LIMIT 1"
        cursor.execute(query, (start_ts, next_day))
        return cursor.fetchone() is not None
    except Exception:
        return False

def backfill_symbol(symbol):
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=TARGET_WINDOW_DAYS)
    read_conn = sqlite3.connect(DB_PATH, timeout=30)
    
    current_date = start_date
    skipped_count = 0
    
    while current_date <= end_date:
        if not is_trading_day(current_date):
            current_date += timedelta(days=1)
            continue

        date_str = current_date.isoformat()
        
        if check_date_exists(read_conn, symbol, current_date):
            skipped_count += 1
            current_date += timedelta(days=1)
            continue
        
        if skipped_count > 0:
            with print_lock:
                print(f"[{symbol}] Skipped {skipped_count} existing days. Downloading {date_str}...")
            skipped_count = 0
            
        try:
            request = StockTradesRequest(
                symbol_or_symbols=symbol,
                start=current_date,
                end=current_date + timedelta(days=1),
                limit=None
            )
            trades_generator = client.get_stock_trades(request)
            
            trades_list = []
            if hasattr(trades_generator, 'data'):
                trades_list = trades_generator.data.get(symbol, [])
            elif isinstance(trades_generator, dict):
                trades_list = trades_generator.get(symbol, [])

            if not trades_list:
                current_date += timedelta(days=1)
                continue

            batch = []
            day_total = 0
            
            for trade in trades_list:
                wrapper = TradeDataAdapter(trade)
                batch.append(wrapper)
                
                if len(batch) >= BATCH_SIZE:
                    db_write_queue.put(('TICKS', (symbol, list(batch))))
                    day_total += len(batch)
                    batch = []

            if batch:
                db_write_queue.put(('TICKS', (symbol, list(batch))))
                day_total += len(batch)

            db_write_queue.put(('STATUS', (symbol, date_str)))

            with print_lock:
                # Less verbose: Only print total for the day
                print(f"[{symbol}] {date_str}: +{day_total} ticks")
            
        except Exception as e:
            if "429" in str(e):
                with print_lock:
                    print(f"[{symbol}] ⚠️ Rate Limit. Sleeping 20s...")
                time.sleep(20)
            else:
                with print_lock:
                    print(f"[{symbol}] ❌ Error on {date_str}: {e}")
                time.sleep(1)
        
        current_date += timedelta(days=1)
    
    read_conn.close()
    return f"[{symbol}] ✅ Complete."

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", help="Backfill single symbol")
    parser.add_argument("--all", action="store_true", help="Backfill ALL symbols")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Parallel workers")
    args = parser.parse_args()

    init_db()
    
    writer_thread = threading.Thread(target=db_writer_worker, daemon=True)
    writer_thread.start()
    
    targets = []
    if args.symbol:
        targets = [args.symbol.upper()]
    elif args.all:
        targets = SYMBOLS
    else:
        logger.error("Please specify --symbol or --all")
        return

    logger.info(f"Starting Speed Backfill (Synchronous=OFF)")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(backfill_symbol, sym): sym for sym in targets}
        wait(futures)

    db_write_queue.put(None)
    writer_thread.join()
    print("All tasks complete.")

if __name__ == "__main__":
    main()