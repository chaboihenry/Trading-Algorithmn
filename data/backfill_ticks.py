import sys
import os
import glob
import time
import sqlite3
import logging
import argparse
import threading
import queue
import pandas as pd
import boto3
import s3fs
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import date, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait

# --- 1. LOAD CREDENTIALS ---
from dotenv import load_dotenv
project_root = Path(__file__).resolve().parent.parent
env_path = project_root / "config" / ".env"
load_dotenv(dotenv_path=env_path, override=True)

if not os.getenv('AWS_SECRET_ACCESS_KEY'):
    print("❌ CRITICAL: API Keys NOT found. Check config/.env path!")
    sys.exit(1)

# --- IMPORTS ---
import requests
import requests.adapters
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockTradesRequest

# --- PATH SETUP ---
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
BATCH_SIZE = 50000 
MAX_WORKERS = 4 
CHUNK_SIZE = 100000 

S3_BUCKET_NAME = "risklabai-models"

# Queue to handle backpressure
db_write_queue = queue.Queue(maxsize=20) 

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS, max_retries=3)
session.mount('https://', adapter)

client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
print_lock = threading.Lock()

# --- CRITICAL FIX: DISABLE S3 CACHE TO PREVENT DEADLOCKS ---
s3_fs = s3fs.S3FileSystem(
    key=os.getenv('AWS_ACCESS_KEY_ID'),
    secret=os.getenv('AWS_SECRET_ACCESS_KEY'),
    use_listings_cache=False 
)

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
        conn.execute("PRAGMA synchronous=OFF;") 
        conn.execute("PRAGMA cache_size=-64000;") 
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

def cleanup_temp_files():
    data_dir = os.path.join(project_root, "data")
    if not os.path.exists(data_dir): return
    temp_files = glob.glob(os.path.join(data_dir, "*_temp.parquet"))
    if temp_files:
        logger.info(f"Cleaning up {len(temp_files)} leftover temp files...")
        for f in temp_files:
            try:
                os.remove(f)
            except Exception:
                pass

def db_writer_worker():
    storage = TickStorage(DB_PATH)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=OFF;")
    
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
                symbol, status_val = payload
                if isinstance(status_val, tuple):
                    stat, date_str = status_val
                else:
                    stat, date_str = 'active', status_val

                with conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO backfill_status (symbol, status, last_backfilled_date) VALUES (?, ?, ?)",
                        (symbol, stat, date_str)
                    )
            
            elif task_type == 'DROP_TABLE':
                symbol = payload
                safe_symbol = symbol.replace('.', '_').replace('-', '_')
                try:
                    with conn:
                        conn.execute(f"DROP TABLE IF EXISTS ticks_{safe_symbol}")
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                except Exception as e:
                    pass

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

# --- CHECK S3 (Non-Cached Version) ---
def check_s3_exists(symbol):
    """Returns True if a valid Parquet file exists in S3."""
    s3_path = f"{S3_BUCKET_NAME}/{symbol}/parquet/ticks.parquet"
    try:
        # Direct check, no cache lookups
        if s3_fs.exists(s3_path):
            # Verify it's not an empty placeholder
            size = s3_fs.du(s3_path)
            if size > 1024 * 1024: # 1MB
                return True
    except Exception:
        pass
    return False

def check_date_exists(conn, symbol, date_obj):
    safe_symbol = symbol.replace('.', '_').replace('-', '_')
    table_name = f"ticks_{safe_symbol}"
    start_ts = date_obj.strftime("%Y-%m-%d")
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if not cursor.fetchone(): return False
        query = f"SELECT 1 FROM {table_name} WHERE timestamp >= ? LIMIT 1"
        cursor.execute(query, (start_ts,))
        return cursor.fetchone() is not None
    except Exception:
        return False

def export_to_parquet_stream(symbol):
    try:
        safe_symbol = symbol.replace('.', '_').replace('-', '_')
        table_name = f"ticks_{safe_symbol}"
        start_date = date.today() - timedelta(days=TARGET_WINDOW_DAYS)
        start_ts = start_date.strftime("%Y-%m-%d")
        
        s3_path = f"{S3_BUCKET_NAME}/{symbol}/parquet/ticks.parquet"
        
        with print_lock:
            print(f"[{symbol}] ☁️ Streaming to S3... ", end="", flush=True)

        conn = sqlite3.connect(DB_PATH, timeout=60)
        query = f"SELECT * FROM {table_name} WHERE timestamp >= '{start_ts}'"
        
        chunk_iter = pd.read_sql_query(query, conn, chunksize=CHUNK_SIZE)
        
        with s3_fs.open(s3_path, 'wb') as f:
            writer = None
            row_count = 0
            
            for chunk in chunk_iter:
                if chunk.empty: continue
                
                rename_map = {
                    't': 'timestamp', 'p': 'price', 
                    's': 'size', 'v': 'size', 'vol': 'size', 'volume': 'size',
                    'c': 'condition', 'x': 'exchange'
                }
                chunk.rename(columns=rename_map, inplace=True)
                
                if 'timestamp' in chunk.columns:
                    chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], format='mixed', utc=True)
                if 'price' in chunk.columns:
                    chunk['price'] = chunk['price'].astype('float32')
                if 'size' in chunk.columns:
                    chunk['size'] = chunk['size'].fillna(0).astype('uint32')
                
                table = pa.Table.from_pandas(chunk)
                if writer is None:
                    writer = pq.ParquetWriter(f, table.schema, compression='snappy')
                
                writer.write_table(table)
                row_count += len(chunk)
                
                with print_lock:
                    print(".", end="", flush=True)
            
            if writer:
                writer.close()
        
        conn.close()
        
        with print_lock:
            print(f" Done ({row_count} rows)")
        
        if row_count > 0:
            with print_lock:
                print(f"[{symbol}] ✅ Upload Verified. Dropping Local Table.")
            db_write_queue.put(('DROP_TABLE', symbol))
            db_write_queue.put(('STATUS', (symbol, ('archived', date.today().isoformat()))))

    except Exception as e:
        with print_lock:
            print(f"\n[{symbol}] ❌ Stream Failed: {e}")

def backfill_symbol(symbol):
    # --- STEP 1: CHECK S3 FIRST ---
    if check_s3_exists(symbol):
        with print_lock:
            print(f"[{symbol}] ⏭️  Exists in S3. Skipped.")
        # Mark purely so we don't check again next run
        db_write_queue.put(('STATUS', (symbol, ('archived', date.today().isoformat()))))
        return

    # --- STEP 2: BACKFILL IF MISSING ---
    read_conn = sqlite3.connect(DB_PATH, timeout=30)
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=TARGET_WINDOW_DAYS)
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
            for trade in trades_list:
                wrapper = TradeDataAdapter(trade)
                batch.append(wrapper)
                if len(batch) >= BATCH_SIZE:
                    # Backpressure Block
                    if db_write_queue.full():
                        with print_lock:
                            print(f"[{symbol}] ⏳ Disk Busy...")
                    db_write_queue.put(('TICKS', (symbol, list(batch))))
                    batch = []

            if batch:
                db_write_queue.put(('TICKS', (symbol, list(batch))))

            db_write_queue.put(('STATUS', (symbol, date_str)))

            with print_lock:
                print(f"[{symbol}] +{len(trades_list)} ticks ({date_str})")
            
        except Exception as e:
            if "429" in str(e):
                time.sleep(20)
            else:
                time.sleep(1)
        
        current_date += timedelta(days=1)
    
    read_conn.close()
    
    # --- STEP 3: EXPORT ---
    export_to_parquet_stream(symbol)
    
    return f"[{symbol}] ✅ Pipeline Complete."

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", help="Backfill single symbol")
    parser.add_argument("--all", action="store_true", help="Backfill ALL symbols")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Parallel workers")
    args = parser.parse_args()

    init_db()
    cleanup_temp_files()
    
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

    logger.info(f"Starting Smart S3-Aware Pipeline")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(backfill_symbol, sym): sym for sym in targets}
        wait(futures)

    db_write_queue.put(None)
    writer_thread.join()
    print("All tasks complete.")

if __name__ == "__main__":
    main()