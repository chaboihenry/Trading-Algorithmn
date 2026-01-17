import sys
import os
import time
import sqlite3
import concurrent.futures
from datetime import date, timedelta
import alpaca_trade_api as tradeapi

# --- PATH SETUP ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

import data.tick_storage as tick_storage
from utils.market_calendar import is_trading_day
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, DB_PATH
from config.all_symbols import SYMBOLS

# --- CONFIG ---
TARGET_WINDOW_DAYS = 365 # 1 Year history
MAX_WORKERS = 10         # Number of parallel downloads (Don't go too high or Alpaca will rate limit)
BATCH_SIZE = 20000       # Larger batch size for fewer DB writes

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url='https://paper-api.alpaca.markets')

def backfill_symbol(symbol):
    """
    Worker function to backfill a single symbol.
    """
    # 1. Determine Date Range
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT last_backfilled_date FROM backfill_status WHERE symbol=?", (symbol,))
    row = cursor.fetchone()
    conn.close()

    target_end_date = date.today() - timedelta(days=1)
    default_start = target_end_date - timedelta(days=TARGET_WINDOW_DAYS)
    
    start_date = default_start
    if row and row[0]:
        last_date = date.fromisoformat(row[0])
        start_date = last_date + timedelta(days=1)
        if start_date < default_start: start_date = default_start

    if start_date > target_end_date:
        return f"✓ {symbol} is up to date."

    print(f"[{symbol}] Starting: {start_date} -> {target_end_date}")

    # 2. Iterate Days
    current_date = start_date
    while current_date <= target_end_date:
        if not is_trading_day(current_date):
            current_date += timedelta(days=1)
            continue

        date_str = current_date.isoformat()
        
        try:
            # Fetch Ticks
            trades = api.get_trades(symbol, start=date_str, end=date_str, limit=1_000_000).df
            
            # Convert to list of objects compatible with TickStorage
            # (The API returns a DataFrame, TickStorage expects objects/tuples)
            # We reconstruct the 'trade' objects locally to reuse existing save logic
            if not trades.empty:
                # Optimized: We write raw SQL here or reuse storage class if adapted
                # For compatibility, use the storage class
                # but we need to convert DF rows to the expected object format
                class TradeObj:
                    def __init__(self, t, p, s, c, x):
                        self.t = t
                        self.p = p
                        self.s = s
                        self.c = c
                        self.x = x

                batch = []
                for index, row in trades.iterrows():
                    # Handle conditions list/str mismatch
                    conds = row['conditions'] if isinstance(row['conditions'], list) else [row['conditions']]
                    batch.append(TradeObj(index, row['price'], row['size'], conds, row['exchange']))
                
                tick_storage.save_ticks(symbol, batch)
                
                # Update Status
                conn = sqlite3.connect(DB_PATH)
                conn.execute(
                    "INSERT OR REPLACE INTO backfill_status (symbol, status, last_backfilled_date) VALUES (?, ?, ?)", 
                    (symbol, 'active', date_str)
                )
                conn.commit()
                conn.close()
                print(f"[{symbol}] Saved {len(batch)} ticks for {date_str}")
            
        except Exception as e:
            print(f"[{symbol}] Error on {date_str}: {e}")
            # If rate limit, sleep
            if '429' in str(e):
                time.sleep(5)

        current_date += timedelta(days=1)

    return f"✅ {symbol} Finished."

def run():
    print(f"🚀 Starting Parallel Backfill with {MAX_WORKERS} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(backfill_symbol, sym): sym for sym in SYMBOLS}
        
        for future in concurrent.futures.as_completed(futures):
            sym = futures[future]
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"❌ {sym} crashed: {e}")

if __name__ == "__main__":
    run()