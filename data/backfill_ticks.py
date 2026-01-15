import sys
import os
import time
import sqlite3
from datetime import date, timedelta
import alpaca_trade_api as tradeapi

# --- PATH SETUP ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

import data.tick_storage as tick_storage
from utils.market_calendar import is_trading_day
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, DB_PATH

# --- CONFIGURATION ---
TARGET_WINDOW_DAYS = 90  # We want at least this much recent data
BATCH_SIZE = 10000       # Ticks per write

# --- API CONNECTION ---
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url='https://paper-api.alpaca.markets')

def get_symbols_status():
    """Returns dict {symbol: last_backfilled_date_str}"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT symbol, last_backfilled_date FROM backfill_status")
    # Return dict for easy lookup
    status = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return status

def run_backfill():
    print(f"Connecting to DB at: {DB_PATH}")
    symbols_map = get_symbols_status()
    
    # We want data up to yesterday (today is incomplete)
    target_end_date = date.today() - timedelta(days=1)
    
    # Default start for new symbols (90 days ago)
    default_start_date = target_end_date - timedelta(days=TARGET_WINDOW_DAYS)

    print("=" * 60)
    print(f"SMART INCREMENTAL BACKFILL")
    print(f"  Target End: {target_end_date}")
    print(f"  Symbols: {len(symbols_map)}")
    print("=" * 60)

    for symbol, last_date_str in symbols_map.items():
        print(f"\n>>> Checking {symbol}...")

        # 1. Determine Start Date
        if last_date_str:
            last_date = date.fromisoformat(last_date_str)
            # Start from the NEXT day after our last record
            start_date = last_date + timedelta(days=1)
            
            # EDGE CASE: If the bot was off for 6 months, don't download 6 months of data.
            # Just reset to the 90-day window.
            if start_date < default_start_date:
                print(f"    Gap too large (last update: {last_date}). Resetting to 90 days.")
                start_date = default_start_date
        else:
            # First run for this symbol
            start_date = default_start_date

        # 2. Check if up to date
        if start_date > target_end_date:
            print(f"    Already up to date (Last: {last_date_str}). Skipping.")
            continue

        print(f"    Fetching data: {start_date} -> {target_end_date}")

        # 3. Fetch Loop
        current_date = start_date
        while current_date <= target_end_date:
            
            if not is_trading_day(current_date):
                current_date += timedelta(days=1)
                continue

            date_str = current_date.isoformat()
            print(f"    -> {date_str}: Fetching...", end="", flush=True)
            
            try:
                # Use iterator to handle massive volume without RAM crash
                trades_iter = api.get_trades(symbol, start=date_str, end=date_str, limit=1_000_000)
                
                batch = []
                count = 0
                
                for trade in trades_iter:
                    batch.append(trade)
                    if len(batch) >= BATCH_SIZE:
                        tick_storage.save_ticks(symbol, batch)
                        count += len(batch)
                        batch = [] # Clear RAM
                        print(".", end="", flush=True)

                if batch:
                    tick_storage.save_ticks(symbol, batch)
                    count += len(batch)
                
                if count > 0:
                    print(f" Done ({count})")
                else:
                    print(" No ticks")

            except Exception as e:
                print(f"\n       Error: {e}")
                time.sleep(1)
            
            current_date += timedelta(days=1)
        
        # 4. Update Status (Only if we actually finished the loop)
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "UPDATE backfill_status SET status = 'active', last_backfilled_date = ? WHERE symbol = ?", 
            (target_end_date, symbol)
        )
        conn.commit()
        conn.close()

    print("\n" + "=" * 60)
    print("BACKFILL COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    run_backfill()

