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
from config.all_symbols import SYMBOLS  # Import master list to ensure we catch everyone

# --- CONFIGURATION ---
TARGET_WINDOW_DAYS = 90  # We want at least this much recent data
BATCH_SIZE = 10000       # Ticks per write

# --- API CONNECTION ---
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url='https://paper-api.alpaca.markets')

def get_prioritized_symbols():
    """
    Returns a list of symbols sorted by 'Need'.
    Priority 1: Symbols with NO data (None).
    Priority 2: Symbols with OLD data.
    Priority 3: Symbols that are almost up to date.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get current status from DB
    cursor.execute("SELECT symbol, last_backfilled_date FROM backfill_status")
    db_status = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()

    # Merge with Master SYMBOLS list (in case DB is missing some)
    final_list = []
    for sym in SYMBOLS:
        last_date = db_status.get(sym)
        # Use "1900-01-01" for None so they sort to the top (oldest first)
        sort_date = last_date if last_date else "1900-01-01"
        final_list.append((sym, last_date, sort_date))

    # SORT: Ascending order of date. 
    # "1900" (Empty) comes first. "2026-01-14" comes last.
    final_list.sort(key=lambda x: x[2])

    return final_list

def run_backfill():
    print(f"Connecting to DB at: {DB_PATH}")
    
    # 1. Get Sorted List
    priority_queue = get_prioritized_symbols()
    
    # We want data up to yesterday (today is usually incomplete during market hours)
    # If running after market close, you could use today, but yesterday is safer for consistency.
    target_end_date = date.today() - timedelta(days=1)
    
    # Default start for new symbols (90 days ago)
    default_start_date = target_end_date - timedelta(days=TARGET_WINDOW_DAYS)

    print("=" * 60)
    print(f"SMART BACKFILL (PRIORITY QUEUE)")
    print(f"  Target End: {target_end_date}")
    print(f"  Queue Size: {len(priority_queue)}")
    print(f"  Strategy:   Filling EMPTY/OLD symbols first.")
    print("=" * 60)

    for symbol, last_date_str, _ in priority_queue:
        # Determine Start Date
        if last_date_str:
            last_date = date.fromisoformat(last_date_str)
            # Start from the NEXT day after our last record
            start_date = last_date + timedelta(days=1)
            
            # EDGE CASE: If the bot was off for 6 months, reset to 90 days 
            # to avoid downloading years of useless data.
            if start_date < default_start_date:
                start_date = default_start_date
        else:
            # First run for this symbol
            start_date = default_start_date

        # Check if up to date
        if start_date > target_end_date:
            # This symbol is already done. Since we sorted, we might hit a block of these at the end.
            print(f"✓ {symbol} is up to date ({last_date_str}).")
            continue

        print(f"\n>>> Processing {symbol} (Last: {last_date_str or 'NEVER'})")
        print(f"    Target: {start_date} -> {target_end_date}")

        # Fetch Loop
        current_date = start_date
        while current_date <= target_end_date:
            
            if not is_trading_day(current_date):
                current_date += timedelta(days=1)
                continue

            date_str = current_date.isoformat()
            print(f"    -> {date_str}: Fetching...", end="", flush=True)
            
            try:
                # Use iterator to handle massive volume
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
                    print(" No data")

                # Update DB Status immediately after every successful day
                # This acts as a "Save Point" so if you crash, you don't lose progress on this day
                conn = sqlite3.connect(DB_PATH)
                conn.execute(
                    "INSERT OR REPLACE INTO backfill_status (symbol, status, last_backfilled_date) VALUES (?, ?, ?)", 
                    (symbol, 'active', date_str)
                )
                conn.commit()
                conn.close()

            except KeyboardInterrupt:
                print("\n\n🛑 STOPPING safely. Progress saved.")
                sys.exit(0)
            except Exception as e:
                print(f"\n       Error: {e}")
                time.sleep(2) # Back off slightly on error
            
            current_date += timedelta(days=1)

    print("\n" + "=" * 60)
    print("ALL SYMBOLS UP TO DATE")
    print("=" * 60)

if __name__ == "__main__":
    run_backfill()