import sys
import os
import time
import sqlite3
from datetime import date, timedelta
import alpaca_trade_api as tradeapi

# --- PATH SETUP ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

import tick_storage
from utils.market_calendar import is_trading_day
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, DB_PATH

# --- API CONNECTION ---
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url='https://paper-api.alpaca.markets')

def get_symbols():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT symbol FROM backfill_status")
    symbols = [row[0] for row in cursor.fetchall()]
    conn.close()
    return symbols

def run_backfill():
    print(f"Connecting to DB at: {DB_PATH}")
    symbols = get_symbols()
    
    # --- CONFIG: 90 DAYS ---
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=90)

    print(f"Backfilling {len(symbols)} symbols.")
    print(f"Window: {start_date} to {end_date}")

    for symbol in symbols:
        print(f"\n=== Processing {symbol} ===")
        
        current_date = start_date
        while current_date <= end_date:
            
            # --- SMART CALENDAR CHECK ---
            if not is_trading_day(current_date):
                current_date += timedelta(days=1)
                continue

            date_str = current_date.isoformat()
            print(f"   -> {date_str}: Fetching...", end="", flush=True)
            
            try:
                # Ask for effectively unlimited trades (limit=10,000,000)
                # The library will auto-paginate in chunks of 10k in the background.
                trades_iter = api.get_trades(symbol, start=date_str, end=date_str, limit=10_000_000)
                
                batch = []
                total_ticks_day = 0
                
                # Iterate through the stream of trades
                for trade in trades_iter:
                    batch.append(trade)
                    
                    # When batch hits 10k, save it
                    if len(batch) >= 10000:
                        tick_storage.save_ticks(symbol, batch)
                        total_ticks_day += len(batch)
                        print(".", end="", flush=True) # Visual progress
                        batch = [] # Reset batch

                # Save any remaining trades at the end of the day
                if batch:
                    tick_storage.save_ticks(symbol, batch)
                    total_ticks_day += len(batch)
                
                print(f" Done. ({total_ticks_day} ticks)")

            except Exception as e:
                print(f"\n      Error on {date_str}: {e}")
                time.sleep(1) # Back off slightly on error
            
            current_date += timedelta(days=1)
        
        # Mark Symbol Complete in DB
        conn = sqlite3.connect(DB_PATH)
        conn.execute("UPDATE backfill_status SET status = 'completed', last_backfilled_date = ? WHERE symbol = ?", (date.today(), symbol))
        conn.commit()
        conn.close()
        print(f"Finished {symbol}")

if __name__ == "__main__":
    run_backfill()