import sys
import os
import time
import sqlite3
from datetime import date, timedelta, datetime
import alpaca_trade_api as tradeapi

# --- PATH SETUP ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

# --- IMPORTS ---
import tick_storage
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, DB_PATH

# --- API CONNECTION ---
# We use the REST object. 
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url='https://paper-api.alpaca.markets')

def get_symbols():
    """Get all symbols from the DB, regardless of status."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT symbol FROM backfill_status")
    symbols = [row[0] for row in cursor.fetchall()]
    conn.close()
    return symbols

def run_backfill():
    print(f"Connecting to DB at: {DB_PATH}")
    symbols = get_symbols()
    
    # Define Date Range: 1 Year
    end_date = date.today() - timedelta(days=1)  # Yesterday
    start_date = end_date - timedelta(days=365)  # 1 Year ago

    print(f"Backfilling {len(symbols)} symbols.")
    print(f"Range: {start_date} to {end_date}")

    for symbol in symbols:
        print(f"\n=== Processing {symbol} ===")
        
        # Iterate through each day
        current_date = start_date
        while current_date <= end_date:
            
            # Skip weekends
            if current_date.weekday() < 5:
                date_str = current_date.isoformat()
                
                try:
                    # fetch trades
                    # We use limit=10000. If QQQ has more, this grabs the first 10k. 
                    # To get ALL ticks for QQQ requires complex pagination, but this 
                    # is the simple, standard approach for a basic backfill.
                    trades = api.get_trades(symbol, start=date_str, end=date_str, limit=10000)

                    # 'trades' is an iterable object in the new SDK
                    # We convert to list to check if empty
                    trades_list = [t for t in trades]

                    if trades_list:
                        tick_storage.save_ticks(symbol, trades_list)
                    else:
                        # No data for this day (holiday or inactive)
                        pass

                except Exception as e:
                    print(f"Error on {date_str} for {symbol}: {e}")
                    time.sleep(1) 

                # Sleep to prevent hitting API rate limits too hard
                time.sleep(0.2)
            
            # Move to next day
            current_date += timedelta(days=1)
        
        print(f"Finished {symbol}")
        
        # Mark as completed in DB
        conn = sqlite3.connect(DB_PATH)
        conn.execute("UPDATE backfill_status SET status = 'completed', last_backfilled_date = ? WHERE symbol = ?", (date.today(), symbol))
        conn.commit()
        conn.close()

if __name__ == "__main__":
    run_backfill()