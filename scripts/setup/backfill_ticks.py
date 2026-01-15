import sqlite3
import time
from datetime import date

# --- IMPORTS ---
from data.tick_storage import save_ticks_to_db

DB_PATH = "market_data.db"
TARGET_DATE = "2025-01-03" # The specific date we are filling

def get_pending_symbols():
    """Fetch symbols that are still marked 'pending' in the status table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT symbol FROM backfill_status WHERE status = 'pending'")
    symbols = [row[0] for row in cursor.fetchall()]
    conn.close()
    return symbols

def update_status(symbol, status):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE backfill_status 
        SET status = ?, last_backfilled_date = ? 
        WHERE symbol = ?
    """, (status, date.today(), symbol))
    conn.commit()
    conn.close()

def run_backfill():
    symbols_to_run = get_pending_symbols()
    
    if not symbols_to_run:
        print("No pending symbols found. Did you run init_tick_tables.py?")
        return

    print(f"Starting backfill for {len(symbols_to_run)} symbols...")

    for symbol in symbols_to_run:
        print(f"\n--- Processing {symbol} ---")
        
        try:
            # 1. FETCH DATA
            # Replace the line below with your real API call:
            # data = get_polygon_ticks(symbol, TARGET_DATE)
            
            print(f"Fetching data for {symbol}...")
            
            # --- TEST DATA (DELETE AFTER VERIFYING) ---
            import random
            data = [{'t': 1672531200000, 'p': random.uniform(100, 200), 's': 100, 'c': [1], 'z': 1}]
            # ------------------------------------------

            # 2. SAVE DATA
            save_ticks_to_db(symbol, data)

            # 3. UPDATE STATUS
            update_status(symbol, 'completed')

        except Exception as e:
            print(f"Failed to backfill {symbol}: {e}")
            # Optional: update_status(symbol, 'failed')

        # Sleep to avoid rate limits (adjust based on your API tier)
        time.sleep(1) 
    
    print("\n--- Backfill Run Complete ---")

if __name__ == "__main__":
    run_backfill()