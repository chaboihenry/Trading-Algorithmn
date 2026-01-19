import sqlite3
import sys
import os
from pathlib import Path

# --- PATH SETUP ---
# Adds project root to python path so we can import config
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.settings import DB_PATH
from config.all_symbols import SYMBOLS

def verify_database():
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at: {DB_PATH}")
        return

    print(f"🔎 Scanning Database: {DB_PATH}")
    print(f"{'SYMBOL':<8} | {'DAYS':<6} | {'STATUS':<10} | {'DATE RANGE'}")
    print("-" * 60)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    missing_symbols = []
    incomplete_symbols = []
    
    # Approx trading days in 365 days (allowing for holidays)
    REQUIRED_DAYS = 240 

    for symbol in SYMBOLS:
        safe_symbol = symbol.replace('.', '_').replace('-', '_')
        table_name = f"ticks_{safe_symbol}"
        
        # 1. Check if table exists
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if not cursor.fetchone():
            print(f"{symbol:<8} | {'0':<6} | ❌ MISSING  | N/A")
            missing_symbols.append(symbol)
            continue

        # 2. Count Unique Days & Get Range
        # extracting YYYY-MM-DD from timestamp string for speed
        query = f"""
            SELECT 
                COUNT(DISTINCT substr(timestamp, 1, 10)),
                MIN(substr(timestamp, 1, 10)),
                MAX(substr(timestamp, 1, 10))
            FROM {table_name}
        """
        try:
            cursor.execute(query)
            result = cursor.fetchone()
            day_count = result[0]
            start_date = result[1]
            end_date = result[2]
            
            # 3. Determine Status
            if day_count >= REQUIRED_DAYS:
                status = "✅ OK"
            else:
                status = "⚠️ LOW"
                incomplete_symbols.append(symbol)

            print(f"{symbol:<8} | {day_count:<6} | {status:<10} | {start_date} -> {end_date}")

        except Exception as e:
            print(f"{symbol:<8} | {'ERR':<6} | ❌ ERROR    | {e}")

    conn.close()

    print("-" * 60)
    print("SUMMARY:")
    if missing_symbols:
        print(f"❌ Missing Tables: {len(missing_symbols)}")
        print(f"   {', '.join(missing_symbols)}")
    if incomplete_symbols:
        print(f"⚠️ Incomplete Data (<{REQUIRED_DAYS} days): {len(incomplete_symbols)}")
        print(f"   {', '.join(incomplete_symbols)}")
    
    if not missing_symbols and not incomplete_symbols:
        print("🎉 SUCCESS: All symbols have valid 1-year history.")

if __name__ == "__main__":
    verify_database()