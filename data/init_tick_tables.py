import sqlite3
import sys
import os

# Add Project Root to Path so we can find 'config' and 'all_symbols'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

# Import from our new settings file
from config.settings import DB_PATH
from config.all_symbols import SYMBOLS

def init_db():
    print(f"Target Database: {DB_PATH}")
    print(f"Initializing tables for {len(SYMBOLS)} symbols...")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS backfill_status (
            symbol TEXT PRIMARY KEY,
            last_backfilled_date DATE,
            status TEXT DEFAULT 'pending'
        )
    """)

    for symbol in SYMBOLS:
        safe_symbol = symbol.replace('.', '_').replace('-', '_')
        table_name = f"ticks_{safe_symbol}"
        
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                uid INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                price REAL,
                volume INTEGER,
                conditions TEXT,
                tape TEXT,
                UNIQUE(timestamp, price, volume) ON CONFLICT IGNORE
            )
        """)

        cursor.execute("INSERT OR IGNORE INTO backfill_status (symbol) VALUES (?)", (symbol,))

    conn.commit()
    conn.close()
    print("Database initialized.")

if __name__ == "__main__":
    init_db()