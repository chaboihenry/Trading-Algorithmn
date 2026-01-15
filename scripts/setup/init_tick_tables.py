import sqlite3
from config.all_symbols import SYMBOLS  # Imports the list directly from your file

DB_PATH = "market_data.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print(f"--- Initializing Database for {len(SYMBOLS)} symbols ---")

    # 1. Create the Master Backfill Status Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS backfill_status (
            symbol TEXT PRIMARY KEY,
            last_backfilled_date DATE,
            status TEXT DEFAULT 'pending'
        )
    """)
    print("Created table: backfill_status")

    # 2. Create a dedicated table for each symbol in your imported list
    for symbol in SYMBOLS:
        # Sanitize symbol just in case (e.g. BRK.B -> BRK_B)
        safe_symbol = symbol.replace(".", "_").replace("-", "_")
        table_name = f"ticks_{safe_symbol}"
        
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                uid INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timestamp DATETIME,
                price REAL,
                volume INTEGER,
                conditions TEXT,
                tape TEXT,
                UNIQUE(timestamp, price, volume) ON CONFLICT IGNORE
            )
        """)
        
        # Add to status table if not already there
        cursor.execute("""
            INSERT OR IGNORE INTO backfill_status (symbol, status)
            VALUES (?, 'pending')
        """, (symbol,))
        
    conn.commit()
    conn.close()
    print("--- Database Setup Complete. All tables created. ---")

if __name__ == "__main__":
    init_db()