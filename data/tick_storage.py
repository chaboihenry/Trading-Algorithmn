import sqlite3
import sys
import os

# --- PATH SETUP ---
# Database is one level up in the project root (or on Vault via symlink/config)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

from config.settings import DB_PATH

def save_ticks(symbol, ticks):
    """
    Saves a list of Alpaca Trade objects to the database.
    Uses Alpaca V2 single-letter attributes: t (time), p (price), s (size), c (cond), x (exchange)
    """
    if not ticks:
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    safe_symbol = symbol.replace('.', '_').replace('-', '_')
    table_name = f"ticks_{safe_symbol}"

    data_tuples = []
    
    for t in ticks:
        # Map Alpaca V2 attributes to DB columns
        # t.t -> Timestamp (usually a pandas Timestamp object)
        # t.p -> Price
        # t.s -> Size (Volume)
        # t.c -> Conditions (list)
        # t.x -> Exchange (Tape)
        
        # We use .isoformat() on the timestamp to make it safe for SQLite
        data_tuples.append((
            t.t.isoformat(),
            t.p,
            t.s,
            ",".join(t.c) if t.c else "",
            t.x
        ))

    # Bulk Insert
    # No try/except blocks here; if the DB is locked or broken, we want to know immediately.
    query = f"""
        INSERT INTO {table_name} (timestamp, price, volume, conditions, tape)
        VALUES (?, ?, ?, ?, ?)
    """
    
    cursor.executemany(query, data_tuples)
    conn.commit()
    conn.close()
    
    print(f"[{symbol}] Saved {len(ticks)} ticks to Vault.")