import sqlite3
import sys
import os

# --- PATH SETUP ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

from config.settings import DB_PATH

def save_ticks(symbol, ticks):
    """
    Saves a list of Alpaca Trade objects to the database.
    """
    if not ticks:
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    safe_symbol = symbol.replace('.', '_').replace('-', '_')
    table_name = f"ticks_{safe_symbol}"

    data_tuples = []
    
    for t in ticks:
        # FIX: Check if timestamp is a datetime object or already a string
        if hasattr(t.t, 'isoformat'):
            ts = t.t.isoformat()
        else:
            ts = str(t.t)

        # Handle conditions (Alpaca V2 usually sends a list)
        if isinstance(t.c, list):
            cond = ",".join(t.c)
        else:
            cond = str(t.c) if t.c else ""

        data_tuples.append((
            ts,
            t.p,
            t.s,
            cond,
            t.x
        ))

    query = f"""
        INSERT INTO {table_name} (timestamp, price, volume, conditions, tape)
        VALUES (?, ?, ?, ?, ?)
    """
    
    cursor.executemany(query, data_tuples)
    conn.commit()
    conn.close()
    
    print(f"[{symbol}] Saved {len(ticks)} ticks to Vault.")