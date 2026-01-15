import sqlite3
import pandas as pd
import os
import sys

# --- PATH SETUP ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

from config.settings import DB_PATH

class TickStorage:
    """
    Manages storage and retrieval of tick data from SQLite.
    """
    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH
        self.conn = None

    def _connect(self):
        if not self.conn:
            self.conn = sqlite3.connect(self.db_path)

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def save_ticks(self, symbol, ticks):
        """
        Saves a list of Alpaca Trade objects to the database.
        Static method wrapper for backward compatibility or direct usage.
        """
        # This allows calling TickStorage.save_ticks() directly if needed
        # or we can instantiate the class.
        if not ticks:
            return

        conn = sqlite3.connect(DB_PATH) # Use fresh connection for safety in threads
        cursor = conn.cursor()

        safe_symbol = symbol.replace('.', '_').replace('-', '_')
        table_name = f"ticks_{safe_symbol}"
        
        # Ensure table exists (idempotent)
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                timestamp TEXT,
                price REAL,
                volume REAL,
                conditions TEXT,
                tape TEXT
            )
        """)
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{safe_symbol}_ts ON {table_name} (timestamp)")

        data_tuples = []
        for t in ticks:
            # Check if timestamp is a datetime object or already a string
            if hasattr(t.t, 'isoformat'):
                ts = t.t.isoformat()
            else:
                ts = str(t.t)

            # Handle conditions
            if hasattr(t, 'c') and isinstance(t.c, list):
                cond = ",".join(t.c)
            elif hasattr(t, 'c'):
                cond = str(t.c)
            else:
                cond = ""
            
            # Handle tape
            tape = t.x if hasattr(t, 'x') else ""

            data_tuples.append((ts, t.p, t.s, cond, tape))

        query = f"INSERT INTO {table_name} (timestamp, price, volume, conditions, tape) VALUES (?, ?, ?, ?, ?)"
        
        cursor.executemany(query, data_tuples)
        conn.commit()
        conn.close()
        # print(f"[{symbol}] Saved {len(ticks)} ticks.")

    def load_ticks(self, symbol, start_date=None, end_date=None):
        """
        Loads ticks from the database. Returns a list of tuples or objects.
        Used by the training scripts.
        """
        self._connect()
        cursor = self.conn.cursor()
        
        safe_symbol = symbol.replace('.', '_').replace('-', '_')
        table_name = f"ticks_{safe_symbol}"

        # Check if table exists
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if not cursor.fetchone():
            return []

        query = f"SELECT timestamp, price, volume FROM {table_name}"
        params = []
        
        if start_date or end_date:
            query += " WHERE 1=1"
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
        
        query += " ORDER BY timestamp ASC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Return format expected by generate_bars_from_ticks: list of (timestamp, price, volume)
        return rows

# --- Standalone function for the backfill script to use easily ---
def save_ticks(symbol, ticks):
    """Standalone wrapper so backfill_ticks.py doesn't break."""
    storage = TickStorage()
    storage.save_ticks(symbol, ticks)