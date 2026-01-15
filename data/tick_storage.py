import sqlite3

DB_PATH = "market_data.db"

def save_ticks_to_db(symbol, ticks_data):
    """
    Saves a list of tick dictionaries to the specific table for that symbol.
    """
    if not ticks_data:
        print(f"[{symbol}] No data to save.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Determine table name dynamically
    safe_symbol = symbol.replace(".", "_").replace("-", "_")
    table_name = f"ticks_{safe_symbol}"

    # Prepare data for insertion
    # Maps Polygon keys to DB columns: (symbol, timestamp, price, volume, conditions, tape)
    formatted_data = []
    for t in ticks_data:
        ts = t.get('t')            # Timestamp
        price = t.get('p')         # Price
        size = t.get('s')          # Size/Volume
        cond = str(t.get('c', [])) # Conditions (list to string)
        tape = t.get('z', '')      # Tape
        
        formatted_data.append((symbol, ts, price, size, cond, tape))

    try:
        # Bulk insert is very fast
        cursor.executemany(f"""
            INSERT INTO {table_name} (symbol, timestamp, price, volume, conditions, tape)
            VALUES (?, ?, ?, ?, ?, ?)
        """, formatted_data)
        
        conn.commit()
        print(f"[{symbol}] Saved {cursor.rowcount} ticks to table '{table_name}'.")

    except sqlite3.OperationalError as e:
        print(f"[{symbol}] Database Error: {e}")
        print(f"Hint: Did you run init_tick_tables.py? Table '{table_name}' might be missing.")
    finally:
        conn.close()