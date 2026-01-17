import sqlite3
from config.settings import DB_PATH

def enable_wal():
    conn = sqlite3.connect(DB_PATH)
    # Enable WAL mode for concurrency
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.commit()
    conn.close()
    print(f"✅ WAL Mode enabled for {DB_PATH}. Parallel writes are now safe.")

if __name__ == "__main__":
    enable_wal()