import sys
import os
import time
import sqlite3
from datetime import date, timedelta
import alpaca_trade_api as tradeapi

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

import data.tick_storage as tick_storage
from utils.market_calendar import is_trading_day
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, DB_PATH
from config.all_symbols import SYMBOLS
from config.logging_config import setup_logging

logger = setup_logging(script_name="backfill_ticks")

# --- CONFIGURATION ---
# Changed from 90 to 365 to capture a full year of market regimes (Bull, Bear, Chop)
TARGET_WINDOW_DAYS = 365 
BATCH_SIZE = 10000

# Connect to Alpaca
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url='https://paper-api.alpaca.markets')

def get_prioritized_symbols():
    """
    Returns symbols sorted by 'least recently updated' to ensure fairness.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Ensure status table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS backfill_status (
            symbol TEXT PRIMARY KEY,
            status TEXT,
            last_backfilled_date TEXT
        )
    """)
    conn.commit()
    
    cursor.execute("SELECT symbol, last_backfilled_date FROM backfill_status")
    db_status = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()

    final_list = []
    for sym in SYMBOLS:
        last_date = db_status.get(sym)
        # Default to a very old date if never backfilled
        sort_date = last_date if last_date else "1900-01-01"
        final_list.append((sym, last_date, sort_date))

    # Sort by date (oldest first)
    final_list.sort(key=lambda x: x[2])
    return final_list

def run_backfill():
    logger.info(f"Connecting to DB at: {DB_PATH}")
    priority_queue = get_prioritized_symbols()
    
    target_end_date = date.today() - timedelta(days=1)
    default_start_date = target_end_date - timedelta(days=TARGET_WINDOW_DAYS)

    logger.info("=" * 60)
    logger.info(f"SMART BACKFILL (PRIORITY QUEUE)")
    logger.info(f"  Target Window: {TARGET_WINDOW_DAYS} Days")
    logger.info(f"  Start Date:    {default_start_date}")
    logger.info(f"  Queue Size:    {len(priority_queue)} Symbols")
    logger.info("=" * 60)

    for symbol, last_date_str, _ in priority_queue:
        # Determine where to resume from
        if last_date_str:
            last_date = date.fromisoformat(last_date_str)
            start_date = last_date + timedelta(days=1)
            # If the database has older data than our 365 target, we just append recent data.
            # If the database is missing the deep history, we might want to force start_date back.
            # For now, we assume we want to fill GAPS forward.
            if start_date < default_start_date:
                start_date = default_start_date
        else:
            start_date = default_start_date

        if start_date > target_end_date:
            logger.info(f"✓ {symbol} is up to date ({last_date_str}).")
            continue

        logger.info(f">>> Processing {symbol} (Last: {last_date_str or 'NEVER'})")
        logger.info(f"    Target: {start_date} -> {target_end_date}")

        current_date = start_date
        while current_date <= target_end_date:
            if not is_trading_day(current_date):
                current_date += timedelta(days=1)
                continue

            date_str = current_date.isoformat()
            
            try:
                # Fetch trades for the single day
                # limit=None usually implies auto-pagination in recent SDKs, but we use high number
                trades_iter = api.get_trades(symbol, start=date_str, end=date_str, limit=None)
                
                batch = []
                count = 0
                
                for trade in trades_iter:
                    # Convert to dictionary for storage compatibility
                    t_dict = {
                        'timestamp': trade.t.isoformat() if hasattr(trade, 't') else trade.timestamp,
                        'price': float(trade.p) if hasattr(trade, 'p') else float(trade.price),
                        'size': float(trade.s) if hasattr(trade, 's') else float(trade.size),
                        'exchange': trade.x if hasattr(trade, 'x') else getattr(trade, 'exchange', ''),
                        'conditions': ",".join(trade.c) if hasattr(trade, 'c') else str(getattr(trade, 'conditions', ''))
                    }
                    batch.append(t_dict)
                    
                    if len(batch) >= BATCH_SIZE:
                        tick_storage.save_ticks(symbol, batch)
                        count += len(batch)
                        batch = []

                if batch:
                    tick_storage.save_ticks(symbol, batch)
                    count += len(batch)
                
                if count > 0:
                    logger.info(f"    -> {date_str}: Done ({count} ticks)")
                else:
                    logger.info(f"    -> {date_str}: No data found")

                # Update Status in DB
                conn = sqlite3.connect(DB_PATH)
                conn.execute(
                    "INSERT OR REPLACE INTO backfill_status (symbol, status, last_backfilled_date) VALUES (?, ?, ?)", 
                    (symbol, 'active', date_str)
                )
                conn.commit()
                conn.close()

            except KeyboardInterrupt:
                logger.warning("STOPPING safely. Progress saved.")
                sys.exit(0)
            except Exception as e:
                logger.error(f"Error processing {symbol} on {date_str}: {e}")
                time.sleep(2) # Backoff
            
            current_date += timedelta(days=1)

    logger.info("ALL SYMBOLS UP TO DATE")

if __name__ == "__main__":
    run_backfill()