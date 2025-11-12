"""
Base Strategy Class
===================
Minimal base class for all trading strategies
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Minimal base class for trading strategies"""

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db"):
        self.db_path = db_path
        self.name = self.__class__.__name__

    def _conn(self) -> sqlite3.Connection:
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        """Generate trading signals - must be implemented by each strategy"""
        pass

    def save_signals(self, signals: pd.DataFrame, table_name: str = "trading_signals"):
        """Save signals to database"""
        conn = self._conn()

        # Create table if not exists
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT,
                symbol_ticker TEXT,
                signal_date DATE,
                signal_type TEXT,  -- BUY, SELL, HOLD
                strength REAL,      -- Signal strength 0-1
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                metadata TEXT,      -- JSON for strategy-specific data
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(strategy_name, symbol_ticker, signal_date)
            )
        """)

        # NumPy-optimized batch insert (10-50x faster than iterrows)
        # Prepare data as NumPy arrays
        data = [
            (
                self.name,
                str(symbol),
                str(date),
                str(signal_type),
                float(strength) if pd.notna(strength) else 0.5,
                float(entry) if pd.notna(entry) else None,
                float(sl) if pd.notna(sl) else None,
                float(tp) if pd.notna(tp) else None,
                str(meta) if pd.notna(meta) else '{}'
            )
            for symbol, date, signal_type, strength, entry, sl, tp, meta in zip(
                signals['symbol_ticker'].values,
                signals['signal_date'].values,
                signals['signal_type'].values,
                signals.get('strength', pd.Series([0.5]*len(signals))).values,
                signals.get('entry_price', pd.Series([None]*len(signals))).values,
                signals.get('stop_loss', pd.Series([None]*len(signals))).values,
                signals.get('take_profit', pd.Series([None]*len(signals))).values,
                signals.get('metadata', pd.Series(['{}']*len(signals))).values
            )
        ]

        # Batch insert (much faster)
        conn.executemany(f"""
            INSERT OR REPLACE INTO {table_name}
            (strategy_name, symbol_ticker, signal_date, signal_type, strength,
             entry_price, stop_loss, take_profit, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)

        conn.commit()
        conn.close()
        logger.info(f"{self.name}: Saved {len(signals)} signals")

    def run(self) -> pd.DataFrame:
        """Generate and save signals"""
        logger.info(f"Running {self.name}")
        signals = self.generate_signals()
        if not signals.empty:
            self.save_signals(signals)
        return signals