"""
Base Strategy Class
===================
Minimal base class for all trading strategies with dynamic signal filtering
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Base class for trading strategies with dynamic filtering and Kelly Criterion"""

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db"):
        self.db_path = db_path
        self.name = self.__class__.__name__

    def _conn(self) -> sqlite3.Connection:
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def _get_market_volatility(self) -> float:
        """
        Calculate current market volatility using VIX or SPY volatility
        Used for dynamic threshold adjustment
        """
        conn = self._conn()

        # Try VIX first (preferred)
        try:
            query = """
                SELECT vix
                FROM economic_indicators
                WHERE vix IS NOT NULL
                ORDER BY indicator_date DESC
                LIMIT 1
            """
            result = pd.read_sql(query, conn)
            if not result.empty:
                vix = result['vix'].iloc[0]
                # Convert VIX to decimal (VIX is in percentage form)
                market_vol = vix / 100.0
                conn.close()
                return market_vol
        except:
            pass

        # Fallback: Calculate SPY 20-day volatility
        try:
            query = """
                SELECT close
                FROM raw_price_data
                WHERE symbol_ticker = 'SPY'
                ORDER BY price_date DESC
                LIMIT 21
            """
            prices = pd.read_sql(query, conn)
            if len(prices) >= 20:
                returns = np.diff(np.log(prices['close'].values))
                volatility = np.std(returns) * np.sqrt(252)  # Annualized
                conn.close()
                return volatility
        except:
            pass

        conn.close()
        return 0.20  # Default: 20% annual volatility (market average)

    def _get_dynamic_threshold(self, market_volatility: float) -> float:
        """
        Adaptive confidence thresholds based on market regime

        High volatility → Higher threshold (only very confident signals)
        Low volatility → Lower threshold (accept more signals)

        Args:
            market_volatility: Current market volatility (annualized)

        Returns:
            Confidence threshold for signal filtering
        """
        if market_volatility > 0.30:  # Very high volatility (30%+)
            return 0.85  # Only extremely high confidence
        elif market_volatility > 0.25:  # High volatility (25-30%)
            return 0.75
        elif market_volatility > 0.15:  # Normal volatility (15-25%)
            return 0.65
        else:  # Low volatility (<15%)
            return 0.55  # Accept more signals

    def _calculate_kelly_position_size(
        self,
        win_probability: float,
        win_loss_ratio: float = 1.5,
        max_position: float = 0.25
    ) -> float:
        """
        Kelly Criterion position sizing for optimal capital allocation

        Formula: f* = (p*b - q) / b
        where:
            p = win probability (signal strength)
            q = loss probability (1 - p)
            b = win/loss ratio (expected profit/loss)

        Args:
            win_probability: Signal strength/confidence (0-1)
            win_loss_ratio: Expected win/loss ratio (default 1.5 from backtesting)
            max_position: Maximum position size cap (default 25%)

        Returns:
            Position size as fraction of portfolio (0-max_position)
        """
        if win_probability <= 0.5:
            return 0.0  # No edge, no position

        loss_probability = 1 - win_probability

        # Kelly formula
        kelly_fraction = (win_probability * win_loss_ratio - loss_probability) / win_loss_ratio

        # Half-Kelly for safety (full Kelly can be too aggressive)
        safe_kelly = kelly_fraction * 0.5

        # Cap at maximum position size
        position_size = max(0.0, min(safe_kelly, max_position))

        return position_size

    def apply_dynamic_filtering(
        self,
        signals: pd.DataFrame,
        min_position_size: float = 0.02,
        portfolio_value: float = 100000.0
    ) -> pd.DataFrame:
        """
        Apply dynamic signal filtering with Kelly Criterion position sizing

        This method:
        1. Gets current market volatility
        2. Sets adaptive confidence threshold
        3. Filters signals by dynamic threshold
        4. Calculates Kelly position sizes
        5. Filters out positions below minimum size

        Args:
            signals: DataFrame with signals (must have 'strength' column)
            min_position_size: Minimum position size (default 2%)
            portfolio_value: Total portfolio value for position calculation

        Returns:
            Filtered signals with position_size column added
        """
        if signals.empty:
            return signals

        # Get current market volatility
        market_vol = self._get_market_volatility()
        dynamic_threshold = self._get_dynamic_threshold(market_vol)

        logger.info(f"Market volatility: {market_vol:.1%}, Dynamic threshold: {dynamic_threshold:.1%}")

        # Filter by dynamic threshold
        filtered_signals = signals[signals['strength'] >= dynamic_threshold].copy()

        if filtered_signals.empty:
            logger.info(f"No signals meet dynamic threshold of {dynamic_threshold:.1%}")
            return pd.DataFrame()

        # Calculate Kelly position sizes (vectorized)
        filtered_signals['position_size'] = filtered_signals['strength'].apply(
            lambda x: self._calculate_kelly_position_size(x)
        )

        # Calculate dollar amounts
        filtered_signals['position_value'] = (
            filtered_signals['position_size'] * portfolio_value
        )

        # Filter out positions below minimum size
        filtered_signals = filtered_signals[
            filtered_signals['position_size'] >= min_position_size
        ].copy()

        # Update metadata with position sizing info
        def add_position_metadata(row):
            try:
                existing_meta = json.loads(row.get('metadata', '{}'))
            except:
                existing_meta = {}

            existing_meta.update({
                'market_volatility': round(market_vol, 4),
                'dynamic_threshold': round(dynamic_threshold, 4),
                'kelly_position_size': round(row['position_size'], 4),
                'position_value': round(row['position_value'], 2)
            })

            return json.dumps(existing_meta)

        filtered_signals['metadata'] = filtered_signals.apply(add_position_metadata, axis=1)

        logger.info(f"Dynamic filtering: {len(signals)} → {len(filtered_signals)} signals")
        logger.info(f"Average position size: {filtered_signals['position_size'].mean():.1%}")

        return filtered_signals

    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        """Generate trading signals - must be implemented by each strategy"""
        pass

    def save_signals(self, signals: pd.DataFrame, table_name: str = "trading_signals"):
        """Save signals to database with position sizing"""
        conn = self._conn()

        # Create table if not exists (now includes position sizing columns)
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
                position_size REAL, -- Kelly position size (fraction)
                position_value REAL, -- Dollar value of position
                metadata TEXT,      -- JSON for strategy-specific data
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(strategy_name, symbol_ticker, signal_date)
            )
        """)

        # NumPy-optimized batch insert (10-50x faster than iterrows)
        # Prepare data as NumPy arrays (now includes position sizing)
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
                float(pos_size) if pd.notna(pos_size) else None,
                float(pos_val) if pd.notna(pos_val) else None,
                str(meta) if pd.notna(meta) else '{}'
            )
            for symbol, date, signal_type, strength, entry, sl, tp, pos_size, pos_val, meta in zip(
                signals['symbol_ticker'].values,
                signals['signal_date'].values,
                signals['signal_type'].values,
                signals.get('strength', pd.Series([0.5]*len(signals))).values,
                signals.get('entry_price', pd.Series([None]*len(signals))).values,
                signals.get('stop_loss', pd.Series([None]*len(signals))).values,
                signals.get('take_profit', pd.Series([None]*len(signals))).values,
                signals.get('position_size', pd.Series([None]*len(signals))).values,
                signals.get('position_value', pd.Series([None]*len(signals))).values,
                signals.get('metadata', pd.Series(['{}']*len(signals))).values
            )
        ]

        # Batch insert (much faster)
        conn.executemany(f"""
            INSERT OR REPLACE INTO {table_name}
            (strategy_name, symbol_ticker, signal_date, signal_type, strength,
             entry_price, stop_loss, take_profit, position_size, position_value, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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