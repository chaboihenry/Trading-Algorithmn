"""
Statistical Arbitrage Strategy
==============================
Finds mispricings in correlated pairs using cointegration
Target: 15-30% annual returns

Strategy Logic:
- Identifies pairs with extreme z-scores (>2 or <-2)
- Requires cointegration (statistical relationship)
- Optimal half-life: 5-60 days (mean reversion speed)
- Higher z-score = higher confidence

Signal Types:
- LONG_PAIR: Buy undervalued, short overvalued
- SHORT_PAIR: Short overvalued, buy undervalued
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PairsArbitrageStrategy:
    """Statistical arbitrage strategy for pairs trading"""

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db"):
        """
        Initialize pairs arbitrage strategy

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.strategy_name = "pairs_arbitrage"
        logger.info(f"Initialized {self.strategy_name} strategy")

    def _get_db_connection(self) -> sqlite3.Connection:
        """Create database connection"""
        return sqlite3.connect(self.db_path)

    def generate_signals(self, max_signals: int = 10, min_zscore: float = 2.0) -> pd.DataFrame:
        """
        Generate trading signals based on pairs statistics

        Args:
            max_signals: Maximum number of signals to return
            min_zscore: Minimum absolute z-score threshold

        Returns:
            DataFrame with trading signals
        """
        conn = self._get_db_connection()

        logger.info(f"Generating signals with min_zscore={min_zscore}")

        query = """
            SELECT
                symbol_ticker_1,
                symbol_ticker_2,
                current_zscore,
                half_life,
                correlation,
                CASE
                    WHEN current_zscore > ? THEN 'SHORT_PAIR'
                    WHEN current_zscore < -? THEN 'LONG_PAIR'
                    ELSE 'NO_SIGNAL'
                END as signal,
                ABS(current_zscore) - ? as confidence,
                calculation_date
            FROM pairs_statistics
            WHERE is_cointegrated = 1
                AND ABS(current_zscore) > ?
                AND half_life BETWEEN 5 AND 60
            ORDER BY ABS(current_zscore) DESC
            LIMIT ?
        """

        signals = pd.read_sql(
            query,
            conn,
            params=(min_zscore, min_zscore, min_zscore, min_zscore, max_signals)
        )
        conn.close()

        if signals.empty:
            logger.warning("No signals generated")
            return pd.DataFrame()

        # Add strategy metadata
        signals['strategy'] = self.strategy_name
        signals['signal_date'] = datetime.now().strftime('%Y-%m-%d')

        # Normalize confidence to 0-1 scale
        if len(signals) > 0:
            max_conf = signals['confidence'].max()
            if max_conf > 0:
                signals['confidence'] = signals['confidence'] / max_conf

        logger.info(f"Generated {len(signals)} signals")
        logger.info(f"LONG_PAIR signals: {len(signals[signals['signal'] == 'LONG_PAIR'])}")
        logger.info(f"SHORT_PAIR signals: {len(signals[signals['signal'] == 'SHORT_PAIR'])}")

        return signals

    def get_pair_details(self, ticker1: str, ticker2: str) -> dict:
        """
        Get detailed information about a specific pair

        Args:
            ticker1: First ticker symbol
            ticker2: Second ticker symbol

        Returns:
            Dictionary with pair details
        """
        conn = self._get_db_connection()

        query = """
            SELECT *
            FROM pairs_statistics
            WHERE (symbol_ticker_1 = ? AND symbol_ticker_2 = ?)
               OR (symbol_ticker_1 = ? AND symbol_ticker_2 = ?)
            ORDER BY calculation_date DESC
            LIMIT 1
        """

        result = pd.read_sql(query, conn, params=(ticker1, ticker2, ticker2, ticker1))
        conn.close()

        if result.empty:
            return {}

        return result.iloc[0].to_dict()

    def backtest_signal(self, ticker1: str, ticker2: str, days_forward: int = 30) -> dict:
        """
        Simple backtest for a pair signal

        Args:
            ticker1: First ticker symbol
            ticker2: Second ticker symbol
            days_forward: Number of days to simulate forward

        Returns:
            Dictionary with backtest results
        """
        conn = self._get_db_connection()

        # Get historical prices for both tickers
        query = """
            SELECT price_date, close_price
            FROM price_data
            WHERE symbol_ticker = ?
            ORDER BY price_date DESC
            LIMIT ?
        """

        prices1 = pd.read_sql(query, conn, params=(ticker1, days_forward + 10))
        prices2 = pd.read_sql(query, conn, params=(ticker2, days_forward + 10))
        conn.close()

        if prices1.empty or prices2.empty:
            return {"error": "Insufficient price data"}

        # Calculate simple returns
        prices1 = prices1.sort_values('price_date')
        prices2 = prices2.sort_values('price_date')

        # Merge on date
        merged = pd.merge(
            prices1, prices2,
            on='price_date',
            suffixes=('_1', '_2')
        )

        if len(merged) < days_forward:
            return {"error": "Insufficient overlapping data"}

        # Calculate returns for long/short position
        entry_price1 = merged.iloc[0]['close_price_1']
        entry_price2 = merged.iloc[0]['close_price_2']
        exit_price1 = merged.iloc[min(days_forward, len(merged)-1)]['close_price_1']
        exit_price2 = merged.iloc[min(days_forward, len(merged)-1)]['close_price_2']

        return1 = (exit_price1 - entry_price1) / entry_price1
        return2 = (exit_price2 - entry_price2) / entry_price2

        return {
            "ticker1": ticker1,
            "ticker2": ticker2,
            "days_simulated": min(days_forward, len(merged)-1),
            "return_ticker1": return1,
            "return_ticker2": return2,
            "pair_return": (return1 - return2) / 2  # Simple pair return
        }


if __name__ == "__main__":
    # Example usage
    strategy = PairsArbitrageStrategy()

    # Generate signals
    signals = strategy.generate_signals(max_signals=10, min_zscore=2.0)

    if not signals.empty:
        print("\n" + "="*80)
        print("PAIRS ARBITRAGE SIGNALS")
        print("="*80)
        print(signals.to_string(index=False))
        print("="*80)

        # Show detailed info for top signal
        if len(signals) > 0:
            top_signal = signals.iloc[0]
            print(f"\nTop Signal Details:")
            print(f"Pair: {top_signal['symbol_ticker_1']} / {top_signal['symbol_ticker_2']}")
            print(f"Signal: {top_signal['signal']}")
            print(f"Z-Score: {top_signal['current_zscore']:.2f}")
            print(f"Half-Life: {top_signal['half_life']:.1f} days")
            print(f"Confidence: {top_signal['confidence']:.2%}")
    else:
        print("\nNo signals generated with current thresholds")
