"""
Event-Driven Sentiment Strategy
================================
Trades on sentiment/price divergence opportunities
Target: 20-40% annual returns

Strategy Logic:
- Identifies divergence between sentiment and price action
- Combines with RSI for overbought/oversold confirmation
- Positive divergence: bullish sentiment, price down → BUY
- Negative divergence: bearish sentiment, price up → SELL

Signal Types:
- BUY: Positive sentiment divergence + oversold (RSI < 30)
- SELL: Negative sentiment divergence + overbought (RSI > 70)
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentEventStrategy:
    """Event-driven sentiment divergence strategy"""

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db"):
        """
        Initialize sentiment event strategy

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.strategy_name = "sentiment_events"
        logger.info(f"Initialized {self.strategy_name} strategy")

    def _get_db_connection(self) -> sqlite3.Connection:
        """Create database connection"""
        return sqlite3.connect(self.db_path)

    def generate_signals(self,
                        max_signals: int = 10,
                        min_divergence: float = 0.2,
                        rsi_oversold: float = 30,
                        rsi_overbought: float = 70) -> pd.DataFrame:
        """
        Generate trading signals based on sentiment-price divergence

        Args:
            max_signals: Maximum number of signals to return
            min_divergence: Minimum sentiment divergence threshold
            rsi_oversold: RSI threshold for oversold condition
            rsi_overbought: RSI threshold for overbought condition

        Returns:
            DataFrame with trading signals
        """
        conn = self._get_db_connection()

        logger.info(f"Generating signals with min_divergence={min_divergence}")

        query = """
            SELECT
                m.symbol_ticker,
                m.feature_date,
                m.sentiment_score,
                m.sentiment_price_divergence,
                m.sentiment_ma_7d,
                t.rsi_14,
                m.return_5d,
                CASE
                    WHEN m.sentiment_price_divergence > ? AND t.rsi_14 < ? THEN 'BUY'
                    WHEN m.sentiment_price_divergence < -? AND t.rsi_14 > ? THEN 'SELL'
                    ELSE 'NO_SIGNAL'
                END as signal,
                ABS(m.sentiment_price_divergence) as confidence
            FROM ml_features m
            JOIN technical_indicators t
                ON m.symbol_ticker = t.symbol_ticker
                AND m.feature_date = t.indicator_date
            WHERE ABS(m.sentiment_price_divergence) > ?
                AND m.sentiment_score IS NOT NULL
                AND t.rsi_14 IS NOT NULL
            ORDER BY ABS(m.sentiment_price_divergence) DESC
            LIMIT ?
        """

        signals = pd.read_sql(
            query,
            conn,
            params=(
                min_divergence, rsi_oversold,
                min_divergence, rsi_overbought,
                min_divergence, max_signals
            )
        )
        conn.close()

        if signals.empty:
            logger.warning("No signals generated")
            return pd.DataFrame()

        # Filter out NO_SIGNAL entries
        signals = signals[signals['signal'] != 'NO_SIGNAL'].copy()

        if signals.empty:
            logger.warning("No valid signals after filtering")
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
        logger.info(f"BUY signals: {len(signals[signals['signal'] == 'BUY'])}")
        logger.info(f"SELL signals: {len(signals[signals['signal'] == 'SELL'])}")

        return signals

    def get_sentiment_history(self, ticker: str, days: int = 30) -> pd.DataFrame:
        """
        Get sentiment history for a ticker

        Args:
            ticker: Stock ticker symbol
            days: Number of days to retrieve

        Returns:
            DataFrame with sentiment history
        """
        conn = self._get_db_connection()

        query = """
            SELECT
                sentiment_date,
                sentiment_score,
                source_count
            FROM sentiment_data
            WHERE symbol_ticker = ?
            ORDER BY sentiment_date DESC
            LIMIT ?
        """

        history = pd.read_sql(query, conn, params=(ticker, days))
        conn.close()

        return history

    def get_divergence_analysis(self, ticker: str, days: int = 30) -> pd.DataFrame:
        """
        Get detailed divergence analysis for a ticker

        Args:
            ticker: Stock ticker symbol
            days: Number of days to analyze

        Returns:
            DataFrame with divergence metrics
        """
        conn = self._get_db_connection()

        query = """
            SELECT
                m.feature_date,
                m.sentiment_score,
                m.sentiment_price_divergence,
                m.return_1d,
                m.return_5d,
                t.rsi_14,
                m.volatility_20d
            FROM ml_features m
            LEFT JOIN technical_indicators t
                ON m.symbol_ticker = t.symbol_ticker
                AND m.feature_date = t.indicator_date
            WHERE m.symbol_ticker = ?
                AND m.sentiment_score IS NOT NULL
            ORDER BY m.feature_date DESC
            LIMIT ?
        """

        analysis = pd.read_sql(query, conn, params=(ticker, days))
        conn.close()

        return analysis

    def backtest_signal(self, ticker: str, signal_date: str, days_forward: int = 30) -> dict:
        """
        Simple backtest for a sentiment signal

        Args:
            ticker: Stock ticker symbol
            signal_date: Date of signal generation
            days_forward: Number of days to hold position

        Returns:
            Dictionary with backtest results
        """
        conn = self._get_db_connection()

        query = """
            SELECT price_date, close_price
            FROM price_data
            WHERE symbol_ticker = ?
                AND price_date >= ?
            ORDER BY price_date
            LIMIT ?
        """

        prices = pd.read_sql(query, conn, params=(ticker, signal_date, days_forward + 1))
        conn.close()

        if len(prices) < 2:
            return {"error": "Insufficient price data"}

        entry_price = prices.iloc[0]['close_price']
        exit_price = prices.iloc[-1]['close_price']
        total_return = (exit_price - entry_price) / entry_price

        # Calculate max drawdown
        prices['cumulative_return'] = (prices['close_price'] - entry_price) / entry_price
        max_drawdown = prices['cumulative_return'].min()

        return {
            "ticker": ticker,
            "entry_date": prices.iloc[0]['price_date'],
            "exit_date": prices.iloc[-1]['price_date'],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "days_held": len(prices) - 1
        }


if __name__ == "__main__":
    # Example usage
    strategy = SentimentEventStrategy()

    # Generate signals
    signals = strategy.generate_signals(
        max_signals=10,
        min_divergence=0.2,
        rsi_oversold=30,
        rsi_overbought=70
    )

    if not signals.empty:
        print("\n" + "="*80)
        print("SENTIMENT EVENT SIGNALS")
        print("="*80)
        print(signals[['symbol_ticker', 'signal', 'sentiment_score',
                      'sentiment_price_divergence', 'rsi_14', 'confidence']].to_string(index=False))
        print("="*80)

        # Show detailed info for top signal
        if len(signals) > 0:
            top_signal = signals.iloc[0]
            print(f"\nTop Signal Details:")
            print(f"Ticker: {top_signal['symbol_ticker']}")
            print(f"Signal: {top_signal['signal']}")
            print(f"Sentiment Score: {top_signal['sentiment_score']:.2f}")
            print(f"Divergence: {top_signal['sentiment_price_divergence']:.2f}")
            print(f"RSI: {top_signal['rsi_14']:.1f}")
            print(f"Confidence: {top_signal['confidence']:.2%}")
    else:
        print("\nNo signals generated with current thresholds")
