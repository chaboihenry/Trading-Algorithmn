"""
Trading Signal Generator
========================

Generates trading signals from ml_features and pairs_statistics
Populates trading_signals table with actionable trading opportunities

Signal Types:
1. Pairs Signals: Based on z-score divergence (mean reversion)
2. Sentiment Signals: Based on sentiment-price divergence
3. Volatility Signals: Based on regime changes
4. Ensemble Signals: Combined signals with confidence scores

Output: trading_signals table with ranked trading opportunities
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generate trading signals from features and pair statistics"""

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db"):
        """Initialize signal generator"""
        self.db_path = db_path

        # Signal thresholds
        self.pairs_z_threshold = 2.0  # Z-score for pairs divergence
        self.sentiment_divergence_threshold = 0.5  # Sentiment-price divergence
        self.volatility_regime_change_threshold = 0.3  # Volatility regime change

        logger.info(f"Initialized SignalGenerator")
        logger.info(f"Database: {db_path}")

    def _get_db_connection(self) -> sqlite3.Connection:
        """Create database connection"""
        return sqlite3.connect(self.db_path)

    def _create_signals_table(self) -> None:
        """Create trading_signals table if not exists"""
        conn = self._get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trading_signals (
                signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_date DATE NOT NULL,
                symbol_ticker_1 TEXT,
                symbol_ticker_2 TEXT,
                signal_type TEXT NOT NULL,
                signal_direction TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                signal_strength REAL,

                -- Pairs signal details
                z_score REAL,
                cointegration_pvalue REAL,
                half_life REAL,
                spread_direction TEXT,

                -- Sentiment signal details
                sentiment_score REAL,
                sentiment_divergence REAL,

                -- Volatility signal details
                volatility_regime TEXT,
                volatility_change REAL,

                -- ML features
                rsi_14 REAL,
                macd REAL,
                return_5d REAL,

                -- Ensemble
                ensemble_score REAL,
                risk_score REAL,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(signal_date, symbol_ticker_1, symbol_ticker_2, signal_type)
            )
        """)

        conn.commit()
        conn.close()
        logger.info("trading_signals table created/verified")

    def generate_pairs_signals(self) -> List[Dict]:
        """
        Generate signals from pairs statistics

        Returns:
            List of pairs trading signals
        """
        conn = self._get_db_connection()

        query = """
            SELECT
                ps.symbol_ticker_1,
                ps.symbol_ticker_2,
                ps.stat_date,
                ps.spread_zscore,
                ps.cointegration_pvalue,
                ps.half_life_mean_reversion,
                ps.spread_direction,
                ps.spread_volatility_20d
            FROM pairs_statistics ps
            WHERE ps.stat_date = (SELECT MAX(stat_date) FROM pairs_statistics)
              AND ABS(ps.spread_zscore) > ?
              AND ps.cointegration_pvalue < 0.20
              AND ps.half_life_mean_reversion BETWEEN 3 AND 100
            ORDER BY ABS(ps.spread_zscore) DESC
        """

        df = pd.read_sql(query, conn, params=(self.pairs_z_threshold,))
        conn.close()

        signals = []

        for _, row in df.iterrows():
            # Determine signal direction
            # Negative z-score = Long Spread (buy stock1, sell stock2)
            # Positive z-score = Short Spread (sell stock1, buy stock2)
            direction = 'LONG_SPREAD' if row['spread_zscore'] < 0 else 'SHORT_SPREAD'

            # Calculate confidence based on z-score magnitude and cointegration
            z_score_conf = min(abs(row['spread_zscore']) / 4.0, 1.0)  # Normalize to 0-1
            coint_conf = 1.0 - row['cointegration_pvalue']  # Lower p-value = higher confidence
            half_life_conf = 1.0 / (1.0 + row['half_life_mean_reversion'] / 50.0)  # Faster mean reversion = better

            confidence = (z_score_conf * 0.5 + coint_conf * 0.3 + half_life_conf * 0.2)

            signals.append({
                'signal_date': row['stat_date'],
                'symbol_ticker_1': row['symbol_ticker_1'],
                'symbol_ticker_2': row['symbol_ticker_2'],
                'signal_type': 'PAIRS',
                'signal_direction': direction,
                'confidence_score': confidence,
                'signal_strength': abs(row['spread_zscore']),
                'z_score': row['spread_zscore'],
                'cointegration_pvalue': row['cointegration_pvalue'],
                'half_life': row['half_life_mean_reversion'],
                'spread_direction': row['spread_direction']
            })

        logger.info(f"Generated {len(signals)} pairs signals")
        return signals

    def generate_sentiment_signals(self) -> List[Dict]:
        """
        Generate signals from sentiment-price divergence

        Returns:
            List of sentiment trading signals
        """
        conn = self._get_db_connection()

        query = """
            SELECT
                mf.symbol_ticker,
                mf.feature_date,
                mf.sentiment_score,
                mf.sentiment_price_divergence,
                mf.return_5d,
                mf.rsi_14,
                mf.volatility_20d
            FROM ml_features mf
            WHERE mf.feature_date = (
                SELECT MAX(feature_date) FROM ml_features WHERE symbol_ticker = mf.symbol_ticker
            )
            AND ABS(mf.sentiment_price_divergence) > ?
            AND mf.sentiment_score IS NOT NULL
            ORDER BY ABS(mf.sentiment_price_divergence) DESC
        """

        df = pd.read_sql(query, conn, params=(self.sentiment_divergence_threshold,))
        conn.close()

        signals = []

        for _, row in df.iterrows():
            # Positive divergence: sentiment bullish but price down → BUY
            # Negative divergence: sentiment bearish but price up → SELL
            direction = 'BUY' if row['sentiment_price_divergence'] > 0 else 'SELL'

            # Calculate confidence
            divergence_conf = min(abs(row['sentiment_price_divergence']), 1.0)
            sentiment_conf = abs(row['sentiment_score']) if row['sentiment_score'] else 0.5

            confidence = (divergence_conf * 0.6 + sentiment_conf * 0.4)

            # Risk assessment based on volatility
            risk_score = row['volatility_20d'] if row['volatility_20d'] else 0.5

            signals.append({
                'signal_date': row['feature_date'],
                'symbol_ticker_1': row['symbol_ticker'],
                'symbol_ticker_2': None,
                'signal_type': 'SENTIMENT',
                'signal_direction': direction,
                'confidence_score': confidence,
                'signal_strength': abs(row['sentiment_price_divergence']),
                'sentiment_score': row['sentiment_score'],
                'sentiment_divergence': row['sentiment_price_divergence'],
                'rsi_14': row['rsi_14'],
                'return_5d': row['return_5d'],
                'risk_score': risk_score
            })

        logger.info(f"Generated {len(signals)} sentiment signals")
        return signals

    def generate_volatility_signals(self) -> List[Dict]:
        """
        Generate signals from volatility regime changes

        Returns:
            List of volatility trading signals
        """
        conn = self._get_db_connection()

        # Get current and previous volatility regimes
        query = """
            WITH ranked_features AS (
                SELECT
                    symbol_ticker,
                    feature_date,
                    volatility_20d,
                    volatility_regime,
                    volatility_20d_lag1,
                    rsi_14,
                    macd,
                    ROW_NUMBER() OVER (PARTITION BY symbol_ticker ORDER BY feature_date DESC) as rn
                FROM ml_features
                WHERE volatility_regime IS NOT NULL
            )
            SELECT
                symbol_ticker,
                feature_date,
                volatility_20d,
                volatility_regime,
                volatility_20d_lag1,
                rsi_14,
                macd
            FROM ranked_features
            WHERE rn = 1
            AND volatility_20d_lag1 IS NOT NULL
        """

        df = pd.read_sql(query, conn)
        conn.close()

        signals = []

        for _, row in df.iterrows():
            # Calculate volatility change
            vol_change = (row['volatility_20d'] - row['volatility_20d_lag1']) / (row['volatility_20d_lag1'] + 1e-8)

            # Only signal on significant regime changes
            if abs(vol_change) < self.volatility_regime_change_threshold:
                continue

            # Low to High volatility = Caution/Reduce exposure
            # High to Low volatility = Opportunity/Increase exposure
            if row['volatility_regime'] == 'high':
                direction = 'REDUCE'
            elif row['volatility_regime'] == 'low':
                direction = 'INCREASE'
            else:
                continue  # Skip normal regime

            confidence = min(abs(vol_change) / 0.5, 1.0)  # Normalize

            signals.append({
                'signal_date': row['feature_date'],
                'symbol_ticker_1': row['symbol_ticker'],
                'symbol_ticker_2': None,
                'signal_type': 'VOLATILITY',
                'signal_direction': direction,
                'confidence_score': confidence,
                'signal_strength': abs(vol_change),
                'volatility_regime': row['volatility_regime'],
                'volatility_change': vol_change,
                'rsi_14': row['rsi_14'],
                'macd': row['macd']
            })

        logger.info(f"Generated {len(signals)} volatility signals")
        return signals

    def generate_ensemble_signals(self, pairs_signals: List[Dict],
                                 sentiment_signals: List[Dict],
                                 volatility_signals: List[Dict]) -> List[Dict]:
        """
        Combine signals into ensemble with boosted confidence

        Args:
            pairs_signals: Pairs trading signals
            sentiment_signals: Sentiment signals
            volatility_signals: Volatility signals

        Returns:
            List of ensemble signals
        """
        ensemble_signals = []

        # Create lookup dictionaries for cross-referencing
        sentiment_lookup = {s['symbol_ticker_1']: s for s in sentiment_signals}
        volatility_lookup = {s['symbol_ticker_1']: s for s in volatility_signals}

        # Enhance pairs signals with sentiment/volatility
        for pairs_sig in pairs_signals:
            ticker1 = pairs_sig['symbol_ticker_1']
            ticker2 = pairs_sig['symbol_ticker_2']

            # Check if we have supporting signals
            sentiment1 = sentiment_lookup.get(ticker1)
            sentiment2 = sentiment_lookup.get(ticker2)
            volatility1 = volatility_lookup.get(ticker1)
            volatility2 = volatility_lookup.get(ticker2)

            # Calculate ensemble score
            base_score = pairs_sig['confidence_score']
            bonus = 0.0

            # Sentiment confirmation
            if sentiment1 and pairs_sig['signal_direction'] == 'LONG_SPREAD':
                if sentiment1['signal_direction'] == 'BUY':
                    bonus += 0.1
            if sentiment2 and pairs_sig['signal_direction'] == 'SHORT_SPREAD':
                if sentiment2['signal_direction'] == 'SELL':
                    bonus += 0.1

            # Volatility confirmation (prefer trades in low volatility)
            if volatility1 and volatility1['volatility_regime'] == 'low':
                bonus += 0.05
            if volatility2 and volatility2['volatility_regime'] == 'low':
                bonus += 0.05

            ensemble_score = min(base_score + bonus, 1.0)

            # Risk score (average volatility)
            risk_scores = []
            if volatility1:
                risk_scores.append(volatility1['signal_strength'])
            if volatility2:
                risk_scores.append(volatility2['signal_strength'])
            risk_score = np.mean(risk_scores) if risk_scores else 0.5

            ensemble_sig = pairs_sig.copy()
            ensemble_sig['ensemble_score'] = ensemble_score
            ensemble_sig['risk_score'] = risk_score
            ensemble_signals.append(ensemble_sig)

        logger.info(f"Generated {len(ensemble_signals)} ensemble signals")
        return ensemble_signals

    def save_signals(self, signals: List[Dict]) -> int:
        """
        Save signals to database

        Args:
            signals: List of signal dictionaries

        Returns:
            Number of signals saved
        """
        if not signals:
            logger.warning("No signals to save")
            return 0

        conn = self._get_db_connection()
        cursor = conn.cursor()

        saved_count = 0

        for signal in signals:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO trading_signals (
                        signal_date, symbol_ticker_1, symbol_ticker_2,
                        signal_type, signal_direction, confidence_score, signal_strength,
                        z_score, cointegration_pvalue, half_life, spread_direction,
                        sentiment_score, sentiment_divergence,
                        volatility_regime, volatility_change,
                        rsi_14, macd, return_5d,
                        ensemble_score, risk_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal.get('signal_date'),
                    signal.get('symbol_ticker_1'),
                    signal.get('symbol_ticker_2'),
                    signal.get('signal_type'),
                    signal.get('signal_direction'),
                    signal.get('confidence_score'),
                    signal.get('signal_strength'),
                    signal.get('z_score'),
                    signal.get('cointegration_pvalue'),
                    signal.get('half_life'),
                    signal.get('spread_direction'),
                    signal.get('sentiment_score'),
                    signal.get('sentiment_divergence'),
                    signal.get('volatility_regime'),
                    signal.get('volatility_change'),
                    signal.get('rsi_14'),
                    signal.get('macd'),
                    signal.get('return_5d'),
                    signal.get('ensemble_score'),
                    signal.get('risk_score')
                ))
                saved_count += 1
            except Exception as e:
                logger.error(f"Error saving signal: {e}")

        conn.commit()
        conn.close()

        logger.info(f"✓ Saved {saved_count} signals to database")
        return saved_count

    def run(self) -> None:
        """Generate all trading signals"""
        logger.info("="*60)
        logger.info("TRADING SIGNAL GENERATION STARTED")
        logger.info("="*60)

        # Create table
        self._create_signals_table()

        # Generate signals from each source
        pairs_signals = self.generate_pairs_signals()
        sentiment_signals = self.generate_sentiment_signals()
        volatility_signals = self.generate_volatility_signals()

        # Generate ensemble signals
        ensemble_signals = self.generate_ensemble_signals(
            pairs_signals, sentiment_signals, volatility_signals
        )

        # Combine all signals
        all_signals = ensemble_signals + sentiment_signals + volatility_signals

        # Save to database
        saved = self.save_signals(all_signals)

        logger.info("="*60)
        logger.info("TRADING SIGNAL GENERATION COMPLETED")
        logger.info("="*60)
        logger.info(f"Pairs signals: {len(pairs_signals)}")
        logger.info(f"Sentiment signals: {len(sentiment_signals)}")
        logger.info(f"Volatility signals: {len(volatility_signals)}")
        logger.info(f"Ensemble signals: {len(ensemble_signals)}")
        logger.info(f"Total saved: {saved}")
        logger.info("="*60)


if __name__ == "__main__":
    generator = SignalGenerator()
    generator.run()