"""Feature Engineering Pipeline for ML Trading Models.

Extracts comprehensive features from database tables:
- technical_indicators: RSI, MACD, Bollinger Bands, ADX, SMAs
- volatility_metrics: Multiple volatility estimators, trends
- sentiment_data: News sentiment scores
- fundamental_data: Financial ratios, earnings
- market_conditions: VIX, economic indicators
"""

import logging
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Extract and engineer features from database for ML models."""

    def __init__(self, db_path: str):
        """Initialize feature engineer.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        logger.info(f"✅ Feature engineer initialized (DB: {db_path})")

    def extract_features(self, symbol: str, as_of_date: Optional[str] = None) -> Optional[Dict[str, float]]:
        """Extract comprehensive feature set for a symbol.

        Args:
            symbol: Stock ticker
            as_of_date: Date to extract features for (YYYY-MM-DD), default=latest

        Returns:
            Dict of feature_name -> value, or None if insufficient data
        """
        features = {}

        # 1. Technical indicators (MACD, RSI, Bollinger Bands, ADX, SMAs)
        technical_features = self._get_technical_features(symbol, as_of_date)
        if technical_features:
            features.update(technical_features)

        # 2. Volatility metrics (multiple estimators)
        volatility_features = self._get_volatility_features(symbol, as_of_date)
        if volatility_features:
            features.update(volatility_features)

        # 3. Price momentum and trends
        momentum_features = self._get_momentum_features(symbol, as_of_date)
        if momentum_features:
            features.update(momentum_features)

        # 4. Volume analysis
        volume_features = self._get_volume_features(symbol, as_of_date)
        if volume_features:
            features.update(volume_features)

        # 5. Sentiment scores (if available)
        sentiment_features = self._get_sentiment_features(symbol, as_of_date)
        if sentiment_features:
            features.update(sentiment_features)

        # Minimum feature count check
        if len(features) < 20:
            logger.warning(f"Insufficient features for {symbol}: {len(features)} < 20")
            return None

        return features

    def _get_technical_features(self, symbol: str, as_of_date: Optional[str] = None) -> Dict[str, float]:
        """Extract technical indicators."""
        features = {}

        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                SELECT
                    rsi_14, rsi_7,
                    macd, macd_signal, macd_histogram,
                    bb_upper, bb_middle, bb_lower, bb_width, bb_percent,
                    adx_14, plus_di, minus_di,
                    sma_10, sma_20, sma_50, sma_200,
                    ema_12, ema_26,
                    stochastic_k, stochastic_d,
                    atr_14, atr_20
                FROM technical_indicators
                WHERE symbol_ticker = ?
                """

                if as_of_date:
                    query += " AND indicator_date <= ?"
                    params = (symbol, as_of_date)
                else:
                    params = (symbol,)

                query += " ORDER BY indicator_date DESC LIMIT 1"

                df = pd.read_sql(query, conn, params=params)

                if not df.empty:
                    for col in df.columns:
                        val = df[col].iloc[0]
                        if val is not None and not np.isnan(val):
                            features[f'tech_{col}'] = float(val)

        except Exception as e:
            logger.debug(f"Error fetching technical features: {e}")

        return features

    def _get_volatility_features(self, symbol: str, as_of_date: Optional[str] = None) -> Dict[str, float]:
        """Extract volatility metrics."""
        features = {}

        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                SELECT
                    close_to_close_vol_10d, close_to_close_vol_20d, close_to_close_vol_60d,
                    parkinson_vol_10d, parkinson_vol_20d,
                    garman_klass_vol_10d, garman_klass_vol_20d,
                    rogers_satchell_vol_10d, rogers_satchell_vol_20d,
                    yang_zhang_vol_10d, yang_zhang_vol_20d,
                    realized_vol_percentile_1y, realized_vol_percentile_3y,
                    volatility_of_volatility_20d,
                    vol_clustering_index,
                    volatility_trend,
                    volatility_acceleration
                FROM volatility_metrics
                WHERE symbol_ticker = ?
                """

                if as_of_date:
                    query += " AND vol_date <= ?"
                    params = (symbol, as_of_date)
                else:
                    params = (symbol,)

                query += " ORDER BY vol_date DESC LIMIT 1"

                df = pd.read_sql(query, conn, params=params)

                if not df.empty:
                    for col in df.columns:
                        if col == 'volatility_trend':
                            # Encode text as numeric
                            trend = df[col].iloc[0]
                            if trend == 'increasing':
                                features['vol_trend'] = 1.0
                            elif trend == 'decreasing':
                                features['vol_trend'] = -1.0
                            else:
                                features['vol_trend'] = 0.0
                        else:
                            val = df[col].iloc[0]
                            if val is not None and not np.isnan(val):
                                features[f'vol_{col}'] = float(val)

        except Exception as e:
            logger.debug(f"Error fetching volatility features: {e}")

        return features

    def _get_momentum_features(self, symbol: str, as_of_date: Optional[str] = None) -> Dict[str, float]:
        """Calculate price momentum features."""
        features = {}

        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                SELECT close, indicator_date
                FROM technical_indicators
                WHERE symbol_ticker = ?
                """

                if as_of_date:
                    query += " AND indicator_date <= ?"
                    params = (symbol, as_of_date)
                else:
                    params = (symbol,)

                query += " ORDER BY indicator_date DESC LIMIT 60"

                df = pd.read_sql(query, conn, params=params)

                if len(df) >= 20:
                    prices = df['close'].values

                    # Returns over different windows
                    if len(prices) >= 5:
                        features['momentum_5d_return'] = (prices[0] / prices[4] - 1) if prices[4] != 0 else 0
                    if len(prices) >= 10:
                        features['momentum_10d_return'] = (prices[0] / prices[9] - 1) if prices[9] != 0 else 0
                    if len(prices) >= 20:
                        features['momentum_20d_return'] = (prices[0] / prices[19] - 1) if prices[19] != 0 else 0
                    if len(prices) >= 60:
                        features['momentum_60d_return'] = (prices[0] / prices[59] - 1) if prices[59] != 0 else 0

                    # Price acceleration (change in momentum)
                    if len(prices) >= 10:
                        recent_momentum = (prices[0] / prices[4] - 1) if prices[4] != 0 else 0
                        past_momentum = (prices[5] / prices[9] - 1) if prices[9] != 0 else 0
                        features['momentum_acceleration'] = recent_momentum - past_momentum

        except Exception as e:
            logger.debug(f"Error calculating momentum features: {e}")

        return features

    def _get_volume_features(self, symbol: str, as_of_date: Optional[str] = None) -> Dict[str, float]:
        """Extract volume-based features."""
        features = {}

        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                SELECT obv, volume_sma_20, volume_ratio
                FROM technical_indicators
                WHERE symbol_ticker = ?
                """

                if as_of_date:
                    query += " AND indicator_date <= ?"
                    params = (symbol, as_of_date)
                else:
                    params = (symbol,)

                query += " ORDER BY indicator_date DESC LIMIT 1"

                df = pd.read_sql(query, conn, params=params)

                if not df.empty:
                    for col in df.columns:
                        val = df[col].iloc[0]
                        if val is not None and not np.isnan(val):
                            features[f'volume_{col}'] = float(val)

        except Exception as e:
            logger.debug(f"Error fetching volume features: {e}")

        return features

    def _get_sentiment_features(self, symbol: str, as_of_date: Optional[str] = None) -> Dict[str, float]:
        """Extract sentiment scores."""
        features = {}

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get sentiment from last 7 days
                query = """
                SELECT avg(sentiment_score) as avg_sentiment,
                       count(*) as sentiment_count
                FROM sentiment_data
                WHERE symbol_ticker = ?
                """

                if as_of_date:
                    query += " AND date(published_at) <= date(?)"
                    query += " AND date(published_at) >= date(?, '-7 days')"
                    params = (symbol, as_of_date, as_of_date)
                else:
                    query += " AND date(published_at) >= date('now', '-7 days')"
                    params = (symbol,)

                df = pd.read_sql(query, conn, params=params)

                if not df.empty and df['avg_sentiment'].iloc[0] is not None:
                    features['sentiment_avg_7d'] = float(df['avg_sentiment'].iloc[0])
                    features['sentiment_count_7d'] = float(df['sentiment_count'].iloc[0])

        except Exception as e:
            logger.debug(f"Error fetching sentiment features: {e}")

        return features

    def extract_features_bulk(self, symbols: List[str], as_of_date: Optional[str] = None) -> pd.DataFrame:
        """Extract features for multiple symbols.

        Args:
            symbols: List of stock tickers
            as_of_date: Date to extract features for

        Returns:
            DataFrame with rows=symbols, columns=features
        """
        logger.info(f"Extracting features for {len(symbols)} symbols...")

        feature_data = []
        for symbol in symbols:
            features = self.extract_features(symbol, as_of_date)
            if features:
                features['symbol'] = symbol
                feature_data.append(features)

        if not feature_data:
            logger.warning("No features extracted for any symbols")
            return pd.DataFrame()

        df = pd.DataFrame(feature_data)
        df.set_index('symbol', inplace=True)

        # Fill NaN with 0 (missing features)
        df.fillna(0, inplace=True)

        logger.info(f"✅ Extracted {len(df)} symbol features with {len(df.columns)} features each")

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of all possible feature names.

        Returns:
            List of feature names
        """
        # This is a representative list - actual features may vary by symbol
        return [
            # Technical indicators
            'tech_rsi_14', 'tech_rsi_7',
            'tech_macd', 'tech_macd_signal', 'tech_macd_histogram',
            'tech_bb_upper', 'tech_bb_middle', 'tech_bb_lower', 'tech_bb_width', 'tech_bb_percent',
            'tech_adx_14', 'tech_plus_di', 'tech_minus_di',
            'tech_sma_10', 'tech_sma_20', 'tech_sma_50', 'tech_sma_200',
            'tech_ema_12', 'tech_ema_26',
            'tech_stochastic_k', 'tech_stochastic_d',
            'tech_atr_14', 'tech_atr_20',

            # Volatility
            'vol_close_to_close_vol_10d', 'vol_close_to_close_vol_20d', 'vol_close_to_close_vol_60d',
            'vol_parkinson_vol_10d', 'vol_parkinson_vol_20d',
            'vol_garman_klass_vol_10d', 'vol_garman_klass_vol_20d',
            'vol_yang_zhang_vol_10d', 'vol_yang_zhang_vol_20d',
            'vol_realized_vol_percentile_1y', 'vol_realized_vol_percentile_3y',
            'vol_volatility_of_volatility_20d',
            'vol_trend',
            'vol_volatility_acceleration',

            # Momentum
            'momentum_5d_return', 'momentum_10d_return', 'momentum_20d_return', 'momentum_60d_return',
            'momentum_acceleration',

            # Volume
            'volume_obv', 'volume_volume_sma_20', 'volume_volume_ratio',

            # Sentiment
            'sentiment_avg_7d', 'sentiment_count_7d'
        ]
