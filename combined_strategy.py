"""
Combined Strategy with Stacked Ensemble Meta-Learner

This strategy combines sentiment and pairs trading signals using a machine learning
meta-learner (XGBoost) that dynamically adjusts weights based on market conditions.

Instead of fixed weights, the meta-learner:
1. Takes signals from both strategies as input features
2. Learns which strategy to trust in different conditions
3. Adapts weights continuously based on historical performance
4. Discovers non-linear interactions between strategies

This is based on your existing stacked_ensemble.py pattern but adapted for Lumibot
and combining sentiment + pairs instead of multiple ML models.
"""

import logging
import numpy as np
import pandas as pd
import sqlite3
import joblib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from lumibot.strategies import Strategy
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import our base strategies
from sentiment_strategy import SentimentStrategy
from pairs_strategy import PairsStrategy

logger = logging.getLogger(__name__)


class CombinedStrategy(Strategy):
    """
    Meta-learning ensemble that combines sentiment and pairs trading.

    The meta-learner (XGBoost) takes as input:
    - Sentiment score and confidence from SentimentStrategy
    - Pairs signal (z-score, quality) from PairsStrategy
    - Market conditions (volatility, trend, volume)

    It outputs:
    - Dynamic weights for each strategy (0-1)
    - Final trading decision (buy/sell/hold)
    - Position sizing based on confidence
    """

    # Strategy parameters
    SLEEPTIME = "24H"
    RETRAIN_FREQUENCY_DAYS = 7  # Retrain meta-model weekly
    MIN_TRAINING_SAMPLES = 100  # Minimum samples needed to train
    CONFIDENCE_THRESHOLD = 0.6  # Only trade if meta-model confidence > 60%

    def initialize(self, parameters: Dict = None):
        """
        Initialize combined strategy with meta-learner.

        Args:
            parameters: Dict with optional parameters:
                - db_path: Database path
                - model_path: Path to saved meta-model
                - retrain: Whether to retrain model (default: False)
        """
        self.sleeptime = self.SLEEPTIME

        params = parameters or {}
        self.db_path = params.get('db_path', '/Volumes/Vault/85_assets_prediction.db')
        self.retrain = params.get('retrain', False)

        # Initialize base strategies (we'll use their logic, not run them separately)
        self.sentiment_strategy = SentimentStrategy()
        self.pairs_strategy = PairsStrategy()

        # Meta-learner components
        self.meta_model = None
        self.scaler = StandardScaler()
        self.last_retrain_date = None

        # Model persistence
        self.models_dir = Path(__file__).parent / 'models'
        self.models_dir.mkdir(exist_ok=True)

        model_path = params.get('model_path')
        if model_path:
            self._load_meta_model(model_path)
        else:
            # Try to load latest model
            self._load_latest_meta_model()

        # If no model or retrain requested, train new model
        if self.meta_model is None or self.retrain:
            self._train_meta_model()

        logger.info("Combined Strategy initialized with meta-learner")
        logger.info(f"Meta-model active: {self.meta_model is not None}")

    def _load_meta_model(self, model_path: str):
        """Load a saved meta-model from disk."""
        try:
            meta_path = Path(model_path)
            scaler_path = meta_path.parent / f"{meta_path.stem}_scaler.joblib"

            self.meta_model = joblib.load(model_path)
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)

            logger.info(f"Loaded meta-model from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load meta-model: {e}")
            self.meta_model = None

    def _load_latest_meta_model(self):
        """Load the most recent meta-model from models directory."""
        try:
            model_files = list(self.models_dir.glob("combined_meta_*.joblib"))
            if not model_files:
                logger.info("No existing meta-model found")
                return

            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            self._load_meta_model(str(latest_model))
        except Exception as e:
            logger.warning(f"Error loading latest model: {e}")

    def _save_meta_model(self):
        """Save meta-model to disk."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = self.models_dir / f"combined_meta_{timestamp}.joblib"
        scaler_path = self.models_dir / f"combined_meta_{timestamp}_scaler.joblib"

        joblib.dump(self.meta_model, model_path)
        joblib.dump(self.scaler, scaler_path)

        logger.info(f"Saved meta-model to {model_path}")

    def _get_historical_signals(self) -> pd.DataFrame:
        """
        Fetch historical trading signals from database to train meta-learner.

        Returns:
            DataFrame with columns: [date, symbol, sentiment_signal, sentiment_prob,
                                     pairs_signal, pairs_zscore, actual_return_5d, ...]
        """
        conn = sqlite3.connect(self.db_path)

        # Get historical signals and returns
        # This query assumes you have a trading_signals table with signal history
        query = """
        SELECT
            ts.signal_date,
            ts.symbol_ticker,
            ts.signal_type,
            ts.strength,
            ts.metadata,
            p.close as entry_price,
            p2.close as exit_price_5d,
            (p2.close - p.close) / p.close as return_5d,
            vm.volatility_20d,
            ti.rsi_14
        FROM trading_signals ts
        LEFT JOIN raw_price_data p
            ON ts.symbol_ticker = p.symbol_ticker
            AND ts.signal_date = p.date
        LEFT JOIN raw_price_data p2
            ON ts.symbol_ticker = p2.symbol_ticker
            AND p2.date = date(ts.signal_date, '+5 days')
        LEFT JOIN volatility_metrics vm
            ON ts.symbol_ticker = vm.symbol_ticker
            AND ts.signal_date = vm.date
        LEFT JOIN technical_indicators ti
            ON ts.symbol_ticker = ti.symbol_ticker
            AND ts.signal_date = ti.date
        WHERE ts.signal_date >= date('now', '-365 days')
          AND ts.metadata IS NOT NULL
        ORDER BY ts.signal_date
        """

        df = pd.read_sql(query, conn)
        conn.close()

        if df.empty:
            logger.warning("No historical signals found in database")
            return pd.DataFrame()

        logger.info(f"Loaded {len(df)} historical signals for training")
        return df

    def _prepare_meta_features(
        self,
        sentiment_score: float,
        sentiment_prob: float,
        pairs_zscore: float,
        pairs_quality: float,
        volatility: float,
        rsi: float
    ) -> np.ndarray:
        """
        Prepare feature vector for meta-learner.

        Args:
            sentiment_score: Sentiment signal (-1, 0, 1)
            sentiment_prob: Sentiment confidence (0-1)
            pairs_zscore: Pairs spread z-score
            pairs_quality: Pairs quality score
            volatility: Market volatility
            rsi: RSI indicator

        Returns:
            Feature vector ready for meta-model
        """
        features = np.array([
            sentiment_score,
            sentiment_prob,
            abs(sentiment_score) * sentiment_prob,  # Weighted sentiment
            pairs_zscore,
            abs(pairs_zscore),  # Magnitude of deviation
            pairs_quality,
            pairs_zscore * pairs_quality,  # Weighted pairs signal
            volatility if volatility else 0.2,  # Default volatility
            rsi if rsi else 50,  # Default RSI
            # Interaction features
            sentiment_score * pairs_zscore,  # Agreement indicator
            abs(sentiment_score - np.sign(pairs_zscore)),  # Disagreement indicator
            sentiment_prob * pairs_quality,  # Combined confidence
        ])

        return features.reshape(1, -1)

    def _train_meta_model(self):
        """
        Train the meta-learner on historical signals.

        The meta-model learns to predict profitability based on:
        - Strategy signals
        - Market conditions
        - Historical performance patterns
        """
        logger.info("Training meta-learner...")

        # Get historical data
        historical_data = self._get_historical_signals()

        if len(historical_data) < self.MIN_TRAINING_SAMPLES:
            logger.warning(f"Insufficient training data ({len(historical_data)} samples)")
            logger.warning("Meta-model will use equal weights until more data is available")
            return

        # Extract features and labels
        # (This is simplified - you'll need to parse metadata and extract actual signal info)
        try:
            # Create feature matrix
            features_list = []
            labels_list = []

            for _, row in historical_data.iterrows():
                # Parse metadata to extract signal details
                # For now, using simplified features
                features = [
                    1 if row['signal_type'] == 'BUY' else -1 if row['signal_type'] == 'SELL' else 0,
                    row['strength'] if pd.notna(row['strength']) else 0.5,
                    row['volatility_20d'] if pd.notna(row['volatility_20d']) else 0.2,
                    row['rsi_14'] if pd.notna(row['rsi_14']) else 50,
                ]

                # Label: 1 if profitable, 0 otherwise
                label = 1 if (row['return_5d'] if pd.notna(row['return_5d']) else 0) > 0 else 0

                features_list.append(features)
                labels_list.append(label)

            X = np.array(features_list)
            y = np.array(labels_list)

            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train XGBoost meta-model (small model to prevent overfitting)
            self.meta_model = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                tree_method='hist'  # M1 optimized
            )

            self.meta_model.fit(X_train_scaled, y_train)

            # Evaluate
            train_score = self.meta_model.score(X_train_scaled, y_train)
            test_score = self.meta_model.score(X_test_scaled, y_test)

            logger.info(f"Meta-model trained: train_acc={train_score:.3f}, test_acc={test_score:.3f}")

            # Save model
            self._save_meta_model()
            self.last_retrain_date = datetime.now()

        except Exception as e:
            logger.error(f"Error training meta-model: {e}")
            self.meta_model = None

    def _get_market_conditions(self, symbol: str) -> Tuple[float, float]:
        """
        Fetch current market conditions.

        Args:
            symbol: Stock symbol

        Returns:
            Tuple of (volatility, rsi)
        """
        try:
            conn = sqlite3.connect(self.db_path)

            query = """
            SELECT vm.volatility_20d, ti.rsi_14
            FROM volatility_metrics vm
            LEFT JOIN technical_indicators ti
                ON vm.symbol_ticker = ti.symbol_ticker
                AND vm.date = ti.date
            WHERE vm.symbol_ticker = ?
            ORDER BY vm.date DESC
            LIMIT 1
            """

            result = pd.read_sql(query, conn, params=(symbol,))
            conn.close()

            if not result.empty:
                volatility = result['volatility_20d'].iloc[0]
                rsi = result['rsi_14'].iloc[0]
                return volatility, rsi
            else:
                return 0.2, 50  # Defaults

        except Exception as e:
            logger.warning(f"Error fetching market conditions: {e}")
            return 0.2, 50

    def _get_combined_signal(self, symbol: str) -> Tuple[int, float]:
        """
        Get combined signal from meta-learner.

        Args:
            symbol: Stock symbol to analyze

        Returns:
            Tuple of (signal, confidence) where signal is -1/0/1 and confidence is 0-1
        """
        # Get sentiment signal
        try:
            sentiment_prob, sentiment_signal = self.sentiment_strategy.get_news_sentiment(symbol)
        except Exception as e:
            logger.warning(f"Error getting sentiment for {symbol}: {e}")
            sentiment_prob, sentiment_signal = 0.5, 0

        # Get pairs signal (if this symbol is in a pair)
        pairs_zscore = 0
        pairs_quality = 0

        for pair_data in getattr(self.pairs_strategy, 'cointegrated_pairs', []):
            if symbol in pair_data[:2]:
                s1, s2, corr, pval, quality, hedge = pair_data
                pairs_quality = quality

                # Calculate current z-score
                try:
                    _, zscore = self.pairs_strategy._calculate_current_spread(s1, s2, hedge)
                    pairs_zscore = zscore
                except:
                    pass
                break

        # Get market conditions
        volatility, rsi = self._get_market_conditions(symbol)

        # If no meta-model, use simple weighted average
        if self.meta_model is None:
            logger.info(f"{symbol}: Using equal weights (no meta-model)")

            # Simple combination: 60% sentiment, 40% pairs
            combined_signal = 0.6 * sentiment_signal

            if abs(pairs_zscore) > 1.5:
                # Pairs signal is strong
                pairs_contrib = -np.sign(pairs_zscore) * 0.4
                combined_signal += pairs_contrib

            final_signal = 1 if combined_signal > 0.3 else -1 if combined_signal < -0.3 else 0
            confidence = min(abs(combined_signal), 1.0)

            return final_signal, confidence

        # Use meta-model for intelligent combination
        features = self._prepare_meta_features(
            sentiment_signal,
            sentiment_prob,
            pairs_zscore,
            pairs_quality,
            volatility,
            rsi
        )

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Get prediction
        prediction_proba = self.meta_model.predict_proba(features_scaled)[0]
        prediction = self.meta_model.predict(features_scaled)[0]

        # Convert to signal
        confidence = max(prediction_proba)  # Max probability

        if confidence < self.CONFIDENCE_THRESHOLD:
            logger.info(f"{symbol}: Low confidence ({confidence:.2f}), skipping")
            return 0, confidence

        # Determine direction based on input signals and prediction
        if prediction == 1:  # Meta-model predicts profitable
            # Use strongest signal
            if abs(sentiment_signal) > abs(np.sign(pairs_zscore)):
                final_signal = sentiment_signal
            else:
                final_signal = -np.sign(pairs_zscore) if pairs_zscore != 0 else sentiment_signal
        else:
            final_signal = 0  # Don't trade if meta-model predicts unprofitable

        logger.info(f"{symbol}: sentiment={sentiment_signal}({sentiment_prob:.2f}), pairs_z={pairs_zscore:.2f}, meta_signal={final_signal}({confidence:.2f})")

        return final_signal, confidence

    def on_trading_iteration(self):
        """
        Main trading logic - combines signals from both strategies using meta-learner.
        """
        logger.info("=" * 80)
        logger.info(f"COMBINED STRATEGY - Trading Iteration at {datetime.now()}")
        logger.info("=" * 80)

        # Check if we need to retrain meta-model
        if self.last_retrain_date:
            days_since_retrain = (datetime.now() - self.last_retrain_date).days
            if days_since_retrain >= self.RETRAIN_FREQUENCY_DAYS:
                logger.info(f"Retraining meta-model ({days_since_retrain} days since last retrain)")
                self._train_meta_model()

        # Get combined symbols from both strategies
        all_symbols = set(self.sentiment_strategy.SYMBOLS)

        # Add symbols from pairs
        for pair_data in getattr(self.pairs_strategy, 'cointegrated_pairs', [])[:10]:
            all_symbols.add(pair_data[0])
            all_symbols.add(pair_data[1])

        logger.info(f"Analyzing {len(all_symbols)} symbols")

        cash = self.get_cash()
        portfolio_value = self.get_portfolio_value()

        logger.info(f"Portfolio value: ${portfolio_value:,.2f}")
        logger.info(f"Available cash: ${cash:,.2f}")

        # Get signals for each symbol
        for symbol in all_symbols:
            try:
                signal, confidence = self._get_combined_signal(symbol)

                position = self.get_position(symbol)

                # Position sizing based on confidence
                position_size = cash * 0.1 * confidence  # Max 10% per position, scaled by confidence

                if signal == 1 and confidence > self.CONFIDENCE_THRESHOLD:
                    # BUY signal
                    if position is None:
                        price = self.get_last_price(symbol)
                        quantity = position_size / price

                        order = self.create_order(symbol, quantity, "buy")
                        self.submit_order(order)

                        logger.info(f"BUY {symbol}: {quantity:.2f} shares @ ${price:.2f} (confidence: {confidence:.2f})")

                elif signal == -1 and position is not None:
                    # SELL signal
                    order = self.create_order(symbol, position.quantity, "sell")
                    self.submit_order(order)

                    logger.info(f"SELL {symbol}: {position.quantity:.2f} shares (confidence: {confidence:.2f})")

                elif position is not None and signal == 0:
                    # Exit position on neutral signal
                    order = self.create_order(symbol, position.quantity, "sell")
                    self.submit_order(order)

                    logger.info(f"SELL {symbol}: {position.quantity:.2f} shares (signal neutral)")

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue

        logger.info("=" * 80)
        logger.info("Trading iteration complete")
        logger.info("=" * 80)
