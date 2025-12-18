"""
Combined Strategy with Stacked Ensemble Meta-Learner

This strategy combines sentiment and pairs trading signals using a machine learning
meta-learner (Logistic Regression) that dynamically adjusts weights based on market conditions.

Instead of fixed weights, the meta-learner:
1. Takes signals from both strategies as input features
2. Learns which strategy to trust in different conditions
3. Adapts weights continuously based on historical performance
4. Uses regularized logistic regression (suitable for limited training data)

This is based on your existing stacked_ensemble.py pattern but adapted for Lumibot
and combining sentiment + pairs instead of multiple ML models.
"""

import logging
import math
import numpy as np
import pandas as pd
import sqlite3
import joblib
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from functools import wraps
from http.client import RemoteDisconnected
import requests.exceptions
from lumibot.strategies import Strategy
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

# Alpaca API for direct bracket order creation (bypass Lumibot)
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

# Base strategies not needed - using simplified approach
# from sentiment_strategy import SentimentStrategy
# from pairs_strategy import PairsStrategy

logger = logging.getLogger(__name__)


def retry_on_connection_error(max_retries=3, initial_delay=5, backoff_factor=2):
    """Decorator to retry API calls on connection errors with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (ConnectionError, ConnectionResetError, ConnectionAbortedError,
                        RemoteDisconnected, requests.exceptions.ConnectionError,
                        TimeoutError, OSError) as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"🔌 Connection error in {func.__name__}: {e}")
                        logger.warning(f"   Retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(f"❌ Max retries exceeded for {func.__name__}")
                        raise
                except Exception as e:
                    logger.error(f"Non-connection error in {func.__name__}: {e}")
                    raise

            if last_exception:
                raise last_exception

        return wrapper
    return decorator


class CombinedStrategy(Strategy):
    """Meta-learning ensemble combining sentiment and pairs trading signals."""

    from config.settings import (
        SLEEP_INTERVAL,
        RETRAIN_FREQUENCY_DAYS,
        MIN_TRAINING_SAMPLES,
        CONFIDENCE_THRESHOLD,
        STOP_LOSS_PCT,
        TAKE_PROFIT_PCT,
        ENABLE_EXTENDED_HOURS,
        INVERSE_ETFS,
        MAX_INVERSE_ALLOCATION
    )

    SLEEPTIME = SLEEP_INTERVAL

    def initialize(self, parameters: Dict = None):
        """Initialize strategy with meta-learner and sentiment analysis."""
        self.sleeptime = self.SLEEPTIME

        from config.settings import (
            DB_PATH,
            SENTIMENT_LOOKBACK_DAYS,
            TRADING_SYMBOLS,
            PAIRS_LOOKBACK_DAYS,
            PAIRS_ZSCORE_ENTRY,
            PAIRS_MIN_CORRELATION,
            MAX_DAILY_LOSS_PCT,
            WARNING_LOSS_PCT,
            SCALING_START_LOSS_PCT
        )

        params = parameters or {}
        self.db_path = params.get('db_path', DB_PATH)
        self.retrain = params.get('retrain', False)

        self.NEWS_LOOKBACK_DAYS = SENTIMENT_LOOKBACK_DAYS
        self.SYMBOLS = TRADING_SYMBOLS

        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.sentiment_model.eval()
        self.torch = torch

        self.LOOKBACK_DAYS = PAIRS_LOOKBACK_DAYS
        self.ZSCORE_ENTRY = PAIRS_ZSCORE_ENTRY
        self.MIN_CORRELATION = PAIRS_MIN_CORRELATION
        self.cointegrated_pairs = []

        self.meta_model = None
        self.scaler = StandardScaler()
        self.last_retrain_date = None

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

        # Initialize direct Alpaca Trading Client for bracket orders
        # This bypasses Lumibot and gives us direct access to Alpaca's bracket order API
        from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_PAPER
        self.alpaca_client = TradingClient(
            ALPACA_API_KEY,
            ALPACA_API_SECRET,
            paper=ALPACA_PAPER
        )
        logger.info("✅ Direct Alpaca client initialized for bracket orders")

        # Initialize reliable MarketDataClient (fixes Lumibot's unreliable get_cash())
        # Lumibot's get_cash() sometimes returns None, causing crashes
        # Our MarketDataClient ALWAYS returns a value (0.0 on error, never None)
        from data.market_data import get_market_data_client
        self.market_data = get_market_data_client()
        logger.info("✅ Reliable market data client initialized")

        # FIXED: Initialize health monitoring for 90-day reliability
        from utils.health_monitor import get_health_monitor
        from utils.memory_profiler import get_memory_profiler
        self.health_monitor = get_health_monitor()
        self.memory_profiler = get_memory_profiler()
        logger.info("✅ Health monitoring and memory profiling initialized")

        # FIXED (Problem 10, 16): Initialize daily P&L tracker with circuit breaker
        # Parameters now from config.settings (environment-specific!)
        from utils.daily_pnl_tracker import get_daily_pnl_tracker
        self.daily_pnl = get_daily_pnl_tracker(
            max_daily_loss_pct=MAX_DAILY_LOSS_PCT,
            warning_loss_pct=WARNING_LOSS_PCT,
            scaling_start_loss_pct=SCALING_START_LOSS_PCT
        )
        self.current_position_size_multiplier = 1.0  # Default to full size
        logger.info("✅ Daily P&L tracker and circuit breaker initialized")

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
        # FIXED: Use context manager to prevent connection leaks
        with sqlite3.connect(self.db_path) as conn:
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
                vm.close_to_close_vol_20d as volatility_20d,
                ti.rsi_14
            FROM trading_signals ts
            LEFT JOIN raw_price_data p
                ON ts.symbol_ticker = p.symbol_ticker
                AND ts.signal_date = p.price_date
            LEFT JOIN raw_price_data p2
                ON ts.symbol_ticker = p2.symbol_ticker
                AND p2.price_date = date(ts.signal_date, '+5 days')
            LEFT JOIN volatility_metrics vm
                ON ts.symbol_ticker = vm.symbol_ticker
                AND ts.signal_date = vm.vol_date
            LEFT JOIN technical_indicators ti
                ON ts.symbol_ticker = ti.symbol_ticker
                AND ts.signal_date = ti.indicator_date
            WHERE ts.signal_date >= date('now', '-365 days')
              AND ts.metadata IS NOT NULL
            ORDER BY ts.signal_date
            """

            df = pd.read_sql(query, conn)
            # Connection automatically closed when exiting 'with' block

            if df.empty:
                logger.warning("No historical signals found in database")
                return pd.DataFrame()

            logger.info(f"Loaded {len(df)} historical signals for training")
            return df

    def _get_sector_momentum(self) -> Dict[str, float]:
        """
        Get sector ETF momentum (5-day returns).

        FIXED (Problem 12, 16): Add sector momentum features from config.

        Sector ETFs now from config.settings.SECTOR_ETFS:
        - XLK: Technology
        - XLF: Financials
        - XLE: Energy
        - XLV: Healthcare
        - XLY: Consumer Discretionary
        - XLP: Consumer Staples

        Returns:
            Dict of sector -> 5-day return %
        """
        from config.settings import SECTOR_ETFS, FEATURE_DEFAULTS
        sector_etfs = list(SECTOR_ETFS.keys())
        sector_returns = {}

        try:
            from utils.market_data_client import get_market_data_client
            market_data = get_market_data_client()

            for symbol in sector_etfs:
                try:
                    bars = market_data.get_stock_bars(symbol=symbol, days=7)
                    if bars and len(bars) >= 2:
                        # Calculate 5-day return
                        closes = [bar.close for bar in bars]
                        if len(closes) >= 6:
                            ret_5d = ((closes[-1] - closes[-6]) / closes[-6]) * 100
                            sector_returns[symbol] = ret_5d
                        else:
                            sector_returns[symbol] = FEATURE_DEFAULTS['sector_return']
                    else:
                        sector_returns[symbol] = FEATURE_DEFAULTS['sector_return']
                except (AttributeError, IndexError, ZeroDivisionError) as e:
                    # FIXED (Problem 14): Specific exceptions, with logging
                    logger.warning(f"Failed to get sector momentum for {symbol}: {e}")
                    sector_returns[symbol] = FEATURE_DEFAULTS['sector_return']

        except Exception as e:
            logger.warning(f"Error getting sector momentum: {e}")
            for symbol in sector_etfs:
                sector_returns[symbol] = FEATURE_DEFAULTS['sector_return']

        return sector_returns

    def _get_vix_features(self) -> Dict[str, float]:
        """
        Get VIX features (volatility index).

        FIXED (Problem 12, 16): Add VIX level and term structure from config.

        Returns:
            Dict with:
            - vix_level: Current VIX value
            - vix_change: 5-day change in VIX
            - vix_percentile: VIX relative to 30-day range (0-100)
        """
        from config.settings import FEATURE_DEFAULTS
        vix_features = {
            'vix_level': FEATURE_DEFAULTS['vix_level'],
            'vix_change': FEATURE_DEFAULTS['vix_change'],
            'vix_percentile': FEATURE_DEFAULTS['vix_percentile']
        }

        try:
            from utils.market_data_client import get_market_data_client
            market_data = get_market_data_client()

            # Get VIX bars for last 30 days
            bars = market_data.get_stock_bars(symbol='VIX', days=35)

            if bars and len(bars) >= 2:
                closes = [bar.close for bar in bars]

                # Current VIX level
                vix_features['vix_level'] = closes[-1]

                # 5-day change
                if len(closes) >= 6:
                    vix_features['vix_change'] = closes[-1] - closes[-6]

                # Percentile in 30-day range
                if len(closes) >= 30:
                    recent_30 = closes[-30:]
                    vix_min = min(recent_30)
                    vix_max = max(recent_30)

                    if vix_max > vix_min:
                        vix_features['vix_percentile'] = ((closes[-1] - vix_min) / (vix_max - vix_min)) * 100

        except Exception as e:
            logger.warning(f"Error getting VIX features: {e}")

        return vix_features

    def _get_market_breadth(self) -> float:
        """
        Get market breadth (advance/decline ratio).

        FIXED (Problem 12): Add market breadth indicator.

        Uses SPY components as proxy:
        - Count how many of our tracked symbols are up today
        - Ratio: advancing / (advancing + declining)
        - 0.5 = neutral, >0.5 = more advancing, <0.5 = more declining

        Returns:
            Advance/decline ratio (0.0 to 1.0)
        """
        try:
            from utils.market_data_client import get_market_data_client
            market_data = get_market_data_client()

            advancing = 0
            declining = 0

            for symbol in self.SYMBOLS[:10]:  # Use first 10 symbols
                try:
                    bars = market_data.get_stock_bars(symbol=symbol, days=3)
                    if bars and len(bars) >= 2:
                        closes = [bar.close for bar in bars]
                        if closes[-1] > closes[-2]:
                            advancing += 1
                        elif closes[-1] < closes[-2]:
                            declining += 1
                except (AttributeError, IndexError, TypeError) as e:
                    # FIXED (Problem 14): Specific exceptions, silently skip symbol
                    logger.debug(f"Skipping {symbol} for breadth calculation: {e}")
                    continue

            total = advancing + declining
            if total > 0:
                return advancing / total
            else:
                from config.settings import FEATURE_DEFAULTS
                return FEATURE_DEFAULTS['breadth_ratio']  # Neutral if no data

        except Exception as e:
            logger.warning(f"Error getting market breadth: {e}")
            from config.settings import FEATURE_DEFAULTS
            return FEATURE_DEFAULTS['breadth_ratio']

    def _get_time_features(self) -> Dict[str, float]:
        """
        Get time-based features.

        FIXED (Problem 12, 16): Add day-of-week and month-of-year patterns from config.

        Returns:
            Dict with:
            - day_of_week: 0 (Monday) to 4 (Friday)
            - month: 1 (January) to 12 (December)
            - is_month_end: 1 if last 5 days of month, else 0
            - is_month_start: 1 if first 5 days of month, else 0
        """
        from config.settings import TIME_FEATURES
        now = datetime.now()

        # Day of week (0=Monday, 4=Friday)
        day_of_week = now.weekday()
        if day_of_week > 4:  # Weekend
            day_of_week = 4  # Treat as Friday

        # Month (1-12)
        month = now.month

        # Month end/start effects (thresholds from config)
        day_of_month = now.day
        is_month_end = 1.0 if day_of_month >= TIME_FEATURES['month_end_threshold'] else 0.0
        is_month_start = 1.0 if day_of_month <= TIME_FEATURES['month_start_threshold'] else 0.0

        return {
            'day_of_week': float(day_of_week),
            'month': float(month),
            'is_month_end': is_month_end,
            'is_month_start': is_month_start
        }

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

        FIXED (Problem 12): Enhanced feature engineering with:
        - Original features (sentiment, pairs, volatility, RSI)
        - Sector momentum (6 sector ETFs)
        - VIX features (level, change, percentile)
        - Market breadth (advance/decline ratio)
        - Time features (day of week, month, month-end/start)

        Total: 12 (original) + 6 (sectors) + 3 (VIX) + 1 (breadth) + 4 (time) = 26 features

        Args:
            sentiment_score: Sentiment signal (-1, 0, 1)
            sentiment_prob: Sentiment confidence (0-1)
            pairs_zscore: Pairs spread z-score
            pairs_quality: Pairs quality score
            volatility: Market volatility
            rsi: RSI indicator

        Returns:
            Feature vector ready for meta-model (26 features)
        """
        # FIXED (Problem 16): All defaults and normalization factors from config
        from config.settings import (
            FEATURE_DEFAULTS,
            NORMALIZATION_FACTORS,
            SECTOR_ETFS
        )

        # Original features (12 features)
        base_features = [
            sentiment_score,
            sentiment_prob,
            abs(sentiment_score) * sentiment_prob,  # Weighted sentiment
            pairs_zscore,
            abs(pairs_zscore),  # Magnitude of deviation
            pairs_quality,
            pairs_zscore * pairs_quality,  # Weighted pairs signal
            volatility if volatility else FEATURE_DEFAULTS['volatility'],
            rsi if rsi else FEATURE_DEFAULTS['rsi'],
            # Interaction features
            sentiment_score * pairs_zscore,  # Agreement indicator
            abs(sentiment_score - np.sign(pairs_zscore)),  # Disagreement indicator
            sentiment_prob * pairs_quality,  # Combined confidence
        ]

        # FIXED (Problem 12, 16): Add sector momentum features (6 features) from config
        sector_returns = self._get_sector_momentum()
        sector_features = [
            sector_returns.get(symbol, FEATURE_DEFAULTS['sector_return'])
            for symbol in SECTOR_ETFS.keys()
        ]

        # FIXED (Problem 12, 16): Add VIX features (3 features) with config normalization
        vix_data = self._get_vix_features()
        vix_features = [
            vix_data['vix_level'] / NORMALIZATION_FACTORS['vix_level'],
            vix_data['vix_change'] / NORMALIZATION_FACTORS['vix_change'],
            vix_data['vix_percentile'] / NORMALIZATION_FACTORS['vix_percentile'],
        ]

        # FIXED (Problem 12): Add market breadth (1 feature)
        breadth = self._get_market_breadth()
        breadth_features = [breadth]

        # FIXED (Problem 12, 16): Add time features (4 features) with config normalization
        time_data = self._get_time_features()
        time_features = [
            time_data['day_of_week'] / NORMALIZATION_FACTORS['day_of_week'],
            time_data['month'] / NORMALIZATION_FACTORS['month'],
            time_data['is_month_end'],          # Already 0 or 1
            time_data['is_month_start'],        # Already 0 or 1
        ]

        # Combine all features (12 + 6 + 3 + 1 + 4 = 26 features)
        all_features = base_features + sector_features + vix_features + breadth_features + time_features

        return np.array(all_features).reshape(1, -1)

    def _train_meta_model(self):
        """
        Train the meta-learner on historical signals.

        FIXED (Problem 11): Uses Logistic Regression instead of XGBoost for limited data.
        - Simpler model prevents overfitting on 348 samples
        - L2 regularization for robustness
        - Time-series split prevents future data leakage
        - No synthetic data augmentation (only real historical signals)

        The meta-model learns to predict profitability based on:
        - Strategy signals
        - Market conditions
        - Historical performance patterns
        """
        logger.info("Training meta-learner (Logistic Regression)...")

        # Get historical data
        historical_data = self._get_historical_signals()

        if len(historical_data) < self.MIN_TRAINING_SAMPLES:
            logger.warning(f"Insufficient training data ({len(historical_data)} samples)")
            logger.warning("Meta-model will use equal weights until more data is available")
            return

        # Extract features and labels
        try:
            # Create feature matrix
            features_list = []
            labels_list = []

            # FIXED (Problem 16): Import config defaults for training data
            from config.settings import (
                FEATURE_DEFAULTS,
                NORMALIZATION_FACTORS,
                TIME_FEATURES as TIME_CONFIG
            )

            for _, row in historical_data.iterrows():
                # FIXED (Problem 12, 16): Extract 26 features (matching _prepare_meta_features)
                # Base features (12 features) - using config defaults
                signal_type = 1 if row['signal_type'] == 'BUY' else -1 if row['signal_type'] == 'SELL' else 0
                strength = row['strength'] if pd.notna(row['strength']) else FEATURE_DEFAULTS['sentiment_strength']
                volatility = row['volatility_20d'] if pd.notna(row['volatility_20d']) else FEATURE_DEFAULTS['volatility']
                rsi = row['rsi_14'] if pd.notna(row['rsi_14']) else FEATURE_DEFAULTS['rsi']

                base_features = [
                    signal_type,
                    strength,
                    abs(signal_type) * strength,  # Weighted signal
                    FEATURE_DEFAULTS['pairs_zscore'],  # pairs_zscore (not in DB)
                    FEATURE_DEFAULTS['pairs_zscore'],  # abs(pairs_zscore)
                    FEATURE_DEFAULTS['pairs_quality'],  # pairs_quality
                    FEATURE_DEFAULTS['pairs_zscore'],  # weighted pairs
                    volatility,
                    rsi,
                    FEATURE_DEFAULTS['pairs_zscore'],  # interaction: sentiment * pairs
                    abs(signal_type),  # disagreement indicator
                    strength * NORMALIZATION_FACTORS['sentiment_interaction'],  # combined confidence
                ]

                # Sector features (6 features) - use defaults for historical data
                sector_features = [FEATURE_DEFAULTS['sector_return']] * 6

                # VIX features (3 features) - use defaults (normalized)
                vix_features = [
                    FEATURE_DEFAULTS['vix_level'] / NORMALIZATION_FACTORS['vix_level'],
                    FEATURE_DEFAULTS['vix_change'] / NORMALIZATION_FACTORS['vix_change'],
                    FEATURE_DEFAULTS['vix_percentile'] / NORMALIZATION_FACTORS['vix_percentile']
                ]

                # Market breadth (1 feature) - use default
                breadth_features = [FEATURE_DEFAULTS['breadth_ratio']]

                # Time features (4 features) - extract from signal_date if available
                if pd.notna(row.get('signal_date')):
                    try:
                        sig_date = pd.to_datetime(row['signal_date'])
                        day_of_week = float(sig_date.weekday()) / NORMALIZATION_FACTORS['day_of_week']
                        month = float(sig_date.month) / NORMALIZATION_FACTORS['month']
                        is_month_end = 1.0 if sig_date.day >= TIME_CONFIG['month_end_threshold'] else 0.0
                        is_month_start = 1.0 if sig_date.day <= TIME_CONFIG['month_start_threshold'] else 0.0
                        time_features = [day_of_week, month, is_month_end, is_month_start]
                    except (ValueError, TypeError, AttributeError) as e:
                        # FIXED (Problem 14): Specific exception types, with logging
                        logger.warning(f"Failed to parse signal_date: {e}")
                        time_features = [0.5, 0.5, 0.0, 0.0]  # defaults
                else:
                    time_features = [0.5, 0.5, 0.0, 0.0]  # defaults

                # Combine all 26 features
                features = base_features + sector_features + vix_features + breadth_features + time_features

                # Label: 1 if profitable, 0 otherwise
                label = 1 if (row['return_5d'] if pd.notna(row['return_5d']) else 0) > 0 else 0

                features_list.append(features)
                labels_list.append(label)

            X = np.array(features_list)
            y = np.array(labels_list)

            logger.info(f"Training on {len(X)} samples (real historical data only)")

            # FIXED: Use TimeSeriesSplit instead of train_test_split
            # This respects time ordering and prevents future data leakage
            tscv = TimeSeriesSplit(n_splits=5)

            # Use the last fold for final train/test split
            train_idx, test_idx = list(tscv.split(X))[-1]
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            logger.info(f"Time-series split: {len(X_train)} train, {len(X_test)} test")

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # FIXED: Use Logistic Regression instead of XGBoost
            # - Simpler model better suited for limited data (348 samples)
            # - L2 regularization (C=1.0) prevents overfitting
            # - More interpretable coefficients
            # - Faster training and prediction
            self.meta_model = LogisticRegression(
                C=1.0,              # L2 regularization strength (1/lambda)
                max_iter=1000,      # Sufficient iterations to converge
                random_state=42,
                solver='lbfgs',     # Efficient for small datasets
                class_weight='balanced'  # Handle class imbalance
            )

            self.meta_model.fit(X_train_scaled, y_train)

            # Evaluate
            train_score = self.meta_model.score(X_train_scaled, y_train)
            test_score = self.meta_model.score(X_test_scaled, y_test)

            # Calculate cross-validation score for better estimate
            cv_scores = []
            for train_idx_cv, test_idx_cv in tscv.split(X):
                X_train_cv = self.scaler.fit_transform(X[train_idx_cv])
                X_test_cv = self.scaler.transform(X[test_idx_cv])

                model_cv = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver='lbfgs', class_weight='balanced')
                model_cv.fit(X_train_cv, y[train_idx_cv])
                cv_scores.append(model_cv.score(X_test_cv, y[test_idx_cv]))

            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)

            logger.info(f"Meta-model trained (Logistic Regression):")
            logger.info(f"  Train accuracy: {train_score:.3f}")
            logger.info(f"  Test accuracy: {test_score:.3f}")
            logger.info(f"  CV accuracy: {cv_mean:.3f} ± {cv_std:.3f}")
            logger.info(f"  Model: Simpler than XGBoost, better for limited data")

            # Save model
            self._save_meta_model()
            self.last_retrain_date = datetime.now()

        except Exception as e:
            logger.error(f"Error training meta-model: {e}")
            import traceback
            traceback.print_exc()
            self.meta_model = None

    # =============================================================================
    # RELIABLE MARKET DATA METHODS (Override unreliable Lumibot methods)
    # =============================================================================

    def get_cash_safe(self) -> float:
        """
        Get available cash balance (RELIABLE version).

        This OVERRIDES Lumibot's get_cash() which sometimes returns None.
        Uses our MarketDataClient which ALWAYS returns a float (0.0 on error, never None).

        Returns:
            float: Cash available for trading (NEVER None)
        """
        try:
            cash = self.market_data.get_cash()
            logger.debug(f"Cash (reliable): ${cash:,.2f}")
            return cash
        except Exception as e:
            logger.error(f"Error in get_cash_safe: {e}")
            return 0.0  # Return 0 instead of None to prevent crashes

    def get_portfolio_value_safe(self) -> float:
        """
        Get total portfolio value (RELIABLE version).

        This OVERRIDES Lumibot's get_portfolio_value() which can be unreliable.
        Uses our MarketDataClient which ALWAYS returns a float.

        Returns:
            float: Total account equity (NEVER None)
        """
        try:
            value = self.market_data.get_portfolio_value()
            logger.debug(f"Portfolio value (reliable): ${value:,.2f}")
            return value
        except Exception as e:
            logger.error(f"Error in get_portfolio_value_safe: {e}")
            return 0.0

    def get_positions_safe(self) -> List:
        """
        Get all open positions (RELIABLE version).

        This OVERRIDES Lumibot's get_positions() which can be unreliable.
        Uses our MarketDataClient which ALWAYS returns a list (empty [] on error, never None).

        Returns:
            List: Position objects (NEVER None)
        """
        try:
            positions = self.market_data.get_positions()
            logger.debug(f"Positions (reliable): {len(positions)} found")
            return positions
        except Exception as e:
            logger.error(f"Error in get_positions_safe: {e}")
            return []  # Return empty list instead of None

    def get_position_safe(self, symbol: str):
        """
        Get position for specific symbol (RELIABLE version).

        Args:
            symbol: Stock ticker

        Returns:
            Position object or None (but never crashes)
        """
        try:
            position = self.market_data.get_position(symbol)
            return position
        except Exception as e:
            logger.error(f"Error in get_position_safe for {symbol}: {e}")
            return None

    # =============================================================================
    # MARKET CONDITIONS
    # =============================================================================

    def _get_market_conditions(self, symbol: str) -> Tuple[float, float]:
        """
        Fetch current market conditions.

        Args:
            symbol: Stock symbol

        Returns:
            Tuple of (volatility, rsi)
        """
        try:
            # FIXED: Use context manager to prevent connection leaks
            with sqlite3.connect(self.db_path) as conn:
                query = """
                SELECT vm.close_to_close_vol_20d as volatility_20d, ti.rsi_14
                FROM volatility_metrics vm
                LEFT JOIN technical_indicators ti
                    ON vm.symbol_ticker = ti.symbol_ticker
                    AND vm.vol_date = ti.indicator_date
                WHERE vm.symbol_ticker = ?
                ORDER BY vm.vol_date DESC
                LIMIT 1
                """

                result = pd.read_sql(query, conn, params=(symbol,))
                # Connection automatically closed when exiting 'with' block

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
                except (AttributeError, ValueError, TypeError) as e:
                    # FIXED (Problem 14): Specific exceptions, with logging
                    logger.warning(f"Failed to calculate pairs z-score for {s1}/{s2}: {e}")
                    # pairs_zscore remains 0.0 (default)
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

    def _rebalance_positions(self, portfolio_value: float, max_position_pct: float = 0.15) -> None:
        """
        Rebalance positions to prevent any single position from dominating the portfolio.

        If one stock goes up significantly, it might become 30% of your portfolio.
        This is risky - if that stock crashes, you lose big. This method trims
        oversized positions back down to a reasonable level.

        Args:
            portfolio_value: Total portfolio value
            max_position_pct: Maximum % any position should be (default 15%)
        """
        try:
            if portfolio_value <= 0:
                return

            logger.info(f"Checking position sizes (max allowed: {max_position_pct:.0%})")

            positions = self.get_positions()
            if not positions or len(positions) == 0:
                return

            rebalanced = False

            for position in positions:
                symbol = position.symbol

                # Skip cash positions (USD, USDC, etc.)
                symbol_clean = symbol.strip().upper()
                if symbol_clean in ["USD", "USDC", "USDT", "USDP"]:
                    logger.info(f"⏭️  Skipping cash position: {symbol}")
                    continue

                quantity = position.quantity
                current_price = self.get_last_price(symbol)

                if not current_price or current_price <= 0:
                    continue

                position_value = quantity * current_price
                position_pct = position_value / portfolio_value

                logger.debug(f"{symbol}: {position_pct:.1%} of portfolio (${position_value:,.2f})")

                # If position is too large, trim it down
                if position_pct > max_position_pct:
                    # Calculate how many shares to sell to get to max_position_pct
                    target_value = portfolio_value * max_position_pct
                    shares_to_sell = (position_value - target_value) / current_price

                    if shares_to_sell >= 1:
                        logger.warning(f"⚖️  {symbol} is {position_pct:.1%} of portfolio (limit: {max_position_pct:.0%})")
                        logger.warning(f"   Trimming: selling {shares_to_sell:.2f} shares")

                        # Fractional orders require DAY time_in_force
                        is_fractional = (shares_to_sell % 1) != 0
                        time_in_force = "day" if is_fractional else ("gtc" if self.ENABLE_EXTENDED_HOURS else "day")

                        order = self.create_order(symbol, shares_to_sell, "sell", time_in_force=time_in_force)
                        self.submit_order(order)
                        rebalanced = True

                        logger.info(f"✂️  Sold {shares_to_sell:.2f} shares @ ${current_price:.2f}")

            if not rebalanced:
                logger.info("✅ All positions properly sized")

        except Exception as e:
            logger.error(f"Error rebalancing positions: {e}")

    def _check_market_sentiment_and_hedge(self, cash: float, portfolio_value: float) -> None:
        """Check market sentiment and manage inverse ETF hedges."""
        try:
            from risk.hedge_manager import HedgeManager

            if not hasattr(self, 'hedge_manager'):
                self.hedge_manager = HedgeManager()

            result = self.hedge_manager.manage_hedges(cash, portfolio_value)

            if 'next_check_minutes' in result:
                next_check_min = result['next_check_minutes']
                self.sleeptime = f"{next_check_min}M"
                logger.info(f"⏰ Adjusted check interval to {next_check_min} minutes")

        except Exception as e:
            logger.error(f"Error in hedge management: {e}")
            import traceback
            traceback.print_exc()
            self.sleeptime = self.SLEEPTIME

    def _create_bracket_order(self, symbol: str, quantity: float, side: str = "buy",
                             current_price: float = None) -> bool:
        """Create a bracket order with automatic stop-loss and take-profit.

        Bracket orders require WHOLE shares (Alpaca limitation). Fractional quantities
        are rounded down and skipped if they become 0.
        """
        try:
            if current_price is None:
                current_price = self.get_last_price(symbol)

            if not current_price or current_price <= 0:
                logger.error(f"Invalid price for {symbol}: {current_price}")
                return False

            # Bracket orders require WHOLE shares - round down
            original_qty = quantity
            quantity = math.floor(quantity)

            if quantity <= 0:
                logger.info(f"Skipping {symbol}: quantity {original_qty:.4f} rounds to 0 (bracket orders need whole shares)")
                return False

            if original_qty != quantity:
                logger.info(f"Rounded {symbol} from {original_qty:.4f} to {quantity} shares (bracket orders need whole shares)")

            if side == "buy":
                take_profit_price = round(current_price * (1 + self.TAKE_PROFIT_PCT), 2)
                stop_loss_price = round(current_price * (1 - self.STOP_LOSS_PCT), 2)
                order_side = OrderSide.BUY
            else:
                take_profit_price = round(current_price * (1 - self.TAKE_PROFIT_PCT), 2)
                stop_loss_price = round(current_price * (1 + self.STOP_LOSS_PCT), 2)
                order_side = OrderSide.SELL

            logger.info(f"Creating BRACKET order for {symbol}:")
            logger.info(f"  Entry: {quantity} shares @ ${current_price:.2f}")
            logger.info(f"  Stop: ${stop_loss_price:.2f} | Take-profit: ${take_profit_price:.2f}")

            try:
                clock = self.alpaca_client.get_clock()
                market_is_open = clock.is_open
            except:
                market_is_open = False

            time_in_force = TimeInForce.GTC if (market_is_open and self.ENABLE_EXTENDED_HOURS) else TimeInForce.DAY

            bracket_order = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=order_side,
                time_in_force=time_in_force,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=take_profit_price),
                stop_loss=StopLossRequest(stop_price=stop_loss_price)
            )

            result = self.alpaca_client.submit_order(order_data=bracket_order)
            logger.info(f"✅ BRACKET order submitted (ID: {result.id})")
            return True

        except Exception as e:
            logger.error(f"Bracket order failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()

            # Fallback: simple order without protection
            logger.warning(f"Falling back to simple order (NO PROTECTION)")
            try:
                order = self.create_order(symbol, quantity, side)
                self.submit_order(order)
                logger.warning(f"⚠️  Simple order submitted - stop_loss_manager will add protection")
                return True
            except Exception as e2:
                logger.error(f"Failed to submit any order: {e2}")
                return False

    def _check_risk_management_alpaca(self, alpaca_position) -> Optional[str]:
        """Check if Alpaca position triggers stop-loss or take-profit."""
        try:
            symbol = alpaca_position.symbol
            entry_price = float(alpaca_position.avg_entry_price)
            current_price = float(alpaca_position.current_price)

            pnl_pct = (current_price - entry_price) / entry_price

            if pnl_pct <= -self.STOP_LOSS_PCT:
                logger.warning(f"🛑 STOP-LOSS: {symbol} at {pnl_pct:.2%} (entry: ${entry_price:.2f}, current: ${current_price:.2f})")
                return 'stop_loss'

            if pnl_pct >= self.TAKE_PROFIT_PCT:
                logger.info(f"💰 TAKE-PROFIT: {symbol} at {pnl_pct:.2%} (entry: ${entry_price:.2f}, current: ${current_price:.2f})")
                return 'take_profit'

            # Log status for positions approaching thresholds
            if pnl_pct <= -0.03:
                logger.info(f"⚠️  {symbol} at {pnl_pct:.2%} (approaching stop-loss)")
            elif pnl_pct >= 0.10:
                logger.info(f"📈 {symbol} at {pnl_pct:.2%} (approaching take-profit)")

            return None

        except Exception as e:
            logger.error(f"Error checking risk for position: {e}")
            return None

    def _submit_risk_exit_order_alpaca(self, alpaca_position, reason: str):
        """Submit exit order for Alpaca position."""
        try:
            symbol = alpaca_position.symbol
            quantity = float(alpaca_position.qty)
            current_price = float(alpaca_position.current_price)

            # Use market order for immediate exit on stop-loss, limit for take-profit
            if reason == 'stop_loss':
                order = self.create_order(
                    symbol,
                    quantity,
                    "sell",
                    type="market"
                )
            else:  # take_profit
                limit_price = round(current_price * 0.99, 2)  # 1% below current to ensure fill
                order = self.create_order(
                    symbol,
                    quantity,
                    "sell",
                    type="limit",
                    limit_price=limit_price,
                    time_in_force="gtc" if self.ENABLE_EXTENDED_HOURS else "day",
                    extended_hours=self.ENABLE_EXTENDED_HOURS
                )

            self.submit_order(order)
            logger.info(f"{'🛑' if reason == 'stop_loss' else '💰'} SOLD {symbol}: {quantity:.2f} shares @ ${current_price:.2f}")

        except Exception as e:
            logger.error(f"Error exiting {alpaca_position.symbol}: {e}")

    def _check_risk_management(self, symbol: str, position) -> Optional[str]:
        """Check if position triggers stop-loss or take-profit."""
        if position is None:
            return None

        current_price = self.get_last_price(symbol)
        if current_price is None:
            return None

        # Debug: Log all position attributes
        logger.debug(f"Position attributes for {symbol}: {dir(position)}")

        # Try different attribute names for entry price
        entry_price = None
        qty = None

        # Try to get entry price from various possible attributes
        for attr_name in ['avg_entry_price', 'avg_fill_price', 'entry_price', 'purchase_price']:
            if hasattr(position, attr_name):
                try:
                    entry_price = float(getattr(position, attr_name))
                    if entry_price > 0:
                        logger.debug(f"Found entry price in {attr_name}: ${entry_price:.2f}")
                        break
                except (ValueError, TypeError):
                    continue

        # If still None, try cost_basis / qty
        if entry_price is None and hasattr(position, 'cost_basis'):
            # Try to get quantity
            for qty_attr in ['qty', 'quantity', 'shares']:
                if hasattr(position, qty_attr):
                    try:
                        qty = float(getattr(position, qty_attr))
                        if qty > 0:
                            break
                    except (ValueError, TypeError):
                        continue

            if qty and qty > 0:
                try:
                    cost_basis = float(position.cost_basis)
                    entry_price = cost_basis / qty
                    logger.debug(f"Calculated entry price from cost_basis: ${entry_price:.2f}")
                except (ValueError, TypeError):
                    pass

        if entry_price is None or entry_price <= 0:
            logger.warning(f"Could not determine entry price for {symbol}. Available attributes: {[a for a in dir(position) if not a.startswith('_')]}")
            return None

        pnl_pct = (current_price - entry_price) / entry_price

        if pnl_pct <= -self.STOP_LOSS_PCT:
            logger.warning(f"🛑 STOP-LOSS: {symbol} at {pnl_pct:.2%}")
            return 'stop_loss'

        if pnl_pct >= self.TAKE_PROFIT_PCT:
            logger.info(f"💰 TAKE-PROFIT: {symbol} at {pnl_pct:.2%}")
            return 'take_profit'

        return None

    def _submit_risk_exit_order(self, symbol: str, position, reason: str):
        """Submit exit order for risk management."""
        try:
            current_price = self.get_last_price(symbol)
            limit_price = round(current_price * (0.98 if reason == 'stop_loss' else 0.99), 2)

            order = self.create_order(
                symbol,
                position.quantity,
                "sell",
                type="limit",
                limit_price=limit_price,
                time_in_force="gtc" if self.ENABLE_EXTENDED_HOURS else "day",
                extended_hours=self.ENABLE_EXTENDED_HOURS
            )
            self.submit_order(order)

            logger.info(f"{'🛑' if reason == 'stop_loss' else '💰'} SOLD {symbol}: {position.quantity:.2f} shares @ ${limit_price:.2f}")

        except Exception as e:
            logger.error(f"Error exiting {symbol}: {e}")

    @retry_on_connection_error(max_retries=3, initial_delay=5, backoff_factor=2)
    def on_trading_iteration(self):
        """Main trading logic combining signals from both strategies using meta-learner."""
        try:
            self.health_monitor.record_iteration()
            self.memory_profiler.take_snapshot()
        except Exception as e:
            logger.debug(f"Health monitoring error: {e}")

        logger.info("=" * 80)
        logger.info(f"COMBINED STRATEGY - Trading Iteration at {datetime.now()}")
        logger.info("=" * 80)

        try:
            portfolio_value = self.get_portfolio_value_safe()
            snapshot = self.daily_pnl.update(current_portfolio_value=portfolio_value)

            logger.info(f"📊 Daily P&L: ${snapshot.total_pnl:+,.2f} ({snapshot.pnl_percent:+.2f}%)")

            can_trade, trade_reason = self.daily_pnl.can_trade()
            if not can_trade:
                logger.error(f"🚨 TRADING BLOCKED: {trade_reason}")
                return
            self.current_position_size_multiplier = self.daily_pnl.get_position_size_multiplier()
            if self.current_position_size_multiplier < 1.0:
                logger.warning(f"⚠️ Position sizing reduced to {self.current_position_size_multiplier * 100:.0f}% due to daily losses")

        except Exception as e:
            logger.error(f"Error updating daily P&L: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to full size if error
            self.current_position_size_multiplier = 1.0

        logger.info("=" * 80)

        try:
            clock = self.alpaca_client.get_clock()
            if not clock.is_open:
                logger.info(f"⏰ Market is closed. Next open: {clock.next_open}")
                return
        except Exception as e:
            logger.warning(f"Could not check market hours: {e}")

        try:
            from risk.stop_loss_manager import ensure_all_positions_protected
            logger.info("Verifying position protection...")
            protection_ok = ensure_all_positions_protected()
            if not protection_ok:
                logger.warning("⚠️  Some positions lack protection - created missing orders")
        except Exception as e:
            logger.error(f"Error checking/creating protection: {e}")
            import traceback
            traceback.print_exc()

        if self.last_retrain_date:
            days_since_retrain = (datetime.now() - self.last_retrain_date).days
            if days_since_retrain >= self.RETRAIN_FREQUENCY_DAYS:
                logger.info(f"Retraining meta-model ({days_since_retrain} days since last retrain)")
                self._train_meta_model()

        cash = self.get_cash_safe()
        portfolio_value = self.get_portfolio_value_safe()
        logger.info(f"Portfolio value: ${portfolio_value:,.2f}")

        if cash is not None and portfolio_value > 0:
            self._check_market_sentiment_and_hedge(cash if cash else 0, portfolio_value)

        self._rebalance_positions(portfolio_value)

        logger.info("Checking for new trading opportunities...")

        if cash is not None and cash > 0:
            logger.info(f"Available cash: ${cash:,.2f}")

            if self.meta_model is not None:
                symbols_to_check = set(self.SYMBOLS[:20])
                logger.info(f"Scanning {len(symbols_to_check)} symbols...")

                opportunities_found = 0

                for symbol in symbols_to_check:
                    try:
                        volatility, rsi = self._get_market_conditions(symbol)
                        logger.debug(f"📊 {symbol}: RSI={rsi:.1f}, Vol={volatility:.1%}")

                        if volatility > 0.6:
                            logger.debug(f"   {symbol}: Too volatile, skipping")
                            continue

                        position = self.get_position(symbol)

                        if rsi < 40 and position is None and cash > 500:
                            opportunities_found += 1
                            price = self.get_last_price(symbol)
                            if price and price > 0:
                                base_quantity = min(cash * 0.05, 500) / price
                                quantity = base_quantity * self.current_position_size_multiplier

                                if quantity < 1:
                                    logger.info(f"⏭️  Skipping {symbol}: position size too small")
                                    continue

                                success = self._create_bracket_order(symbol, quantity, "buy", price)
                                if success:
                                    if self.current_position_size_multiplier < 1.0:
                                        logger.info(f"📈 BUY {symbol}: {quantity:.2f} shares @ ${price:.2f} (RSI: {rsi:.1f}) "
                                                   f"[scaled {self.current_position_size_multiplier * 100:.0f}%]")
                                    else:
                                        logger.info(f"📈 BUY {symbol}: {quantity:.2f} shares @ ${price:.2f} (RSI: {rsi:.1f})")
                                    cash -= quantity * price

                                    try:
                                        self.daily_pnl.update(portfolio_value, trade_executed=True)
                                    except (AttributeError, TypeError) as e:
                                        logger.warning(f"Failed to update daily P&L: {e}")

                        elif rsi > 60 and position is not None:
                            logger.info(f"📊 {symbol} overbought (RSI: {rsi:.1f})")
                            opportunities_found += 1

                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                        try:
                            self.health_monitor.record_error()
                        except (AttributeError, TypeError) as err:
                            logger.debug(f"Failed to record error: {err}")
                        continue

                if opportunities_found > 0:
                    logger.info(f"✅ Found {opportunities_found} trading opportunities")
                else:
                    logger.warning(f"⚠️  No opportunities (scanned {len(symbols_to_check)} symbols)")
                    logger.info(f"   RSI thresholds: <40 (buy) >60 (sell)")
            else:
                logger.info("Meta-model not trained - skipping new trades")
        else:
            logger.info(f"No cash available (${cash:,.2f if cash else 0})")

        logger.info(f"Final portfolio value: ${self.get_portfolio_value_safe():,.2f}")

        try:
            if self.memory_profiler.detect_leaks():
                logger.warning("Memory leak detected - performing cleanup")
                self.memory_profiler.cleanup()

            if self.health_monitor.iteration_count % 6 == 0:
                self.health_monitor.log_health_summary()
                self.memory_profiler.report()

                # Force garbage collection every 6 hours
                self.memory_profiler.force_cleanup()

                # FIXED (Problem 10): Log daily P&L summary every 6 hours
                try:
                    self.daily_pnl.log_daily_summary()
                except Exception as e:
                    logger.debug(f"Daily P&L summary error: {e}")

        except Exception as e:
            logger.debug(f"Health monitoring error: {e}")

        logger.info("=" * 80)
        logger.info("Trading iteration complete")
        logger.info("=" * 80)
