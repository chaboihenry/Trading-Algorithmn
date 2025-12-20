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
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from functools import wraps
from http.client import RemoteDisconnected
import requests.exceptions
from lumibot.strategies import Strategy

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

        # CRITICAL FIX: Lazy loading for FinBERT sentiment model
        # Loading FinBERT can take 30-60 seconds and blocks Ctrl+C
        # Model will be loaded on first use instead of during initialization
        self.tokenizer = None
        self.sentiment_model = None
        self.torch = None
        self._sentiment_model_loaded = False
        logger.info("⚡ Lazy loading enabled for FinBERT sentiment model")

        self.LOOKBACK_DAYS = PAIRS_LOOKBACK_DAYS
        self.ZSCORE_ENTRY = PAIRS_ZSCORE_ENTRY
        self.MIN_CORRELATION = PAIRS_MIN_CORRELATION
        self.cointegrated_pairs = []

        self.meta_model = None  # Will be Keras MLP neural network
        self.last_retrain_date = None

        # Use root-level models directory (not core/models)
        self.models_dir = Path(__file__).parent.parent / 'models'
        self.models_dir.mkdir(exist_ok=True)

        # CRITICAL FIX: Lazy loading to avoid TensorFlow blocking during initialization
        # TensorFlow import can take 2-5 minutes and blocks Ctrl+C signals
        # Model will be loaded/trained on first use instead of during initialization
        self._model_ready = False
        self._model_load_attempted = False
        logger.info("⚡ Lazy loading enabled - model will load on first use")
        logger.info("   This prevents TensorFlow from blocking startup (avoids 2-5 min delay)")

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
        """Load a saved Keras MLP meta-model from disk."""
        try:
            from models.keras_mlp import load_latest_model

            # Determine number of features (will be set after first training)
            n_features = 40  # Default estimate (will be updated during training)

            self.meta_model = load_latest_model(self.models_dir, n_features=n_features)

            if self.meta_model:
                logger.info(f"Loaded Keras MLP meta-model from {self.models_dir}")
            else:
                logger.info("No existing Keras MLP model found")
        except Exception as e:
            logger.warning(f"Could not load Keras MLP model: {e}")
            self.meta_model = None

    def _load_latest_meta_model(self):
        """Load the most recent Keras MLP meta-model from models directory."""
        try:
            model_files = list(self.models_dir.glob("keras_mlp_*.keras"))
            if not model_files:
                logger.info("No existing Keras MLP model found")
                return

            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Loading latest Keras MLP model: {latest_model.name}")

            from models.keras_mlp import KerasMLP

            # Load the model
            try:
                import tensorflow as tf
                from tensorflow import keras

                self.meta_model = keras.models.load_model(latest_model)
                logger.info(f"✅ Loaded Keras MLP from {latest_model}")
            except Exception as e:
                logger.error(f"Failed to load Keras model: {e}")
                self.meta_model = None

        except Exception as e:
            logger.warning(f"Error loading latest Keras MLP model: {e}")

    def _save_meta_model(self):
        """Save Keras MLP meta-model to disk."""
        if self.meta_model is None:
            logger.warning("No model to save")
            return

        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = self.models_dir / f"keras_mlp_{timestamp}.keras"

            self.meta_model.save(model_path)
            logger.info(f"✅ Saved Keras MLP to {model_path}")

        except Exception as e:
            logger.error(f"Failed to save Keras model: {e}")

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

    def _ensure_model_ready(self) -> bool:
        """
        Ensure Keras MLP model is loaded and ready (lazy loading).

        CRITICAL FIX: This method defers TensorFlow import until first use.
        - Called before any prediction that needs the model
        - Only runs once (tracked by _model_load_attempted flag)
        - Loads existing model or trains new one if needed
        - Returns True if model is ready, False if loading/training failed

        Returns:
            bool: True if model is ready, False otherwise
        """
        # Already attempted to load/train model
        if self._model_load_attempted:
            return self._model_ready

        # Mark that we've attempted to load the model
        self._model_load_attempted = True

        logger.info("=" * 80)
        logger.info("LAZY LOADING: Initializing Keras MLP model on first use")
        logger.info("=" * 80)
        logger.info("⚡ TensorFlow will now initialize (this may take 1-2 minutes on first run)")

        try:
            # Try to load existing model first
            self._load_latest_meta_model()

            # If no model loaded and retrain is requested, train new model
            if self.meta_model is None or self.retrain:
                logger.info("No existing model found - training new Keras MLP...")
                self._train_meta_model()

            # Check if model is ready
            if self.meta_model is not None:
                self._model_ready = True
                logger.info("✅ Keras MLP model ready for predictions")
                return True
            else:
                logger.warning("⚠️  Model loading/training failed - predictions will be disabled")
                self._model_ready = False
                return False

        except Exception as e:
            logger.error(f"Error in lazy model loading: {e}")
            import traceback
            traceback.print_exc()
            self._model_ready = False
            return False

    def _train_meta_model(self):
        """
        Train the Keras MLP meta-learner on comprehensive features from database.

        REPLACED LOGISTIC REGRESSION WITH KERAS MLP:
        - Neural network (2x Dense(16, relu) + Dropout(0.3))
        - Proven hyperparameters (R²=0.97): lr=0.21, epochs=1000
        - Comprehensive features from database (40+ features):
          * Technical indicators: RSI, MACD, Bollinger Bands, ADX, SMAs, EMAs
          * Volatility metrics: Parkinson, Garman-Klass, Yang-Zhang, etc.
          * Momentum: Returns over multiple timeframes
          * Volume: OBV, volume ratios
          * Sentiment: Average scores from news
        - Output: BUY (0), HOLD (1), SELL (2) classification
        """
        logger.info("=" * 80)
        logger.info("TRAINING KERAS MLP META-LEARNER")
        logger.info("=" * 80)

        try:
            # Import Keras MLP and feature engineering
            from models.keras_mlp import KerasMLP
            from models.feature_engineering import FeatureEngineer

            # Initialize feature engineer
            feature_engineer = FeatureEngineer(self.db_path)

            # Get list of symbols to train on
            symbols = self.SYMBOLS[:50]  # Train on top 50 symbols

            logger.info(f"Extracting features for {len(symbols)} symbols...")

            # Extract features for all symbols
            features_df = feature_engineer.extract_features_bulk(symbols)

            if features_df.empty or len(features_df) < 20:
                logger.warning(f"Insufficient feature data ({len(features_df)} symbols)")
                logger.warning("Need at least 20 symbols with complete features")
                return

            # Get historical signals from database for labels
            historical_data = self._get_historical_signals()

            if len(historical_data) < self.MIN_TRAINING_SAMPLES:
                logger.warning(f"Insufficient training samples ({len(historical_data)} < {self.MIN_TRAINING_SAMPLES})")
                return

            # Merge features with historical signals
            # Create labels: BUY=0, HOLD=1, SELL=2
            labels = []
            feature_rows = []

            for _, row in historical_data.iterrows():
                symbol = row['symbol_ticker']

                # Get features for this symbol
                if symbol not in features_df.index:
                    continue

                feature_rows.append(features_df.loc[symbol].values)

                # Create label based on signal type and profitability
                signal_type = row['signal_type']
                return_5d = row['return_5d'] if pd.notna(row['return_5d']) else 0

                if signal_type == 'BUY' and return_5d > 0.02:  # Profitable buy
                    labels.append(0)  # BUY
                elif signal_type == 'SELL' and return_5d < -0.02:  # Profitable sell
                    labels.append(2)  # SELL
                else:
                    labels.append(1)  # HOLD (unprofitable or neutral)

            if len(feature_rows) < self.MIN_TRAINING_SAMPLES:
                logger.warning(f"Insufficient matched samples ({len(feature_rows)} < {self.MIN_TRAINING_SAMPLES})")
                return

            # Convert to numpy arrays
            X = np.array(feature_rows)
            y = np.array(labels)

            logger.info(f"Training dataset:")
            logger.info(f"  Samples: {len(X)}")
            logger.info(f"  Features: {X.shape[1]}")
            logger.info(f"  BUY signals: {np.sum(y == 0)}")
            logger.info(f"  HOLD signals: {np.sum(y == 1)}")
            logger.info(f"  SELL signals: {np.sum(y == 2)}")

            # Initialize Keras MLP
            n_features = X.shape[1]
            mlp = KerasMLP(n_features=n_features, model_dir=self.models_dir)

            # Train model
            history = mlp.train(
                X,
                y,
                validation_split=0.2,
                verbose=1  # Show training progress
            )

            # Evaluate model
            test_split_idx = int(len(X) * 0.8)
            X_test = X[test_split_idx:]
            y_test = y[test_split_idx:]

            eval_results = mlp.evaluate(X_test, y_test)

            logger.info("=" * 80)
            logger.info("KERAS MLP TRAINING COMPLETE")
            logger.info(f"  Test accuracy: {eval_results['accuracy']:.4f}")
            logger.info(f"  Test loss: {eval_results['loss']:.4f}")
            logger.info("=" * 80)

            # Save model
            self.meta_model = mlp.model  # Store the Keras model
            self._save_meta_model()
            self.last_retrain_date = datetime.now()

            logger.info("✅ Keras MLP meta-learner ready for trading!")

        except Exception as e:
            logger.error(f"Error training Keras MLP meta-model: {e}")
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
        Fetch current market conditions with real-time RSI fallback.

        FIXED: Now checks data staleness and calculates real-time RSI when DB is stale (>1 day old).
        This prevents trading on outdated data.

        Args:
            symbol: Stock symbol

        Returns:
            Tuple of (volatility, rsi)
        """
        try:
            # FIXED: Use context manager to prevent connection leaks
            with sqlite3.connect(self.db_path) as conn:
                query = """
                SELECT vm.close_to_close_vol_20d as volatility_20d, ti.rsi_14, ti.indicator_date
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

                if not result.empty and result['rsi_14'].iloc[0] is not None:
                    volatility = result['volatility_20d'].iloc[0] if result['volatility_20d'].iloc[0] is not None else 0.2
                    rsi = result['rsi_14'].iloc[0]
                    indicator_date_str = result['indicator_date'].iloc[0]

                    # FIXED: Check data freshness
                    try:
                        indicator_date = datetime.strptime(indicator_date_str, '%Y-%m-%d')
                        now = datetime.now()
                        data_age_hours = (now - indicator_date).total_seconds() / 3600

                        # If data is more than 24 hours old, calculate real-time RSI
                        if data_age_hours > 24:
                            logger.warning(f"⚠️ {symbol} RSI data is {data_age_hours/24:.1f} days old - calculating real-time")

                            # Use hedge_manager's real-time RSI calculation
                            from risk.hedge_manager import HedgeManager
                            hedge_mgr = HedgeManager()
                            fresh_rsi = hedge_mgr._calculate_rsi_realtime(symbol)

                            if fresh_rsi is not None:
                                rsi = fresh_rsi
                                logger.info(f"✅ Using fresh RSI for {symbol}: {rsi:.1f}")
                            else:
                                logger.warning(f"Failed to calculate real-time RSI for {symbol}, using stale data")

                        return volatility, rsi

                    except ValueError:
                        # Date parsing failed, return what we have
                        return volatility, rsi
                else:
                    # No data in DB - try real-time calculation
                    logger.warning(f"No RSI data for {symbol} - calculating real-time")
                    try:
                        from risk.hedge_manager import HedgeManager
                        hedge_mgr = HedgeManager()
                        fresh_rsi = hedge_mgr._calculate_rsi_realtime(symbol)
                        if fresh_rsi is not None:
                            return 0.2, fresh_rsi
                    except Exception as e:
                        logger.error(f"Failed to calculate real-time RSI: {e}")

                    return 0.2, 50  # Defaults

        except Exception as e:
            logger.warning(f"Error fetching market conditions: {e}")
            return 0.2, 50

    def _get_enhanced_technicals(self, symbol: str) -> Dict[str, float]:
        """
        Fetch enhanced technical indicators for better entry/exit signals.

        Returns:
            Dict with keys: macd, macd_signal, bb_upper, bb_lower, bb_middle, adx_14, current_price
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                SELECT macd, macd_signal, macd_histogram,
                       bb_upper, bb_lower, bb_middle, adx_14,
                       sma_20, sma_50
                FROM technical_indicators
                WHERE symbol_ticker = ?
                ORDER BY indicator_date DESC
                LIMIT 1
                """
                result = pd.read_sql(query, conn, params=(symbol,))

                if not result.empty:
                    return {
                        'macd': result['macd'].iloc[0] if result['macd'].iloc[0] is not None else 0.0,
                        'macd_signal': result['macd_signal'].iloc[0] if result['macd_signal'].iloc[0] is not None else 0.0,
                        'macd_histogram': result['macd_histogram'].iloc[0] if result['macd_histogram'].iloc[0] is not None else 0.0,
                        'bb_upper': result['bb_upper'].iloc[0] if result['bb_upper'].iloc[0] is not None else None,
                        'bb_lower': result['bb_lower'].iloc[0] if result['bb_lower'].iloc[0] is not None else None,
                        'bb_middle': result['bb_middle'].iloc[0] if result['bb_middle'].iloc[0] is not None else None,
                        'adx_14': result['adx_14'].iloc[0] if result['adx_14'].iloc[0] is not None else 0.0,
                        'sma_20': result['sma_20'].iloc[0] if result['sma_20'].iloc[0] is not None else None,
                        'sma_50': result['sma_50'].iloc[0] if result['sma_50'].iloc[0] is not None else None
                    }
                else:
                    return {}

        except Exception as e:
            logger.debug(f"Error fetching enhanced technicals for {symbol}: {e}")
            return {}

    def _calculate_volatility_multiplier(self, volatility: float) -> float:
        """Calculate position size multiplier based on volatility.

        High volatility = smaller positions to manage risk
        Low volatility = larger positions to maximize returns

        Args:
            volatility: 20-day volatility (0.0 to 1.0+)

        Returns:
            Position size multiplier (0.3 to 1.3)
        """
        if volatility < 0.15:
            # Very low volatility: increase position size
            return 1.3
        elif volatility < 0.25:
            # Normal/low volatility: full position size
            return 1.0
        elif volatility < 0.35:
            # Moderate volatility: reduce position slightly
            return 0.8
        elif volatility < 0.50:
            # High volatility: reduce position significantly
            return 0.6
        elif volatility < 0.65:
            # Very high volatility: minimal position
            return 0.4
        else:
            # Extreme volatility: tiny position
            return 0.3

    def _get_combined_signal(self, symbol: str) -> Tuple[int, float]:
        """
        Get combined signal from meta-learner.

        Args:
            symbol: Stock symbol to analyze

        Returns:
            Tuple of (signal, confidence) where signal is -1/0/1 and confidence is 0-1
        """
        # CRITICAL FIX: Ensure model is loaded before use (lazy loading)
        # This is where TensorFlow will be imported on first prediction
        self._ensure_model_ready()

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

        # Use Keras MLP for intelligent combination
        try:
            from models.feature_engineering import FeatureEngineer

            # Extract comprehensive features for this symbol
            feature_engineer = FeatureEngineer(self.db_path)
            features_dict = feature_engineer.extract_features(symbol)

            if features_dict is None or len(features_dict) < 20:
                logger.warning(f"{symbol}: Insufficient features, using simple combination")
                # Fallback to simple combination
                combined_signal = 0.6 * sentiment_signal
                if abs(pairs_zscore) > 1.5:
                    combined_signal += -np.sign(pairs_zscore) * 0.4
                final_signal = 1 if combined_signal > 0.3 else -1 if combined_signal < -0.3 else 0
                confidence = min(abs(combined_signal), 1.0)
                return final_signal, confidence

            # Convert features dict to numpy array (ensure same order as training)
            feature_names = sorted(features_dict.keys())
            features = np.array([features_dict[name] for name in feature_names]).reshape(1, -1)

            # Get Keras MLP prediction (no scaling needed - model handles it)
            prediction_proba = self.meta_model.predict(features, verbose=0)[0]  # Shape: (3,) for [BUY, HOLD, SELL]

            # Get predicted class and confidence
            predicted_class = int(np.argmax(prediction_proba))  # 0=BUY, 1=HOLD, 2=SELL
            confidence = float(np.max(prediction_proba))

            if confidence < self.CONFIDENCE_THRESHOLD:
                logger.debug(f"{symbol}: Low confidence ({confidence:.2f}), HOLD")
                return 0, confidence

            # Convert class to signal
            if predicted_class == 0:  # BUY
                final_signal = 1
            elif predicted_class == 2:  # SELL
                final_signal = -1
            else:  # HOLD
                final_signal = 0

            logger.debug(f"{symbol}: MLP prediction={predicted_class} (BUY/HOLD/SELL), signal={final_signal}, confidence={confidence:.2f}")

            return final_signal, confidence

        except Exception as e:
            logger.warning(f"Error using Keras MLP for {symbol}: {e}")
            # Fallback to simple combination
            combined_signal = 0.6 * sentiment_signal
            if abs(pairs_zscore) > 1.5:
                combined_signal += -np.sign(pairs_zscore) * 0.4
            final_signal = 1 if combined_signal > 0.3 else -1 if combined_signal < -0.3 else 0
            confidence = min(abs(combined_signal), 1.0)
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

    def _monthly_liquidation(self) -> None:
        """Liquidate all positions on the 1st of each month.

        WHY: Realizes all profits and losses, gives a fresh start each month.
        This is like "taking profits off the table" - you capture gains and
        avoid holding losing positions indefinitely.

        HOW IT WORKS:
        1. Get all current positions
        2. Sell each position at market price (instant execution)
        3. Skip inverse ETFs (hedge_manager handles those separately)
        4. Log total proceeds from sales

        OOP CONCEPT: This is a METHOD of CombinedStrategy class.
        It has access to self.get_positions(), self.create_order(), etc.
        """
        try:
            logger.info("=" * 80)
            logger.info("📅 MONTHLY LIQUIDATION - 1st of the month")
            logger.info("=" * 80)

            # Get all current positions (stocks we own)
            positions = self.get_positions()

            if not positions or len(positions) == 0:
                logger.info("No positions to liquidate")
                return

            total_proceeds = 0.0
            positions_sold = 0

            # Inverse ETFs to skip (hedge_manager handles these)
            inverse_etfs = ["SH", "SPXS", "SQQQ", "PSQ", "DOG", "DXD", "SDS", "SDOW", "SPXU", "SOXS", "TZA"]

            logger.info(f"Liquidating {len(positions)} positions...")

            for position in positions:
                try:
                    symbol = position.symbol
                    quantity = position.quantity

                    # Skip inverse ETFs (let hedge_manager handle them)
                    if symbol in inverse_etfs:
                        logger.info(f"⏭️  Skipping inverse ETF: {symbol} (hedge_manager handles this)")
                        continue

                    # Get current price
                    current_price = self.get_last_price(symbol)
                    if not current_price or current_price <= 0:
                        logger.warning(f"Could not get price for {symbol}, skipping")
                        continue

                    # Calculate proceeds (what we'll get from selling)
                    proceeds = quantity * current_price

                    # Create market sell order (sells immediately at best available price)
                    order = self.create_order(
                        symbol,
                        quantity,
                        "sell",
                        type="market"
                    )
                    self.submit_order(order)

                    # Log the sale
                    logger.info(f"💰 SOLD {symbol}: {quantity:.2f} shares @ ${current_price:.2f} = ${proceeds:,.2f}")

                    total_proceeds += proceeds
                    positions_sold += 1

                except Exception as e:
                    logger.error(f"Error liquidating {position.symbol}: {e}")
                    continue

            logger.info("=" * 80)
            logger.info(f"✅ Monthly liquidation complete:")
            logger.info(f"   Positions sold: {positions_sold}")
            logger.info(f"   Total proceeds: ${total_proceeds:,.2f}")
            logger.info(f"   Fresh start for the new month!")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Error during monthly liquidation: {e}")
            import traceback
            traceback.print_exc()

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

        # CRITICAL FIX: Ensure model is loaded on first trading iteration (lazy loading)
        # This ensures TensorFlow doesn't block during bot initialization
        self._ensure_model_ready()

        # Check if it's the 1st of the month for monthly liquidation
        today = datetime.now()
        if today.day == 1:
            logger.info("📅 It's the 1st of the month - running monthly liquidation")
            self._monthly_liquidation()
            logger.info("✅ Monthly liquidation complete - fresh start for the month")
            return  # Exit early, let positions reset

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
                        technicals = self._get_enhanced_technicals(symbol)
                        logger.debug(f"📊 {symbol}: RSI={rsi:.1f}, Vol={volatility:.1%}")

                        # FIXED: Relaxed volatility threshold (was 0.6, now 0.8)
                        if volatility > 0.8:
                            logger.debug(f"   {symbol}: Extremely volatile, skipping")
                            continue

                        position = self.get_position(symbol)
                        price = self.get_last_price(symbol)

                        if not price or price <= 0:
                            continue

                        # MULTI-CONDITION ENTRY LOGIC (FIXED: Much more flexible than RSI < 40)
                        should_buy = False
                        entry_reason = ""

                        if position is None and cash > 500:
                            # CONDITION 1: RSI oversold (relaxed from 40 to 50)
                            if rsi < 50:
                                should_buy = True
                                entry_reason = f"RSI oversold ({rsi:.1f})"

                            # CONDITION 2: MACD bullish crossover
                            elif technicals and technicals.get('macd') and technicals.get('macd_signal'):
                                macd = technicals['macd']
                                macd_signal = technicals['macd_signal']
                                macd_hist = technicals.get('macd_histogram', 0)

                                # Bullish: MACD above signal AND histogram positive
                                if macd > macd_signal and macd_hist > 0:
                                    should_buy = True
                                    entry_reason = f"MACD bullish crossover (hist: {macd_hist:.2f})"

                            # CONDITION 3: Price near lower Bollinger Band (bounce play)
                            elif technicals and technicals.get('bb_lower') is not None:
                                bb_lower = technicals['bb_lower']
                                bb_middle = technicals.get('bb_middle')

                                # Buy if price is within 2% of lower band
                                if price <= bb_lower * 1.02:
                                    should_buy = True
                                    entry_reason = f"Near lower BB (${price:.2f} vs ${bb_lower:.2f})"

                            # CONDITION 4: Price above SMA50 but RSI not overbought (trend following)
                            elif technicals and technicals.get('sma_50') is not None and rsi < 65:
                                sma_50 = technicals['sma_50']
                                if price > sma_50 and rsi >= 45:
                                    should_buy = True
                                    entry_reason = f"Trend following (price above SMA50, RSI {rsi:.1f})"

                        if should_buy:
                            opportunities_found += 1

                            # Position sizing with volatility adjustment
                            base_quantity = min(cash * 0.05, 500) / price
                            vol_multiplier = self._calculate_volatility_multiplier(volatility)
                            quantity = base_quantity * vol_multiplier * self.current_position_size_multiplier

                            if quantity < 1:
                                logger.info(f"⏭️  Skipping {symbol}: position size too small")
                                continue

                            success = self._create_bracket_order(symbol, quantity, "buy", price)
                            if success:
                                # Log with position sizing details
                                sizing_msg = ""
                                if vol_multiplier != 1.0:
                                    sizing_msg += f"vol={vol_multiplier:.1f}x"
                                if self.current_position_size_multiplier < 1.0:
                                    if sizing_msg:
                                        sizing_msg += f", circuit={self.current_position_size_multiplier:.1f}x"
                                    else:
                                        sizing_msg += f"circuit={self.current_position_size_multiplier:.1f}x"

                                if sizing_msg:
                                    logger.info(f"📈 BUY {symbol}: {quantity:.2f} shares @ ${price:.2f} | {entry_reason} [{sizing_msg}]")
                                else:
                                    logger.info(f"📈 BUY {symbol}: {quantity:.2f} shares @ ${price:.2f} | {entry_reason}")
                                cash -= quantity * price

                                try:
                                    self.daily_pnl.update(portfolio_value, trade_executed=True)
                                except (AttributeError, TypeError) as e:
                                    logger.warning(f"Failed to update daily P&L: {e}")

                        elif rsi > 70 and position is not None:
                            logger.info(f"📊 {symbol} overbought (RSI: {rsi:.1f}) - consider selling")
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
                    logger.info(f"   Entry conditions: RSI<50 OR MACD bullish OR near BB lower OR trend following")
                    logger.info(f"   Exit: RSI>70 (overbought)")
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
