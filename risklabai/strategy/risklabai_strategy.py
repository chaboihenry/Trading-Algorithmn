"""
RiskLabAI-Based Trading Strategy

This is the main strategy class that orchestrates all RiskLabAI components:
1. Generate information-driven bars from tick data
2. Apply fractional differentiation for stationary features
3. Use CUSUM filter for event sampling
4. Label with triple-barrier method
5. Train model with purged cross-validation
6. Size bets with meta-labeling
7. Optimize portfolio with HRP
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Import RiskLabAI components
from risklabai.data_structures.bars import BarGenerator
from risklabai.labeling.triple_barrier import TripleBarrierLabeler
from risklabai.labeling.meta_labeling import MetaLabeler
from risklabai.features.fractional_diff import FractionalDifferentiator
from risklabai.sampling.cusum_filter import CUSUMEventFilter
from risklabai.cross_validation.purged_kfold import PurgedCrossValidator
from risklabai.features.feature_importance import FeatureImportanceAnalyzer
from risklabai.portfolio.hrp import HRPPortfolio

# Import tick data components
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from config.tick_config import TICK_DB_PATH, INITIAL_IMBALANCE_THRESHOLD
    from data.tick_storage import TickStorage
    from data.tick_to_bars import generate_bars_from_ticks
    TICK_DATA_AVAILABLE = True
except ImportError:
    TICK_DATA_AVAILABLE = False
    logger.warning("Tick data components not available - train_from_ticks will not work")

logger = logging.getLogger(__name__)


class RiskLabAIStrategy:
    """
    Complete trading strategy using RiskLabAI framework.

    Pipeline:
    ┌──────────────────────────────────────────────────────────┐
    │ Raw Tick/Bar Data                                        │
    │         ↓                                                │
    │ CUSUM Filter (Event Sampling)                            │
    │         ↓                                                │
    │ Fractional Differentiation (Stationary Features)         │
    │         ↓                                                │
    │ Triple-Barrier Labeling (Dynamic Labels)                 │
    │         ↓                                                │
    │ Primary Model (Direction: Long/Short)                    │
    │         ↓                                                │
    │ Meta-Labeling (Bet Sizing Model)                         │
    │         ↓                                                │
    │ Purged CV Validation                                     │
    │         ↓                                                │
    │ HRP Portfolio Optimization                               │
    │         ↓                                                │
    │ Trade Execution                                          │
    └──────────────────────────────────────────────────────────┘

    Attributes:
        cusum_filter: Samples meaningful events
        frac_diff: Makes features stationary
        labeler: Creates triple-barrier labels
        meta_labeler: Sizes bets
        cv: Purged cross-validator
        hrp: Portfolio optimizer
        primary_model: Direction prediction model
        meta_model: Bet sizing model
    """

    def __init__(
        self,
        profit_taking: float = 2.0,
        stop_loss: float = 2.0,
        max_holding: int = 10,
        d: float = None,
        n_cv_splits: int = 5
    ):
        """
        Initialize RiskLabAI strategy.

        Args:
            profit_taking: Take-profit multiplier (vs volatility)
            stop_loss: Stop-loss multiplier (vs volatility)
            max_holding: Max periods before timeout
            d: Fractional differencing parameter (0-1, typically 0.3-0.6)
            n_cv_splits: Cross-validation folds
        """
        # Initialize components
        self.cusum_filter = CUSUMEventFilter()
        self.frac_diff = FractionalDifferentiator(d=d)  # Pass d parameter
        self.labeler = TripleBarrierLabeler(
            profit_taking_mult=profit_taking,
            stop_loss_mult=stop_loss,
            max_holding_period=max_holding
        )
        self.meta_labeler = MetaLabeler()
        self.cv = PurgedCrossValidator(n_splits=n_cv_splits)
        self.fi_analyzer = FeatureImportanceAnalyzer(method='mda')
        self.hrp = HRPPortfolio()

        # Models (initialized during training)
        self.primary_model = None
        self.meta_model = None

        # Feature parameters
        self.feature_names = None
        self.important_features = None

        logger.info("RiskLabAI Strategy initialized")

    def prepare_features(
        self,
        bars: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create feature matrix from bar data.

        Features:
        - Fractionally differentiated close prices
        - Returns at multiple horizons
        - Volatility measures
        - Volume features
        - Technical indicators

        Args:
            bars: OHLCV bar data

        Returns:
            Feature DataFrame (stationary)
        """
        features = pd.DataFrame(index=bars.index)

        # Fractionally differentiated close
        try:
            features['frac_diff_close'] = self.frac_diff.transform(bars['close'])
        except Exception as e:
            logger.warning(f"Fractional diff failed: {e}, using returns")
            features['frac_diff_close'] = bars['close'].pct_change()

        # Returns at multiple horizons
        for h in [1, 5, 10, 20]:
            features[f'ret_{h}'] = bars['close'].pct_change(h)

        # Volatility
        features['volatility'] = bars['close'].pct_change().rolling(20).std()
        features['volatility_ratio'] = features['volatility'] / features['volatility'].rolling(60).mean()

        # Volume features
        if 'volume' in bars.columns:
            features['volume_ma_ratio'] = bars['volume'] / bars['volume'].rolling(20).mean()
            features['volume_volatility'] = bars['volume'].rolling(20).std() / bars['volume'].rolling(20).mean()

        # Price features
        features['close_to_high'] = (bars['high'] - bars['close']) / bars['close']
        features['close_to_low'] = (bars['close'] - bars['low']) / bars['close']
        features['daily_range'] = (bars['high'] - bars['low']) / bars['close']

        # Moving average crossovers
        features['sma_20'] = bars['close'].rolling(20).mean() / bars['close']
        features['sma_50'] = bars['close'].rolling(50).mean() / bars['close']

        # Drop NaN rows
        features = features.dropna()

        self.feature_names = features.columns.tolist()

        logger.debug(f"Created {len(features.columns)} features")

        return features

    def train(
        self,
        bars: pd.DataFrame,
        min_samples: int = 100
    ) -> Dict:
        """
        Train primary and meta models.

        Args:
            bars: OHLCV bar data
            min_samples: Minimum samples required

        Returns:
            Training results dictionary
        """
        logger.info("=" * 60)
        logger.info("TRAINING RISKLABAI MODELS")
        logger.info("=" * 60)

        # Step 1: Sample events with CUSUM filter
        logger.info("Step 1: Filtering events with CUSUM...")
        events = self.cusum_filter.get_events(bars['close'])
        logger.info(f"  Found {len(events)} significant events")

        if len(events) < min_samples:
            logger.warning(f"Insufficient events ({len(events)} < {min_samples})")
            return {'success': False, 'reason': 'insufficient_events'}

        # Step 2: Create features
        logger.info("Step 2: Creating stationary features...")
        features = self.prepare_features(bars)

        # Align features with events
        features = features.loc[features.index.isin(events)]
        logger.info(f"  Feature matrix shape: {features.shape}")

        if len(features) < min_samples:
            logger.warning(f"Insufficient samples ({len(features)} < {min_samples})")
            return {'success': False, 'reason': 'insufficient_samples'}

        # Step 3: Create triple-barrier labels
        logger.info("Step 3: Creating triple-barrier labels...")
        # Create event DataFrame (vertical barrier will be added by labeler)
        event_df = pd.DataFrame(index=events)

        labels = self.labeler.label(
            close=bars['close'],
            events=event_df
        )

        # Align labels with features
        labels = labels.loc[labels.index.isin(features.index)]
        features = features.loc[features.index.isin(labels.index)]

        logger.info(f"  Label distribution: {labels['bin'].value_counts().to_dict()}")

        # Step 4: Train primary model (direction)
        logger.info("Step 4: Training primary model...")
        X = features
        y_direction = labels['bin']

        self.primary_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=42
        )

        # Use purged CV for validation
        # Create samples_info DataFrame with 't0' (start) and 't1' (end) times
        samples_info = pd.DataFrame(index=labels.index)
        samples_info['t1'] = pd.to_datetime(labels['End Time'])

        # CRITICAL FIX: Ensure timezone consistency for cross-validation
        # The purged K-fold can't handle mixed tz-aware and tz-naive timestamps
        if samples_info.index.tz is not None:
            # Index has timezone - make sure t1 also has it
            if samples_info['t1'].dt.tz is None:
                samples_info['t1'] = samples_info['t1'].dt.tz_localize(samples_info.index.tz)
            elif samples_info['t1'].dt.tz != samples_info.index.tz:
                # Different timezones - convert t1 to match index
                samples_info['t1'] = samples_info['t1'].dt.tz_convert(samples_info.index.tz)
        elif samples_info['t1'].dt.tz is not None:
            # t1 has timezone but index doesn't - remove timezone from t1
            samples_info['t1'] = samples_info['t1'].dt.tz_localize(None)

        cv = self.cv.get_cv(samples_info)

        scores = cross_val_score(
            self.primary_model,
            X,
            y_direction,
            cv=cv,
            scoring='accuracy',
            n_jobs=1
        )
        logger.info(f"  Primary model CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

        # Fit on all data
        self.primary_model.fit(X, y_direction)

        # Step 5: Feature importance analysis
        logger.info("Step 5: Analyzing feature importance...")
        try:
            importance = self.fi_analyzer.calculate_importance(
                self.primary_model,
                X,
                y_direction,
                cv=cv
            )
            self.important_features = importance.nlargest(10, 'importance')['feature'].tolist()
            logger.info(f"  Top features: {self.important_features[:5]}")
        except Exception as e:
            logger.warning(f"Feature importance failed: {e}")
            self.important_features = features.columns[:10].tolist()

        # Step 6: Train meta model (bet sizing)
        logger.info("Step 6: Training meta model...")

        # Create meta-labels from triple barrier results
        meta_labels = self.meta_labeler.create_meta_labels(
            events=labels,  # Triple barrier output
            close=bars['close']
        )

        # Align meta labels with features
        meta_labels = meta_labels.loc[meta_labels.index.isin(X.index)]
        X_meta = X.loc[X.index.isin(meta_labels.index)]

        if len(meta_labels) < min_samples // 2:
            logger.warning(f"Insufficient meta-labels ({len(meta_labels)}), skipping meta model")
            self.meta_model = None
            meta_scores_mean = 0.0
        else:
            self.meta_model = RandomForestClassifier(
                n_estimators=50,
                max_depth=3,
                min_samples_leaf=20,
                n_jobs=-1,
                random_state=42
            )

            y_meta = meta_labels['bin']

            self.meta_model.fit(X_meta, y_meta)

            meta_scores = cross_val_score(
                self.meta_model,
                X_meta,
                y_meta,
                cv=cv,
                scoring='accuracy',
                n_jobs=1
            )
            meta_scores_mean = meta_scores.mean()
            logger.info(f"  Meta model CV accuracy: {meta_scores_mean:.3f} ± {meta_scores.std():.3f}")

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)

        return {
            'success': True,
            'n_samples': len(features),
            'primary_accuracy': scores.mean(),
            'meta_accuracy': meta_scores_mean,
            'top_features': self.important_features
        }

    def predict(
        self,
        bars: pd.DataFrame,
        prob_threshold: float = 0.015,
        meta_threshold: float = 0.0001
    ) -> Tuple[int, float]:
        """
        Generate trading signal with bet size.

        Args:
            bars: Recent bar data for prediction
            prob_threshold: Probability threshold for trading (0-1).
                          Higher = more conservative, lower = more aggressive.
                          Default 0.015 (1.5%) optimized for this model.
            meta_threshold: Meta model probability threshold (0-1).
                          Default 0.0001 (0.01%) optimized for this model.

        Returns:
            Tuple of (signal, bet_size):
            - signal: +1 (long), -1 (short), 0 (no trade)
            - bet_size: 0 to 1 (sizing based on confidence)
        """
        if self.primary_model is None:
            logger.warning("Primary model not trained!")
            return 0, 0.0

        # Create features from latest data
        features = self.prepare_features(bars)

        if len(features) == 0:
            return 0, 0.0

        # Get latest feature row
        X = features.iloc[[-1]]

        # Primary model: probability-based direction
        # CRITICAL FIX: Use predict_proba instead of predict to avoid
        # the model being too conservative (always predicting class 0)
        probs = self.primary_model.predict_proba(X)[0]
        # probs[0] = P(short=-1), probs[1] = P(no_trade=0), probs[2] = P(long=1)

        if probs[2] > prob_threshold:
            # Long signal if long probability exceeds threshold
            direction = 1
        elif probs[0] > prob_threshold:
            # Short signal if short probability exceeds threshold
            direction = -1
        else:
            # No trade if neither direction shows sufficient conviction
            direction = 0

        # Meta model: should we trade? (probability)
        if self.meta_model is not None:
            trade_prob = self.meta_model.predict_proba(X)[0, 1]

            # Convert probability to bet size
            if trade_prob < meta_threshold:
                # Don't trade if meta model says no
                return 0, 0.0
            else:
                # Bet size proportional to confidence
                bet_size = min(trade_prob, 1.0)
                logger.debug(f"Signal: {direction}, Bet size: {bet_size:.3f}")
                return int(direction), float(bet_size)
        else:
            # No meta model - use fixed bet size
            logger.debug(f"Signal: {direction}, Bet size: 0.5 (no meta model)")
            return int(direction), 0.5

    def optimize_portfolio(
        self,
        returns: pd.DataFrame
    ) -> pd.Series:
        """
        Optimize portfolio weights using HRP.

        Args:
            returns: DataFrame of asset returns

        Returns:
            Series of optimal weights
        """
        return self.hrp.optimize(returns)

    def save_models(self, path: str):
        """Save trained models to disk."""
        import joblib

        if self.primary_model is None or self.meta_model is None:
            logger.warning("No models to save")
            return

        joblib.dump({
            'primary_model': self.primary_model,
            'meta_model': self.meta_model,
            'feature_names': self.feature_names,
            'important_features': self.important_features
        }, path)

        logger.info(f"Models saved to {path}")

    def load_models(self, path: str):
        """Load trained models from disk."""
        import joblib

        data = joblib.load(path)

        self.primary_model = data['primary_model']
        self.meta_model = data['meta_model']
        self.feature_names = data['feature_names']
        self.important_features = data['important_features']

        logger.info(f"Models loaded from {path}")

    def train_from_ticks(
        self,
        symbol: str,
        threshold: Optional[float] = None,
        min_samples: int = 100
    ) -> Dict:
        """
        Train models from tick data stored in database.

        This method:
        1. Loads tick data from the Vault database
        2. Generates tick imbalance bars
        3. Converts bars to DataFrame
        4. Runs the full RiskLabAI training pipeline

        Args:
            symbol: Stock ticker (e.g., "SPY", "QQQ")
            threshold: Imbalance threshold (uses config default if None)
            min_samples: Minimum samples required for training

        Returns:
            Training results dictionary

        Example:
            >>> strategy = RiskLabAIStrategy()
            >>> results = strategy.train_from_ticks('SPY')
            >>> if results['success']:
            ...     print(f"Trained on {results['n_samples']} samples")
            ...     strategy.save_models('models/risklabai_models.pkl')
        """
        if not TICK_DATA_AVAILABLE:
            raise ImportError(
                "Tick data components not available. "
                "Make sure tick data infrastructure is installed."
            )

        logger.info("=" * 80)
        logger.info(f"TRAINING FROM TICK DATA: {symbol}")
        logger.info("=" * 80)

        # Use configured threshold if not specified
        if threshold is None:
            threshold = INITIAL_IMBALANCE_THRESHOLD
            logger.info(f"Using threshold from config: {threshold:.2f}")

        # Step 1: Load ticks from database
        logger.info("Step 1: Loading ticks from database...")
        storage = TickStorage(TICK_DB_PATH)

        # Get available date range
        date_range = storage.get_date_range(symbol)
        if not date_range:
            storage.close()
            raise ValueError(
                f"No tick data found for {symbol} in database. "
                f"Run scripts/backfill_ticks.py first."
            )

        earliest, latest = date_range
        logger.info(f"  Available data: {earliest} to {latest}")

        # Load all ticks
        ticks = storage.load_ticks(symbol)
        storage.close()

        if not ticks:
            raise ValueError(f"No ticks loaded for {symbol}")

        logger.info(f"  Loaded {len(ticks):,} ticks")

        # Step 2: Generate tick imbalance bars
        logger.info("Step 2: Generating tick imbalance bars...")
        bars_list = generate_bars_from_ticks(ticks, threshold=threshold)

        if not bars_list:
            raise ValueError(f"No bars generated from {len(ticks)} ticks")

        logger.info(f"  Generated {len(bars_list)} bars")
        logger.info(f"  Bars per tick: {len(bars_list)/len(ticks):.4f}")

        # Step 3: Convert to DataFrame for RiskLabAI
        logger.info("Step 3: Converting bars to DataFrame...")

        # Create DataFrame from bars
        bars_df = pd.DataFrame(bars_list)

        # Set datetime index (remove timezone to avoid issues with purged CV)
        bars_df['bar_end'] = pd.to_datetime(bars_df['bar_end'])
        if bars_df['bar_end'].dt.tz is not None:
            bars_df['bar_end'] = bars_df['bar_end'].dt.tz_localize(None)
        bars_df.set_index('bar_end', inplace=True)

        # Ensure we have required OHLCV columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in bars_df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")

        logger.info(f"  DataFrame shape: {bars_df.shape}")
        logger.info(f"  Columns: {bars_df.columns.tolist()}")

        # Step 4: Run RiskLabAI training pipeline
        logger.info("Step 4: Running RiskLabAI training pipeline...")
        results = self.train(bars_df, min_samples=min_samples)

        if results['success']:
            logger.info("=" * 80)
            logger.info(f"✓ TRAINING SUCCESSFUL FROM TICK DATA")
            logger.info("=" * 80)
            logger.info(f"  Tick data: {len(ticks):,} ticks")
            logger.info(f"  Bars generated: {len(bars_list)}")
            logger.info(f"  Samples used: {results['n_samples']}")
            logger.info(f"  Primary accuracy: {results['primary_accuracy']:.3f}")
            logger.info(f"  Meta accuracy: {results['meta_accuracy']:.3f}")
            logger.info("=" * 80)

        return results
