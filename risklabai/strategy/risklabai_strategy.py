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
        n_cv_splits: int = 5
    ):
        """
        Initialize RiskLabAI strategy.

        Args:
            profit_taking: Take-profit multiplier (vs volatility)
            stop_loss: Stop-loss multiplier (vs volatility)
            max_holding: Max periods before timeout
            n_cv_splits: Cross-validation folds
        """
        # Initialize components
        self.cusum_filter = CUSUMEventFilter()
        self.frac_diff = FractionalDifferentiator()
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
        cv = self.cv.get_cv(labels[['t1']])

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
        bars: pd.DataFrame
    ) -> Tuple[int, float]:
        """
        Generate trading signal with bet size.

        Args:
            bars: Recent bar data for prediction

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

        # Primary model: direction
        direction = self.primary_model.predict(X)[0]

        # Meta model: should we trade? (probability)
        if self.meta_model is not None:
            trade_prob = self.meta_model.predict_proba(X)[0, 1]

            # Convert probability to bet size
            if trade_prob < 0.5:
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
