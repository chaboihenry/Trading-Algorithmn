import logging
import pytz
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from risklabai.data_structures.bars import BarGenerator
from risklabai.labeling.triple_barrier import TripleBarrierLabeler
from risklabai.labeling.meta_labeling import MetaLabeler
from risklabai.features.fractional_diff import FractionalDifferentiator
from risklabai.sampling.cusum_filter import CUSUMEventFilter
from risklabai.cross_validation.purged_kfold import PurgedCrossValidator
from risklabai.portfolio.hrp import HRPPortfolio

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from config.tick_config import TICK_DB_PATH, INITIAL_IMBALANCE_THRESHOLD, CUSUM_EVENT_WINDOW_SECONDS
    from data.tick_storage import TickStorage
    from data.tick_to_bars import generate_bars_from_ticks
    TICK_DATA_AVAILABLE = True
except ImportError:
    TICK_DATA_AVAILABLE = False
    logging.warning("Tick data components not available. `train_from_ticks` will not work.")

logger = logging.getLogger(__name__)


class RiskLabAIStrategy:
    """
    An implementation of a trading strategy based on the RiskLabAI framework.
    It orchestrates event sampling, bar generation, feature engineering,
    labeling, model training, and portfolio optimization.
    """

    def __init__(
        self,
        profit_taking: float = 2.5,
        stop_loss: float = 2.5,
        max_holding: int = 10,
        n_cv_splits: int = 5,
        margin_threshold: float = 0.03
    ):
        self.labeler = TripleBarrierLabeler(
            profit_taking_mult=profit_taking,
            stop_loss_mult=stop_loss,
            max_holding_period=max_holding
        )
        self.meta_labeler = MetaLabeler()
        self.cv = PurgedCrossValidator(n_splits=n_cv_splits)
        self.hrp = HRPPortfolio()

        self.primary_model = None
        self.meta_model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.margin_threshold = margin_threshold

        logger.info(f"RiskLabAI Strategy initialized with margin threshold: {self.margin_threshold:.1%}")

    def prepare_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        Generates a set of stationary technical features.
        """
        features = pd.DataFrame(index=bars.index)
        close = bars['close']
        volume = bars.get('volume', pd.Series(0, index=bars.index))

        # Log Returns (1-period, 5-period)
        features['log_ret_1'] = np.log(close / close.shift(1))
        features['log_ret_5'] = np.log(close / close.shift(5))

        # Normalized Volatility (rolling std of returns)
        returns_std = features['log_ret_1'].rolling(window=20).std()
        features['norm_volatility'] = features['log_ret_1'] / (returns_std + 1e-8)

        # RSI (centered at 0.5, scaled -0.5 to 0.5)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        features['rsi'] = (rsi / 100.0) - 0.5  # Scale to [-0.5, 0.5]

        # Bollinger Band Position (normalized z-score)
        rolling_mean = close.rolling(window=20).mean()
        rolling_std = close.rolling(window=20).std()
        features['bb_pos'] = (close - rolling_mean) / (rolling_std + 1e-8)

        # Volume Imbalance (signed volume)
        price_change_direction = np.sign(close.diff())
        features['vol_imbalance'] = price_change_direction * volume

        self.feature_names = features.columns.tolist()
        features = features.dropna()

        logger.info(f"Generated {len(self.feature_names)} features for {len(features)} samples.")
        return features

    def _build_primary_model(self) -> XGBClassifier:
        """Builds the primary XGBoost model with fixed, robust hyperparameters."""
        params = {
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'use_label_encoder': False,
            'n_jobs': -1,
            'random_state': 42,
            # Hardcoded parameters for generalization
            'max_depth': 2,
            'learning_rate': 0.01,
            'n_estimators': 1000,
            'num_class': 3,
        }
        return XGBClassifier(**params)

    def train(self, bars: pd.DataFrame, min_samples: int = 100, symbol: Optional[str] = None) -> Dict:
        """Trains the primary and meta models."""
        logger.info(f"--- Starting training for {symbol or 'asset'} ---")

        if bars.index.tz is not None:
            bars.index = bars.index.tz_convert('UTC').tz_localize(None)

        features = self.prepare_features(bars)
        if len(features) < min_samples:
            logger.warning(f"Training aborted: insufficient samples ({len(features)} < {min_samples}).")
            return {'success': False, 'reason': 'insufficient_samples'}

        labels = self.labeler.label(close=bars['close'], events=pd.DataFrame(index=features.index))
        
        aligned_index = features.index.intersection(labels.index)
        features = features.loc[aligned_index]
        labels = labels.loc[aligned_index]

        label_counts = labels['bin'].value_counts(normalize=True).mul(100)
        logger.info(f"Label distribution: Long={label_counts.get(1, 0):.1f}%, Short={label_counts.get(-1, 0):.1f}%, Neutral={label_counts.get(0, 0):.1f}%")

        X = features
        y_direction = labels['bin']
        self.label_encoder = LabelEncoder().fit(y_direction)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_direction, test_size=0.2, random_state=42, stratify=y_direction
        )
        logger.info(f"Train/test split: {len(X_train)} train, {len(X_test)} test samples.")

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        y_train_encoded = self.label_encoder.transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)

        # Train primary model
        self.primary_model = self._build_primary_model()
        self.primary_model.fit(
            X_train_scaled, y_train_encoded,
            eval_set=[(X_test_scaled, y_test_encoded)],
            verbose=False
        )
        
        train_acc = self.primary_model.score(X_train_scaled, y_train_encoded)
        test_acc = self.primary_model.score(X_test_scaled, y_test_encoded)
        logger.info(f"Primary Model Accuracy - Train: {train_acc:.2%}, Test: {test_acc:.2%}")

        # Train meta model
        primary_train_preds = self.label_encoder.inverse_transform(self.primary_model.predict(X_train_scaled))
        meta_labels_train = (primary_train_preds == y_train.values).astype(int)
        
        self.meta_model = LogisticRegression(class_weight='balanced', random_state=42)
        self.meta_model.fit(X_train_scaled, meta_labels_train)

        primary_test_preds = self.label_encoder.inverse_transform(self.primary_model.predict(X_test_scaled))
        meta_labels_test = (primary_test_preds == y_test.values).astype(int)
        meta_test_acc = self.meta_model.score(X_test_scaled, meta_labels_test)
        logger.info(f"Meta Model Accuracy (Test): {meta_test_acc:.2%}")

        logger.info(f"--- Training for {symbol or 'asset'} complete ---")
        return {
            'success': True,
            'n_samples': len(features),
            'primary_accuracy': test_acc,
            'meta_accuracy': meta_test_acc,
        }

    def predict(self, bars: pd.DataFrame, prob_threshold: float = 0.015, meta_threshold: float = 0.5) -> Tuple[int, float]:
        """Generates a trading signal and bet size."""
        if self.primary_model is None or self.scaler is None or self.label_encoder is None:
            raise ValueError("Model components are not trained. Call train() first.")

        features = self.prepare_features(bars)
        if features.empty:
            return 0, 0.0
        
        if list(features.columns) != self.feature_names:
            raise ValueError("Feature mismatch between training and prediction.")

        X_scaled = self.scaler.transform(features.iloc[[-1]])
        
        probs = self.primary_model.predict_proba(X_scaled)[0]
        label_to_prob = dict(zip(self.label_encoder.classes_, probs))

        prob_long = label_to_prob.get(1, 0.0)
        prob_short = label_to_prob.get(-1, 0.0)

        direction = 0
        if prob_long > prob_short and prob_long - prob_short > self.margin_threshold:
            direction = 1
        elif prob_short > prob_long and prob_short - prob_long > self.margin_threshold:
            direction = -1

        if direction == 0:
            return 0, 0.0

        meta_confidence = self.meta_model.predict_proba(X_scaled)[0][1] # P(primary is correct)
        
        if meta_confidence < meta_threshold:
            return 0, 0.0

        return direction, meta_confidence

    def save_models(self, path: str):
        """Saves trained models and preprocessing objects to a file."""
        import joblib
        if not all([self.primary_model, self.meta_model, self.scaler, self.label_encoder]):
            logger.warning("Attempted to save models, but not all components are trained.")
            return

        model_package = {
            'primary_model': self.primary_model,
            'meta_model': self.meta_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'train_date': datetime.now().isoformat(),
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_package, path)
        logger.info(f"Models saved successfully to {path}")

    def load_models(self, path: str):
        """Loads models and preprocessing objects from a file."""
        import joblib
        model_package = joblib.load(path)
        
        self.primary_model = model_package['primary_model']
        self.meta_model = model_package['meta_model']
        self.scaler = model_package['scaler']
        self.label_encoder = model_package['label_encoder']
        self.feature_names = model_package['feature_names']
        
        train_date = model_package.get('train_date', 'N/A')
        logger.info(f"Models loaded from {path} (trained on {train_date})")

    def train_from_ticks(self, symbol: str, threshold: Optional[float] = None, min_samples: int = 100) -> Dict:
        """Full pipeline: trains models from raw tick data in the database."""
        if not TICK_DATA_AVAILABLE:
            raise ImportError("Tick data components not available.")

        logger.info(f"--- Training from raw ticks for {symbol} ---")
        
        storage = TickStorage(str(TICK_DB_PATH))
        ticks = storage.load_ticks(symbol)
        storage.close()
        
        if not ticks:
            raise ValueError(f"No ticks found for {symbol}. Run backfill script.")
        logger.info(f"Loaded {len(ticks):,} ticks.")

        # In a real scenario, CUSUM filtering on ticks would happen here.
        # For this refactor, we generate bars directly from all ticks.
        
        bar_threshold = threshold or INITIAL_IMBALANCE_THRESHOLD
        bars_list = generate_bars_from_ticks(ticks, threshold=bar_threshold)
        if len(bars_list) < min_samples:
            raise ValueError(f"Not enough bars ({len(bars_list)}) generated. Try adjusting threshold.")
        logger.info(f"Generated {len(bars_list)} bars with threshold {bar_threshold}.")

        bars_df = pd.DataFrame(bars_list)
        bars_df['bar_end'] = pd.to_datetime(bars_df['bar_end'])
        bars_df = bars_df.set_index('bar_end').sort_index()
        bars_df = bars_df[~bars_df.index.duplicated(keep='last')]

        return self.train(bars_df, min_samples=min_samples, symbol=symbol)
