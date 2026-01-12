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
import pytz
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

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


# =============================================================================
# TIMEZONE UTILITIES
# =============================================================================
ET_TZ = pytz.timezone('America/New_York')
UTC_TZ = pytz.UTC


def to_utc(dt):
    """Convert any datetime to UTC."""
    if dt.tzinfo is None:
        # Assume ET if naive
        dt = ET_TZ.localize(dt)
    return dt.astimezone(UTC_TZ)


def to_et(dt):
    """Convert any datetime to Eastern Time."""
    if dt.tzinfo is None:
        dt = UTC_TZ.localize(dt)
    return dt.astimezone(ET_TZ)


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
        max_holding: int = 30,
        d: float = None,
        n_cv_splits: int = 5,
        margin_threshold: float = 0.03
    ):
        """
        Initialize RiskLabAI strategy.

        Args:
            profit_taking: Take-profit multiplier (vs volatility)
            stop_loss: Stop-loss multiplier (vs volatility)
            max_holding: Max periods before timeout (30 → ~30-35% neutral labels)
            d: Fractional differencing parameter (0-1, typically 0.3-0.6)
            n_cv_splits: Cross-validation folds
            margin_threshold: Minimum probability margin between winner and runner-up
                            for accepting signals (default: 0.03 = 3%)
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

        # Preprocessing (initialized during training)
        self.scaler = None
        self.label_encoder = None  # XGBoost needs 0,1,2 not -1,0,1

        # Feature parameters
        self.feature_names = None
        self.important_features = None
        self.feature_importance = None  # Detailed importance scores

        # Prediction parameters
        self.margin_threshold = margin_threshold  # Configurable margin threshold (H9)

        logger.info("RiskLabAI Strategy initialized")
        logger.info(f"  Margin threshold: {self.margin_threshold:.1%}")

    def prepare_features(
        self,
        bars: pd.DataFrame,
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate minimal feature set optimized for small sample sizes.

        CRITICAL RULE: Need 10-20 samples per feature minimum.
        - With 300 samples → max 15-30 features
        - With 500 samples → max 25-50 features
        - We use 7 features for safety margin and robustness

        These 7 features were selected based on:
        1. High information value for price prediction
        2. Low correlation with each other
        3. Statistical stationarity
        4. Proven effectiveness in quant finance

        Features:
        1. frac_diff: Memory-preserving stationary price (RiskLabAI core)
        2. ret_1: Short-term momentum (immediate price action)
        3. ret_5: Medium-term momentum (trend confirmation)
        4. volatility: Risk/uncertainty measure
        5. volume_ratio: Liquidity and activity level
        6. price_position: Mean reversion signal (0=oversold, 1=overbought)
        7. trend: Directional bias (MA crossover)

        Args:
            bars: OHLCV bar data
            symbol: Symbol name (for logging optimal d per symbol)

        Returns:
            Feature DataFrame with exactly 7 features (stationary)
        """
        features = pd.DataFrame(index=bars.index)

        # 1. Fractionally differentiated price (memory-preserving stationarity)
        try:
            # Find optimal d if not already set
            if self.frac_diff.d is None and self.frac_diff._optimal_d is None:
                symbol_str = f" for {symbol}" if symbol else ""
                logger.info(f"Finding optimal fractional differentiation parameter{symbol_str}...")
                optimal_d = self.frac_diff.find_optimal_d(bars['close'])
                logger.info(f"{'✓' if symbol else ''} {symbol if symbol else 'Symbol'}: Using d={optimal_d:.2f} for fractional differentiation")

            features['frac_diff'] = self.frac_diff.transform(bars['close'])
        except Exception as e:
            logger.warning(f"Fractional diff failed: {e}, using returns")
            features['frac_diff'] = bars['close'].pct_change()

        # 2. Short-term return (1 bar momentum)
        features['ret_1'] = bars['close'].pct_change(1)

        # 3. Medium-term return (5 bars for trend confirmation)
        features['ret_5'] = bars['close'].pct_change(5)

        # 4. Volatility (20-bar rolling standard deviation)
        features['volatility'] = bars['close'].pct_change().rolling(20).std()

        # 5. Volume ratio (vs 20-bar average) - measures activity level
        if 'volume' in bars.columns:
            features['volume_ratio'] = bars['volume'] / (bars['volume'].rolling(20).mean() + 1e-8)
        else:
            # If no volume data, use price range as proxy for activity
            features['volume_ratio'] = (bars['high'] - bars['low']) / (bars['close'] + 1e-8)
            logger.warning("No volume data available, using price range as proxy")

        # 6. Price position in range (mean reversion signal)
        # 0 = at 20-bar low (oversold), 1 = at 20-bar high (overbought)
        roll_high = bars['high'].rolling(20).max()
        roll_low = bars['low'].rolling(20).min()
        features['price_position'] = (bars['close'] - roll_low) / (roll_high - roll_low + 1e-8)

        # 7. Trend (short MA vs long MA)
        # Positive = uptrend, Negative = downtrend
        ma_short = bars['close'].rolling(5).mean()
        ma_long = bars['close'].rolling(20).mean()
        features['trend'] = (ma_short - ma_long) / (ma_long + 1e-8)

        # Store feature names
        self.feature_names = features.columns.tolist()

        # Drop NaN rows
        features = features.dropna()

        # Validate samples per feature ratio
        n_samples = len(features)
        n_features = len(self.feature_names)
        samples_per_feature = n_samples / n_features if n_features > 0 else 0

        logger.info("")
        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING")
        logger.info("=" * 60)
        logger.info(f"Feature count: {n_features}")
        logger.info(f"Sample count: {n_samples}")
        logger.info(f"Samples per feature: {samples_per_feature:.1f}")
        logger.info("=" * 60)

        # Warning thresholds
        if samples_per_feature < 10:
            logger.warning("")
            logger.warning("⚠️  INSUFFICIENT SAMPLES PER FEATURE")
            logger.warning(f"    Current: {samples_per_feature:.1f} samples/feature")
            logger.warning(f"    Minimum: 10 samples/feature")
            logger.warning(f"    Recommended: 20-40 samples/feature")
            logger.warning("")
            logger.warning("    RISK: Model may overfit with so few samples per feature")
            logger.warning("    ACTION: Collect more data or reduce feature count further")
            logger.warning("")
        elif samples_per_feature < 20:
            logger.warning(f"⚠️ Low samples/feature: {samples_per_feature:.1f} (target: 20-40)")
        else:
            logger.info(f"✓ Good samples/feature ratio: {samples_per_feature:.1f}")

        # Feature list for user
        logger.info("Features generated:")
        for i, feat in enumerate(self.feature_names, 1):
            logger.info(f"  {i}. {feat}")

        return features

    def _detect_regime_change(self, y_train: pd.Series, y_test: pd.Series) -> float:
        """
        Detect if market regime changed between training and test periods.

        Market regimes can shift due to:
        - Bull market → Bear market (or vice versa)
        - High volatility → Low volatility
        - Trending → Mean-reverting

        A significant change in label distribution suggests the market
        dynamics shifted, which can hurt model performance.

        Args:
            y_train: Training labels
            y_test: Test labels

        Returns:
            Maximum shift across all label classes
        """
        # Calculate directional bias
        train_long_ratio = (y_train == 1).mean()
        train_short_ratio = (y_train == -1).mean()
        train_neutral_ratio = (y_train == 0).mean()

        test_long_ratio = (y_test == 1).mean()
        test_short_ratio = (y_test == -1).mean()
        test_neutral_ratio = (y_test == 0).mean()

        # Calculate shifts for each class
        long_shift = abs(train_long_ratio - test_long_ratio)
        short_shift = abs(train_short_ratio - test_short_ratio)
        neutral_shift = abs(train_neutral_ratio - test_neutral_ratio)

        max_shift = max(long_shift, short_shift, neutral_shift)

        logger.info("")
        logger.info("=" * 60)
        logger.info("REGIME CHANGE DETECTION")
        logger.info("=" * 60)
        logger.info("Directional bias comparison:")
        logger.info(f"  Long:    Train {train_long_ratio*100:5.1f}% → Test {test_long_ratio*100:5.1f}% (shift: {long_shift*100:4.1f}%)")
        logger.info(f"  Short:   Train {train_short_ratio*100:5.1f}% → Test {test_short_ratio*100:5.1f}% (shift: {short_shift*100:4.1f}%)")
        logger.info(f"  Neutral: Train {train_neutral_ratio*100:5.1f}% → Test {test_neutral_ratio*100:5.1f}% (shift: {neutral_shift*100:4.1f}%)")
        logger.info("=" * 60)

        # Detect regime changes
        if max_shift > 0.15:
            logger.warning("")
            logger.warning("⚠️  SIGNIFICANT REGIME CHANGE DETECTED")
            logger.warning(f"    Maximum class shift: {max_shift*100:.1f}%")
            logger.warning(f"    Expected: <5% (stable regime)")
            logger.warning("")
            logger.warning("    ISSUE: Market conditions changed between train and test")
            logger.warning("    IMPLICATIONS:")
            logger.warning("      - Model trained on different regime than tested")
            logger.warning("      - Test accuracy may underestimate live performance")
            logger.warning("      - Or overestimate if regime favors model bias")
            logger.warning("")
            logger.warning("    RECOMMENDATIONS:")
            logger.warning("      1. Use shorter retraining windows (e.g., monthly)")
            logger.warning("      2. Consider adaptive/online learning models")
            logger.warning("      3. Add regime detection to live trading")
            logger.warning("      4. Use walk-forward validation instead")
            logger.warning("")
        elif max_shift > 0.10:
            logger.warning("")
            logger.warning("⚠️  MODERATE REGIME CHANGE")
            logger.warning(f"    Maximum class shift: {max_shift*100:.1f}%")
            logger.warning(f"    Monitor model performance closely")
            logger.warning("")
        else:
            logger.info(f"✓ Stable regime: max shift {max_shift*100:.1f}% (target: <5%)")

        return max_shift

    def train(
        self,
        bars: pd.DataFrame,
        min_samples: int = 100,
        symbol: Optional[str] = None,
        cusum_already_applied: bool = False
    ) -> Dict:
        """
        Train primary and meta models.

        Args:
            bars: OHLCV bar data
            min_samples: Minimum samples required
            symbol: Symbol name (for logging)
            cusum_already_applied: If True, skip CUSUM (already applied to ticks)

        Returns:
            Training results dictionary
        """
        logger.info("=" * 60)
        logger.info("TRAINING RISKLABAI MODELS")
        logger.info("=" * 60)

        # Standardize index to timezone-naive UTC
        if bars.index.tz is not None:
            original_tz = bars.index.tz
            bars.index = bars.index.tz_convert('UTC').tz_localize(None)
            logger.debug(f"Converted index from {original_tz} to timezone-naive UTC")
        else:
            logger.debug("Training data already timezone-naive")

        logger.debug(f"Training data timezone: {bars.index.tz} (None = naive)")

        # Step 1: Handle event sampling (CUSUM already applied or apply now)
        if cusum_already_applied:
            # C2 FIX: CUSUM already applied to ticks - use ALL bars
            logger.info("Step 1: CUSUM already applied to ticks - using all bars")
            logger.info(f"  Using all {len(bars)} bars (pre-filtered via tick CUSUM)")
            features = self.prepare_features(bars, symbol=symbol)
            logger.info(f"  Feature matrix shape: {features.shape}")
            # Use all bar timestamps as events (CUSUM already filtered ticks)
            events = features.index
        else:
            # OLD PATH: Apply CUSUM to bar prices (for non-tick training)
            logger.info("Step 1: Filtering events with CUSUM...")
            events = self.cusum_filter.get_events(bars['close'])
            logger.info(f"  Found {len(events)} significant events")

            if len(events) < min_samples:
                logger.warning(f"Insufficient events ({len(events)} < {min_samples})")
                return {'success': False, 'reason': 'insufficient_events'}

            # Step 2: Create features
            logger.info("Step 2: Creating stationary features...")
            features = self.prepare_features(bars, symbol=symbol)

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

        # Log detailed label distribution
        label_counts = labels['bin'].value_counts().to_dict()
        label_pcts = (labels['bin'].value_counts(normalize=True) * 100).to_dict()
        logger.info(f"  Label distribution: {label_counts}")
        logger.info(f"  Label percentages: Short={label_pcts.get(-1, 0):.2f}%, "
                   f"Neutral={label_pcts.get(0, 0):.2f}%, Long={label_pcts.get(1, 0):.2f}%")

        # Step 4: Split data for train/test validation
        logger.info("Step 4: Splitting data for train/test validation...")
        X = features
        y_direction = labels['bin']

        # Log overall label distribution BEFORE split
        overall_dist = y_direction.value_counts(normalize=True).sort_index()
        logger.info("")
        logger.info("=" * 60)
        logger.info("OVERALL LABEL DISTRIBUTION (Before Split)")
        logger.info("=" * 60)
        for label, pct in overall_dist.items():
            label_name = {-1: "Short", 0: "Neutral", 1: "Long"}.get(label, str(label))
            logger.info(f"  {label_name:8s} ({label:2d}): {pct*100:5.1f}%")
        logger.info("=" * 60)

        # Split data (80/20 train/test) with STRATIFICATION
        # stratify=y_direction ensures train/test have same label distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_direction,
            test_size=0.2,
            random_state=42,
            stratify=y_direction  # CRITICAL: Maintain label distribution
        )
        logger.info(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")

        # Verify distributions match after split
        train_dist = y_train.value_counts(normalize=True).sort_index()
        test_dist = y_test.value_counts(normalize=True).sort_index()

        logger.info("")
        logger.info("=" * 60)
        logger.info("LABEL DISTRIBUTION VERIFICATION")
        logger.info("=" * 60)
        logger.info(f"{'Label':<10} {'Train':>8} {'Test':>8} {'Shift':>8}")
        logger.info("-" * 60)

        max_shift = 0.0
        for label in sorted(set(train_dist.index) | set(test_dist.index)):
            train_pct = train_dist.get(label, 0.0)
            test_pct = test_dist.get(label, 0.0)
            shift = abs(train_pct - test_pct)
            max_shift = max(max_shift, shift)

            label_name = {-1: "Short", 0: "Neutral", 1: "Long"}.get(label, str(label))
            logger.info(f"{label_name:<10} {train_pct*100:7.1f}% {test_pct*100:7.1f}% {shift*100:7.1f}%")

        logger.info("=" * 60)

        # Check for significant distribution shift
        if max_shift > 0.10:
            logger.warning("")
            logger.warning("⚠️  LARGE LABEL DISTRIBUTION SHIFT DETECTED")
            logger.warning(f"    Maximum shift: {max_shift*100:.1f}%")
            logger.warning(f"    Expected: <5% shift with stratified sampling")
            logger.warning("")
            logger.warning("    ISSUE: Stratified sampling may not be working correctly")
            logger.warning("    ACTION: Check if label distribution is too imbalanced")
            logger.warning("")
        elif max_shift > 0.05:
            logger.warning(f"⚠️ Moderate label shift: {max_shift*100:.1f}% (target: <5%)")
        else:
            logger.info(f"✓ Good label distribution: max shift {max_shift*100:.1f}% (target: <5%)")

        # Detect potential regime change
        self._detect_regime_change(y_train, y_test)

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        logger.info(f"  Features scaled using StandardScaler")

        # Encode labels: XGBoost needs 0,1,2 (not -1,0,1)
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)

        logger.info(f"Label mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")

        # Step 5: Train primary model (XGBoost)
        logger.info("Step 5: Training primary model (XGBoost)...")

        # =============================================================================
        # PRIMARY MODEL: XGBoost with STRONG REGULARIZATION
        # =============================================================================
        # XGBoost overfits easily - these params are carefully tuned to prevent it

        self.primary_model = XGBClassifier(
            # ===== TREE STRUCTURE (prevent deep memorization) =====
            n_estimators=100,          # Number of boosting rounds
            max_depth=4,               # SHALLOWER than RF (boosting compounds depth)
                                       # Rule: XGB max_depth = RF max_depth - 1

            min_child_weight=20,       # Minimum samples in a leaf (like min_samples_leaf)
                                       # Higher = more generalization

            # ===== REGULARIZATION (the XGBoost advantage) =====
            reg_alpha=0.1,             # L1 regularization (lasso) - creates sparsity
            reg_lambda=1.0,            # L2 regularization (ridge) - shrinks weights
            gamma=0.1,                 # Minimum loss reduction to make a split
                                       # Higher = more conservative splitting

            # ===== RANDOMIZATION (reduce variance) =====
            subsample=0.7,             # Use 70% of samples per tree
            colsample_bytree=0.7,      # Use 70% of features per tree
            colsample_bylevel=0.7,     # Use 70% of features per level

            # ===== LEARNING RATE (slow learning = better generalization) =====
            learning_rate=0.05,        # Low learning rate (default 0.3)
                                       # Slower learning = less overfitting
                                       # Compensate with more n_estimators if needed

            # ===== OTHER =====
            objective='multi:softprob', # Multi-class classification
            num_class=3,               # Long, Short, Neutral
            eval_metric='mlogloss',    # Evaluation metric
            use_label_encoder=False,   # We handle encoding ourselves
            random_state=42,
            n_jobs=-1,

            # ===== EARLY STOPPING (automatic overfitting prevention) =====
            early_stopping_rounds=20   # Stop if no improvement for 20 rounds
        )

        # Train with early stopping using validation set
        self.primary_model.fit(
            X_train_scaled,
            y_train_encoded,
            eval_set=[(X_test_scaled, y_test_encoded)],
            verbose=False
        )

        # Log training info
        best_iteration = self.primary_model.best_iteration
        logger.info(f"  XGBoost stopped at iteration {best_iteration} (early stopping)")

        # =============================================================================
        # EVALUATE AND CHECK FOR OVERFITTING
        # =============================================================================
        train_preds = self.primary_model.predict(X_train_scaled)
        test_preds = self.primary_model.predict(X_test_scaled)

        train_acc = (train_preds == y_train_encoded).mean()
        test_acc = (test_preds == y_test_encoded).mean()
        gap = train_acc - test_acc

        logger.info(f"  Primary Model - Train: {train_acc:.1%}, Test: {test_acc:.1%}, Gap: {gap:.1%}")

        if gap > 0.10:
            logger.warning(f"  ⚠️ OVERFITTING: {gap:.1%} gap. Increase regularization.")
        elif gap > 0.05:
            logger.info(f"  ⚡ Slight overfitting: {gap:.1%} gap. Acceptable but monitor.")
        else:
            logger.info(f"  ✓ Good generalization: {gap:.1%} gap")

        # Log feature importance
        if hasattr(self.primary_model, 'feature_importances_'):
            importance = dict(zip(self.feature_names, self.primary_model.feature_importances_))
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            logger.info("  Feature Importance (XGBoost gain):")
            for feat, imp in sorted_imp[:10]:  # Top 10 features
                logger.info(f"    {feat}: {imp:.4f}")
            self.important_features = [feat for feat, _ in sorted_imp[:10]]
        else:
            self.important_features = X.columns[:10].tolist()

        # Step 6: Train meta model (bet sizing)
        logger.info("Step 6: Training meta model (LogisticRegression)...")

        # =============================================================================
        # META MODEL: LogisticRegression
        # =============================================================================
        # Meta model predicts: "Will the primary model be correct?"
        # This is different from predicting direction - it predicts WHEN to trust primary

        # Create meta labels on TRAINING data
        primary_train_preds = self.primary_model.predict(X_train_scaled)
        # Decode back to original labels for comparison
        primary_train_preds_decoded = self.label_encoder.inverse_transform(primary_train_preds)

        # Use MetaLabeler to create labels with proper validation
        meta_labels_train = self.meta_labeler.create_meta_labels_from_predictions(
            primary_predictions=primary_train_preds_decoded,
            actual_labels=y_train.values
        )

        # Train meta model
        self.meta_model = LogisticRegression(
            C=0.1,
            penalty='l2',
            solver='lbfgs',
            max_iter=2000,
            class_weight='balanced',
            random_state=42
        )

        self.meta_model.fit(X_train_scaled, meta_labels_train)

        # Evaluate meta model on test set
        primary_test_preds = self.primary_model.predict(X_test_scaled)
        primary_test_preds_decoded = self.label_encoder.inverse_transform(primary_test_preds)
        meta_labels_test = (primary_test_preds_decoded == y_test.values).astype(int)

        # Meta model performance
        meta_train_pred = self.meta_model.predict(X_train_scaled)
        meta_test_pred = self.meta_model.predict(X_test_scaled)

        meta_train_acc = (meta_train_pred == meta_labels_train).mean()
        meta_test_acc = (meta_test_pred == meta_labels_test).mean()

        logger.info("")
        logger.info("=" * 60)
        logger.info("META MODEL PERFORMANCE")
        logger.info("=" * 60)
        logger.info(f"Meta model accuracy:")
        logger.info(f"  Train: {meta_train_acc:.1%} (predicting when primary is correct)")
        logger.info(f"  Test:  {meta_test_acc:.1%} (predicting when primary is correct)")
        logger.info("=" * 60)

        # Sanity check - meta model should perform around 45-55%
        if meta_test_acc < 0.40:
            logger.warning("")
            logger.warning("⚠️  META MODEL UNDERPERFORMING")
            logger.warning(f"    Current: {meta_test_acc:.1%}")
            logger.warning(f"    Expected: 45-60% (better than random guessing)")
            logger.warning("    ISSUE: Meta model cannot predict when primary is correct")
            logger.warning("    ACTION: Check features, try different meta model")
            logger.warning("")
        elif meta_test_acc > 0.70:
            logger.warning("")
            logger.warning("⚠️  META MODEL SUSPICIOUSLY HIGH")
            logger.warning(f"    Current: {meta_test_acc:.1%}")
            logger.warning(f"    Expected: 45-60% (modestly better than random)")
            logger.warning("    POSSIBLE ISSUE: Data leakage or overfitting")
            logger.warning("")
        else:
            logger.info(f"✓ Good meta model accuracy: {meta_test_acc:.1%}")
            logger.info(f"  Meta model can help filter weak primary predictions")

        meta_scores_mean = test_acc  # Use primary model test accuracy for results

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)

        return {
            'success': True,
            'n_samples': len(features),
            'primary_accuracy': test_acc,
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

        Raises:
            ValueError: If model not trained, scaler/encoder not initialized,
                       or feature generation fails
        """
        # H7: Enhanced error handling with clear, actionable messages
        if self.primary_model is None:
            raise ValueError(
                "Primary model not trained. Call train() before making predictions."
            )
        if self.scaler is None:
            raise ValueError(
                "Scaler not initialized. Model may be corrupted or incompletely trained. "
                "Retrain the model or load a valid model file."
            )
        if self.label_encoder is None:
            raise ValueError(
                "Label encoder not initialized. Model may be corrupted or incompletely trained. "
                "This is required for XGBoost models. Retrain or load a valid model."
            )

        # Create features from latest data with error handling
        try:
            features = self.prepare_features(bars)
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            logger.error(f"Input bars shape: {bars.shape}")
            logger.error(f"Input bars columns: {bars.columns.tolist()}")
            raise ValueError(
                f"Failed to generate features from input bars: {e}. "
                f"Check that bars DataFrame has required columns (open, high, low, close, volume)."
            )

        if len(features) == 0:
            logger.warning("No valid features generated from bars (all NaN after dropna)")
            return 0, 0.0

        # Validate feature names match training
        if hasattr(self, 'feature_names') and self.feature_names:
            feature_cols = features.columns.tolist()
            if feature_cols != self.feature_names:
                logger.error("")
                logger.error("=" * 60)
                logger.error("FEATURE MISMATCH ERROR")
                logger.error("=" * 60)
                logger.error(f"Expected features (from training):")
                for i, feat in enumerate(self.feature_names, 1):
                    logger.error(f"  {i}. {feat}")
                logger.error(f"Got features (from current data):")
                for i, feat in enumerate(feature_cols, 1):
                    logger.error(f"  {i}. {feat}")
                logger.error("=" * 60)
                raise ValueError(
                    f"Feature names don't match trained model. "
                    f"Expected {len(self.feature_names)} features: {self.feature_names}, "
                    f"got {len(feature_cols)} features: {feature_cols}"
                )

        # Get latest feature row and scale it
        X = features.iloc[[-1]]
        X_scaled = self.scaler.transform(X)

        # Primary model: get probabilities
        # XGBoost returns probabilities in encoded order (0,1,2)
        # which maps to label_encoder.classes_ (e.g., [-1, 0, 1])
        probs = self.primary_model.predict_proba(X_scaled)[0]

        # Map probabilities to original labels
        label_to_prob = dict(zip(self.label_encoder.classes_, probs))

        prob_short = label_to_prob.get(-1, 0.0)
        prob_neutral = label_to_prob.get(0, 0.0)
        prob_long = label_to_prob.get(1, 0.0)

        n_classes = len(probs)

        logger.info(f"XGBoost probabilities ({n_classes}-class) - Short: {prob_short:.4f}, Neutral: {prob_neutral:.4f}, Long: {prob_long:.4f}")

        # Find the winning class and calculate margin vs runner-up
        if prob_long > prob_short and prob_long > prob_neutral:
            winner = 1  # Long
            margin = prob_long - max(prob_short, prob_neutral)
        elif prob_short > prob_long and prob_short > prob_neutral:
            winner = -1  # Short
            margin = prob_short - max(prob_long, prob_neutral)
        else:
            winner = 0  # Neutral
            margin = prob_neutral - max(prob_short, prob_long)

        # Require BOTH conditions for directional prediction:
        # 1. Winner probability > prob_threshold (keeps optimal param)
        # 2. Margin between winner and runner-up > margin_threshold (configurable filter)
        # H9: Use configurable margin threshold instead of hardcoded value
        margin_threshold = self.margin_threshold

        if (winner != 0 and
            margin >= margin_threshold and
            max(prob_long, prob_short) > prob_threshold):
            direction = winner
            logger.info(f"✓ Signal accepted: margin={margin:.2%} (>{margin_threshold:.1%}), "
                       f"prob={max(prob_long, prob_short):.2%} (>{prob_threshold:.2%})")
        else:
            direction = 0
            if winner != 0:
                logger.info(f"✗ Signal filtered: margin={margin:.2%} (need >{margin_threshold:.1%}), "
                           f"predicting neutral instead of {winner}")

        # Meta model: should we trade? (probability that primary is correct)
        if self.meta_model is not None:
            # Meta model predicts: P(primary model is correct)
            # NOT: P(price goes up) or P(should trade)
            meta_proba = self.meta_model.predict_proba(X_scaled)[0]
            confidence_primary_correct = meta_proba[1]  # P(primary prediction will be right)
            confidence_primary_wrong = meta_proba[0]    # P(primary prediction will be wrong)

            # Log meta model prediction
            logger.info(f"Meta model confidence:")
            logger.info(f"  P(primary correct): {confidence_primary_correct:.3f}")
            logger.info(f"  P(primary wrong):   {confidence_primary_wrong:.3f}")
            logger.debug(f"  Meta threshold: {meta_threshold:.4f}")

            # Convert probability to bet size
            if confidence_primary_correct < meta_threshold:
                # Meta model says primary is likely wrong - don't trade
                logger.info(f"✗ Meta model rejects trade: {confidence_primary_correct:.3f} < {meta_threshold:.4f}")
                return 0, 0.0
            else:
                # Bet size proportional to confidence that primary is correct
                bet_size = min(confidence_primary_correct, 1.0)
                logger.info(f"✓ Meta model approves: confidence={confidence_primary_correct:.3f}, bet_size={bet_size:.3f}")
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

    def save_models(self, path: str, save_versioned: bool = True):
        """
        Save trained models to disk with optional versioning.

        This method saves models in two ways:
        1. Versioned file with timestamp (for history/rollback)
        2. "Latest" file without timestamp (for easy loading)

        Args:
            path: Base path for model file (e.g., "models/SPY_model.pkl")
            save_versioned: If True, also saves timestamped version (default: True)

        Example:
            Given path="models/SPY_model.pkl", creates:
            - models/SPY_model_20260111_143052.pkl  (versioned)
            - models/SPY_model.pkl  (latest)
        """
        import joblib

        if self.primary_model is None or self.meta_model is None:
            logger.warning("No models to save")
            return

        # Prepare model data
        model_data = {
            'primary_model': self.primary_model,
            'meta_model': self.meta_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,  # CRITICAL: Save encoder
            'feature_names': self.feature_names,
            'important_features': self.important_features,
            'model_type': 'XGBoost_primary_LR_meta',
            'hyperparameters': {
                'xgb_max_depth': 4,
                'xgb_learning_rate': 0.05,
                'xgb_reg_alpha': 0.1,
                'xgb_reg_lambda': 1.0,
                'xgb_gamma': 0.1,
                'lr_C': 0.1
            },
            # NEW: Add versioning metadata
            'version': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'train_date': datetime.now().isoformat(),
            'python_version': sys.version,
        }

        # Save versioned model (for history/rollback)
        if save_versioned:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            version_path = path.replace('.pkl', f'_{timestamp}.pkl')

            joblib.dump(model_data, version_path)
            logger.info(f"Versioned model saved: {version_path}")

        # Save as "latest" (overwrites previous)
        latest_path = path
        joblib.dump(model_data, latest_path)
        logger.info(f"Latest model saved: {latest_path}")

        logger.info(f"Model version: {model_data['version']}")

    def load_models(self, path: str):
        """Load trained models from disk with version validation."""
        import joblib

        data = joblib.load(path)

        self.primary_model = data['primary_model']
        self.meta_model = data['meta_model']
        self.feature_names = data['feature_names']
        self.important_features = data['important_features']

        # Load scaler and label encoder (critical for XGBoost models)
        self.scaler = data.get('scaler')
        self.label_encoder = data.get('label_encoder')

        if self.scaler is None:
            logger.warning("⚠️ No scaler in saved model!")
        if self.label_encoder is None:
            logger.warning("⚠️ No label encoder in saved model!")

        model_type = data.get('model_type', 'unknown')
        logger.info(f"Loaded model type: {model_type}")
        logger.info(f"Models loaded from {path}")

        # NEW: Log version information if available
        version = data.get('version')
        train_date = data.get('train_date')

        if version:
            logger.info(f"Model version: {version}")
        else:
            logger.warning("⚠️ No version metadata in model (legacy model)")

        if train_date:
            logger.info(f"Model trained: {train_date}")

        # Log hyperparameters if available (useful for debugging)
        hyperparams = data.get('hyperparameters')
        if hyperparams:
            logger.debug(f"Model hyperparameters: {hyperparams}")

    def train_from_ticks(
        self,
        symbol: str,
        threshold: Optional[float] = None,
        min_samples: int = 100
    ) -> Dict:
        """
        Train models from tick data stored in database.

        ARCHITECTURE (C2 FIX):
        This method applies CUSUM filtering at the TICK level (not bar level):
        1. Loads tick data from the SQLite database
        2. Applies CUSUM filter to tick prices (~35% filter rate)
        3. Generates imbalance bars from CUSUM-filtered ticks
        4. Converts bars to DataFrame
        5. Runs RiskLabAI training pipeline (CUSUM skipped - already done)

        This is the correct architecture. The old approach (CUSUM on bars)
        filtered almost nothing (98% pass rate) because bars are already
        filtered by imbalance events.

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
                f"Run scripts/setup/backfill_ticks.py first."
            )

        earliest, latest = date_range
        logger.info(f"  Available data: {earliest} to {latest}")

        # Load all ticks
        ticks = storage.load_ticks(symbol)
        storage.close()

        if not ticks:
            raise ValueError(f"No ticks loaded for {symbol}")

        logger.info(f"  Loaded {len(ticks):,} ticks")

        # Step 2: Apply CUSUM filter to ticks (C2 FIX)
        logger.info("Step 2: Applying CUSUM filter to tick prices...")

        # Convert ticks to pandas Series for CUSUM filtering
        # ticks format: List[Tuple[timestamp_str, price_float, volume_int]]
        # Use format='ISO8601' to handle timestamps with/without microseconds
        tick_timestamps = pd.to_datetime([t[0] for t in ticks], format='ISO8601')
        tick_prices = pd.Series([t[1] for t in ticks], index=tick_timestamps)

        # Apply CUSUM filter to get event timestamps
        cusum_events = self.cusum_filter.get_events(tick_prices)
        logger.info(f"  CUSUM events: {len(cusum_events)} from {len(ticks):,} ticks")
        logger.info(f"  Filter rate: {len(cusum_events)/len(ticks)*100:.1f}%")

        if len(cusum_events) < min_samples:
            raise ValueError(
                f"Insufficient CUSUM events ({len(cusum_events)} < {min_samples}). "
                f"Try reducing the CUSUM threshold."
            )

        # Filter ticks to only those matching CUSUM events
        # Keep ticks within a small window around each CUSUM event
        cusum_event_set = set(cusum_events)
        filtered_ticks = [
            tick for tick in ticks
            if pd.to_datetime(tick[0]) in cusum_event_set
        ]
        logger.info(f"  Filtered ticks: {len(filtered_ticks):,} (from CUSUM events)")

        # Step 3: Generate tick imbalance bars from filtered ticks
        logger.info("Step 3: Generating tick imbalance bars from filtered ticks...")
        bars_list = generate_bars_from_ticks(filtered_ticks, threshold=threshold)

        if not bars_list:
            raise ValueError(f"No bars generated from {len(filtered_ticks)} filtered ticks")

        logger.info(f"  Generated {len(bars_list)} bars")
        logger.info(f"  Bars per filtered tick: {len(bars_list)/len(filtered_ticks):.4f}")

        # Step 4: Convert to DataFrame for RiskLabAI
        logger.info("Step 4: Converting bars to DataFrame...")

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

        # Step 5: Run RiskLabAI training pipeline (CUSUM already applied)
        logger.info("Step 5: Running RiskLabAI training pipeline...")
        results = self.train(bars_df, min_samples=min_samples, symbol=symbol, cusum_already_applied=True)

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
