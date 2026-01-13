"""
RiskLabAI-Based Trading Strategy

This is the main strategy class that orchestrates all RiskLabAI components:
1. Load raw ticks from storage
2. Apply CUSUM filter on ticks for event sampling
3. Generate information-driven bars from filtered ticks
4. Apply fractional differentiation for stationary features
5. Label with triple-barrier method
6. Train primary + meta models with purged CV validation
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
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterSampler
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Import RiskLabAI components
from risklabai.data_structures.bars import BarGenerator
from risklabai.labeling.triple_barrier import TripleBarrierLabeler
from risklabai.labeling.meta_labeling import MetaLabeler
from risklabai.features.fractional_diff import FractionalDifferentiator
from risklabai.sampling.cusum_filter import CUSUMEventFilter
from risklabai.cross_validation.purged_kfold import PurgedCrossValidator
from risklabai.portfolio.hrp import HRPPortfolio

# Import tick data components
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from config.tick_config import TICK_DB_PATH, INITIAL_IMBALANCE_THRESHOLD, CUSUM_EVENT_WINDOW_SECONDS
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
    │ Raw Tick Data                                            │
    │         ↓                                                │
    │ CUSUM Filter (Tick-Level Event Sampling)                 │
    │         ↓                                                │
    │ Tick Imbalance Bars                                      │
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
        max_holding: int = 20,
        d: float = None,
        n_cv_splits: int = 5,
        margin_threshold: float = 0.03,
        tune_primary: bool = False,
        tune_meta: bool = False,
        primary_gap_tolerance: float = 0.05,
        primary_gap_penalty: float = 0.75,
        tune_primary_trials: int = 25,
        tune_primary_seed: int = 42,
        primary_param_space: Optional[Dict[str, List]] = None,
        meta_c_candidates: Optional[List[float]] = None,
        primary_search_depth: str = "standard",
        meta_search_depth: str = "standard",
        meta_l1_ratios: Optional[List[float]] = None,
        split_method: str = "stratified",
        split_test_size: float = 0.2,
        walk_forward_splits: int = 5,
        walk_forward_train_size: Optional[float] = None,
        walk_forward_test_size: Optional[float] = None,
        walk_forward_step_size: Optional[float] = None,
        walk_forward_expanding: bool = True,
        primary_params_override: Optional[Dict] = None,
        meta_params_override: Optional[Dict] = None
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
            tune_primary: Whether to tune primary model params (default: False)
            tune_meta: Whether to tune meta model params (default: False)
            primary_gap_tolerance: Allowed train/test gap before penalty (default: 0.05)
            primary_gap_penalty: Penalty applied to gap beyond tolerance (default: 0.75)
            tune_primary_trials: Number of primary param samples (default: 25)
            tune_primary_seed: Random seed for primary param sampling
            primary_param_space: Optional param search space override
            meta_c_candidates: Optional list of C values for meta tuning
            primary_search_depth: Search depth for primary params (standard|deep)
            meta_search_depth: Search depth for meta params (standard|deep)
            meta_l1_ratios: Optional list of l1_ratio values for elasticnet meta tuning
            split_method: Train/test split method (stratified|time|walk_forward)
            split_test_size: Fraction of data reserved for test
            walk_forward_splits: Number of walk-forward folds (default: 5)
            walk_forward_train_size: Train window size for walk-forward (int or fraction)
            walk_forward_test_size: Test window size for walk-forward (int or fraction)
            walk_forward_step_size: Step size between folds (int or fraction)
            walk_forward_expanding: Use expanding train window if True
            primary_params_override: Optional fixed primary params to use
            meta_params_override: Optional fixed meta params to use
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
        self.hrp = HRPPortfolio()

        # Models (initialized during training)
        self.primary_model = None
        self.meta_model = None
        self.primary_params = {}

        # Preprocessing (initialized during training)
        self.scaler = None
        self.label_encoder = None  # XGBoost needs 0,1,2 not -1,0,1

        # Feature parameters
        self.feature_names = None
        self.important_features = None
        self.feature_importance = None  # Detailed importance scores

        # Prediction parameters
        self.margin_threshold = margin_threshold  # Configurable margin threshold (H9)
        self.tune_primary = tune_primary
        self.tune_meta = tune_meta
        self.primary_gap_tolerance = primary_gap_tolerance
        self.primary_gap_penalty = primary_gap_penalty
        self.tune_primary_trials = tune_primary_trials
        self.tune_primary_seed = tune_primary_seed
        self.primary_param_space = primary_param_space
        self.meta_c_candidates = meta_c_candidates
        self.primary_search_depth = primary_search_depth
        self.meta_search_depth = meta_search_depth
        self.meta_l1_ratios = meta_l1_ratios
        self.split_method = split_method
        self.split_test_size = split_test_size
        self.walk_forward_splits = walk_forward_splits
        self.walk_forward_train_size = walk_forward_train_size
        self.walk_forward_test_size = walk_forward_test_size
        self.walk_forward_step_size = walk_forward_step_size
        self.walk_forward_expanding = walk_forward_expanding
        self.primary_params_override = primary_params_override
        self.meta_params_override = meta_params_override

        if self.primary_search_depth not in {"standard", "deep"}:
            logger.warning(
                f"Unknown primary_search_depth '{self.primary_search_depth}', defaulting to 'standard'"
            )
            self.primary_search_depth = "standard"

        if self.meta_search_depth not in {"standard", "deep"}:
            logger.warning(
                f"Unknown meta_search_depth '{self.meta_search_depth}', defaulting to 'standard'"
            )
            self.meta_search_depth = "standard"

        if self.split_method not in {"stratified", "time", "walk_forward"}:
            logger.warning(
                f"Unknown split_method '{self.split_method}', defaulting to 'stratified'"
            )
            self.split_method = "stratified"

        if self.walk_forward_splits is not None and self.walk_forward_splits <= 0:
            logger.warning(
                f"Invalid walk_forward_splits '{self.walk_forward_splits}', defaulting to 5"
            )
            self.walk_forward_splits = 5

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
            if self.frac_diff.d is None and self.frac_diff._optimal_d is not None:
                self.frac_diff.d = self.frac_diff._optimal_d
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
            logger.info(f"Samples/feature: {samples_per_feature:.1f} (target: 20-40)")
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

    def _filter_ticks_by_event_window(
        self,
        ticks: List[Tuple[str, float, int]],
        tick_timestamps: pd.DatetimeIndex,
        cusum_events: pd.DatetimeIndex,
        window_seconds: int
    ) -> List[Tuple[str, float, int]]:
        """
        Expand CUSUM events to include ticks within a time window.

        This increases bar counts when tick volumes are low.
        """
        if len(cusum_events) == 0:
            return []

        event_values = cusum_events.values
        tick_values = tick_timestamps.values
        max_delta = np.timedelta64(window_seconds, 's')

        idx = np.searchsorted(event_values, tick_values)

        large_delta = np.timedelta64(10**9, 's')
        prev_delta = np.full(len(tick_values), large_delta)
        next_delta = np.full(len(tick_values), large_delta)

        has_prev = idx > 0
        has_next = idx < len(event_values)

        prev_delta[has_prev] = tick_values[has_prev] - event_values[idx[has_prev] - 1]
        next_delta[has_next] = event_values[idx[has_next]] - tick_values[has_next]

        min_delta = np.minimum(prev_delta, next_delta)
        keep_mask = min_delta <= max_delta

        return [tick for tick, keep in zip(ticks, keep_mask) if keep]

    def _build_primary_model(
        self,
        use_early_stopping: bool = True,
        params_override: Optional[Dict] = None
    ) -> XGBClassifier:
        """
        Build a configured primary XGBoost model.

        Args:
            use_early_stopping: Whether to enable early stopping.

        Returns:
            Configured XGBClassifier instance.
        """
        params = {
            'n_estimators': 200,
            'max_depth': 3,
            'min_child_weight': 30,
            'reg_alpha': 0.3,
            'reg_lambda': 2.0,
            'gamma': 0.2,
            'subsample': 0.6,
            'colsample_bytree': 0.6,
            'colsample_bylevel': 0.6,
            'learning_rate': 0.04,
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'n_jobs': -1
        }

        if params_override:
            params.update(params_override)

        if use_early_stopping:
            params.setdefault('early_stopping_rounds', 15)
        else:
            params.pop('early_stopping_rounds', None)

        return XGBClassifier(**params)

    def _oof_primary_predictions(
        self,
        X: pd.DataFrame,
        y_encoded: pd.Series,
        samples_info: pd.Series,
        expected_classes: Optional[set] = None
    ) -> pd.Series:
        """
        Generate out-of-fold primary predictions using purged time blocks.
        """
        def _is_zero_based_contiguous(class_set: set) -> bool:
            if not class_set:
                return False
            class_list = sorted(class_set)
            return class_list[0] == 0 and class_list[-1] == len(class_list) - 1

        oof_preds = pd.Series(index=X.index, dtype="float64")
        total = len(X)
        covered = 0

        if expected_classes is None:
            expected_classes = set(np.unique(y_encoded))

        for fold, (train_idx, test_idx) in enumerate(
            self.cv.iter_time_block_splits(X, y_encoded, samples_info), 1
        ):
            y_train_fold = y_encoded.iloc[train_idx]
            y_test_fold = y_encoded.iloc[test_idx]

            train_classes = set(np.unique(y_train_fold))
            test_classes = set(np.unique(y_test_fold))
            if train_classes != expected_classes or test_classes != expected_classes:
                logger.info(
                    f"OOF fold {fold}/{self.cv.n_splits}: skipped (classes train={sorted(train_classes)} "
                    f"test={sorted(test_classes)})"
                )
                continue
            if (not _is_zero_based_contiguous(train_classes) or
                    not _is_zero_based_contiguous(test_classes)):
                logger.info(
                    f"OOF fold {fold}/{self.cv.n_splits}: skipped (non-contiguous classes train={sorted(train_classes)} "
                    f"test={sorted(test_classes)})"
                )
                continue

            scaler = StandardScaler()
            X_train_fold = scaler.fit_transform(X.iloc[train_idx])
            X_test_fold = scaler.transform(X.iloc[test_idx])

            classes = np.unique(y_train_fold)
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=y_train_fold
            )
            weight_map = dict(zip(classes, class_weights))
            train_weights = np.array([weight_map[c] for c in y_train_fold])

            model = self._build_primary_model(
                use_early_stopping=False,
                params_override=self.primary_params
            )
            model.fit(X_train_fold, y_train_fold, sample_weight=train_weights, verbose=False)
            preds = model.predict(X_test_fold)

            oof_preds.iloc[test_idx] = preds
            covered += len(test_idx)

        logger.info(f"OOF primary coverage: {covered}/{total} ({covered/total:.1%})")
        return oof_preds

    def train(
        self,
        bars: pd.DataFrame,
        min_samples: int = 100,
        symbol: Optional[str] = None
    ) -> Dict:
        """
        Train primary and meta models.

        Args:
            bars: OHLCV bar data
            min_samples: Minimum samples required
            symbol: Symbol name (for logging)
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

        # Step 1: Event sampling is ONLY applied at the tick level.
        # Never apply CUSUM to bars in this pipeline.
        logger.info("Step 1: Using all bars as events (no bar-level CUSUM)")
        logger.info(f"  Using all {len(bars)} bars")
        logger.info("Step 2: Creating fractional-differentiated features...")
        features = self.prepare_features(bars, symbol=symbol)
        logger.info(f"  Feature matrix shape: {features.shape}")
        events = features.index

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
        event_df = event_df.loc[event_df.index.isin(labels.index)]

        # Build samples_info for purged CV / OOF meta-labeling
        samples_info = None
        try:
            if 'End Time' in labels.columns:
                samples_info = labels['End Time']
            else:
                from RiskLabAI.data.labeling import vertical_barrier
                samples_info = vertical_barrier(
                    close=bars['close'],
                    time_events=labels.index,
                    number_days=self.labeler.max_holding_period
                )
            samples_info = samples_info.reindex(features.index)
            valid_mask = samples_info.notna()
            if valid_mask.sum() < len(features):
                logger.info(f"Purged-CV samples with valid t1: {valid_mask.sum()}/{len(features)}")
            features = features.loc[valid_mask]
            labels = labels.loc[valid_mask]
            event_df = event_df.loc[valid_mask]
            samples_info = samples_info.loc[valid_mask]
        except Exception as e:
            logger.warning(f"Failed to build samples_info for purged CV: {e}")
            samples_info = None

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

        # Fit label encoder on full label set (stable mapping)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y_direction)

        # Split data (default: stratified 80/20)
        if self.split_method == "time":
            X_sorted = X.sort_index()
            y_sorted = y_direction.loc[X_sorted.index]
            split_idx = int(len(X_sorted) * (1 - self.split_test_size))
            split_idx = max(1, min(split_idx, len(X_sorted) - 1))
            X_train = X_sorted.iloc[:split_idx]
            X_test = X_sorted.iloc[split_idx:]
            y_train = y_sorted.iloc[:split_idx]
            y_test = y_sorted.iloc[split_idx:]
            logger.info(
                f"  Split method: time (train={len(X_train)}, test={len(X_test)})"
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_direction,
                test_size=self.split_test_size,
                random_state=42,
                stratify=y_direction  # CRITICAL: Maintain label distribution
            )
            logger.info(
                f"  Split method: stratified (train={len(X_train)}, test={len(X_test)})"
            )

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
        if self.split_method == "time":
            logger.info("Note: time split can show larger label shifts by design")

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
        y_train_encoded = self.label_encoder.transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)

        logger.info(f"Label mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")

        # Step 5: Train primary model (XGBoost)
        logger.info("Step 5: Training primary model (XGBoost)...")

        # =============================================================================
        # PRIMARY MODEL: XGBoost with STRONG REGULARIZATION
        # =============================================================================
        # XGBoost overfits easily - these params are carefully tuned to prevent it

        # Train with early stopping using validation set
        # Class balancing for primary model
        classes = np.unique(y_train_encoded)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_encoded)
        weight_map = dict(zip(classes, class_weights))
        train_weights = np.array([weight_map[c] for c in y_train_encoded])
        test_weights = np.array([weight_map.get(c, 1.0) for c in y_test_encoded])

        fit_kwargs = {
            "X": X_train_scaled,
            "y": y_train_encoded,
            "eval_set": [(X_test_scaled, y_test_encoded)],
            "sample_weight": train_weights,
            "sample_weight_eval_set": [test_weights],
            "verbose": False
        }

        train_bal_acc = None
        test_bal_acc = None
        gap = None

        if self.primary_params_override:
            logger.info("  Using primary params override (skipping tuning)")
            self.primary_model = self._build_primary_model(
                use_early_stopping=True,
                params_override=dict(self.primary_params_override)
            )
            self.primary_params = dict(self.primary_params_override)
            self.primary_model.fit(**fit_kwargs)

            train_preds = self.primary_model.predict(X_train_scaled)
            test_preds = self.primary_model.predict(X_test_scaled)

            train_bal_acc = balanced_accuracy_score(y_train_encoded, train_preds)
            test_bal_acc = balanced_accuracy_score(y_test_encoded, test_preds)
            gap = train_bal_acc - test_bal_acc
        elif self.tune_primary:
            standard_space = {
                "max_depth": [2, 3, 4, 5],
                "min_child_weight": [10, 15, 20, 30, 40, 60],
                "reg_alpha": [0.0, 0.05, 0.1, 0.3, 0.6, 1.0],
                "reg_lambda": [0.8, 1.0, 1.5, 2.0, 3.0, 4.0],
                "gamma": [0.0, 0.05, 0.1, 0.2, 0.3, 0.4],
                "subsample": [0.5, 0.6, 0.7, 0.8],
                "colsample_bytree": [0.5, 0.6, 0.7, 0.8],
                "colsample_bylevel": [0.5, 0.6, 0.7, 0.8],
                "learning_rate": [0.02, 0.03, 0.04, 0.05, 0.07],
                "n_estimators": [150, 200, 250, 300, 400],
                "early_stopping_rounds": [10, 15, 20]
            }
            deep_space = {
                "max_depth": [1, 2, 3, 4, 5, 6],
                "min_child_weight": [1, 5, 10, 20, 30, 40, 60, 80, 100],
                "reg_alpha": [0.0, 0.01, 0.05, 0.1, 0.3, 0.6, 1.0, 2.0],
                "reg_lambda": [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0],
                "gamma": [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0],
                "subsample": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bylevel": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1],
                "n_estimators": [100, 150, 200, 250, 300, 400, 500, 800],
                "early_stopping_rounds": [10, 15, 20, 30]
            }
            param_space = deep_space if self.primary_search_depth == "deep" else standard_space
            if self.primary_param_space:
                param_space = self.primary_param_space
            sampled = list(
                ParameterSampler(
                    param_space,
                    n_iter=self.tune_primary_trials,
                    random_state=self.tune_primary_seed
                )
            )
            candidates = [("base", {})]
            for i, params_override in enumerate(sampled, 1):
                candidates.append((f"sample_{i}", params_override))

            best = None
            results = []
            logger.info("  Tuning primary model parameters...")
            logger.info(
                f"  Primary search depth: {self.primary_search_depth} "
                f"(trials={self.tune_primary_trials})"
            )
            if self.primary_param_space:
                logger.info("  Primary param space override active")
            for name, params_override in candidates:
                model = self._build_primary_model(
                    use_early_stopping=True,
                    params_override=params_override
                )
                model.fit(**fit_kwargs)

                train_preds = model.predict(X_train_scaled)
                test_preds = model.predict(X_test_scaled)

                cand_train_acc = balanced_accuracy_score(y_train_encoded, train_preds)
                cand_test_acc = balanced_accuracy_score(y_test_encoded, test_preds)
                cand_gap = cand_train_acc - cand_test_acc
                penalty = max(0.0, cand_gap - self.primary_gap_tolerance)
                score = cand_test_acc - self.primary_gap_penalty * penalty

                logger.info(
                    f"  Candidate {name}: Train {cand_train_acc:.1%}, "
                    f"Test {cand_test_acc:.1%}, Gap {cand_gap:.1%}, "
                    f"Score {score:.3f}"
                )

                results.append({
                    "name": name,
                    "params": params_override,
                    "train_acc": cand_train_acc,
                    "test_acc": cand_test_acc,
                    "gap": cand_gap,
                    "score": score
                })

                if best is None or score > best["score"]:
                    best = {
                        "name": name,
                        "model": model,
                        "params": params_override,
                        "train_acc": cand_train_acc,
                        "test_acc": cand_test_acc,
                        "gap": cand_gap,
                        "score": score
                    }

            self.primary_model = best["model"]
            self.primary_params = best["params"]
            train_bal_acc = best["train_acc"]
            test_bal_acc = best["test_acc"]
            gap = best["gap"]

            best_iteration = getattr(self.primary_model, "best_iteration", None)
            if best_iteration is not None:
                logger.info(f"  XGBoost stopped at iteration {best_iteration} (early stopping)")
            logger.info(f"  Selected primary params: {best['name']}")

            if results:
                top_candidates = sorted(results, key=lambda r: r["score"], reverse=True)[:5]
                logger.info("  Top primary candidates:")
                for cand in top_candidates:
                    logger.info(
                        f"    {cand['name']}: Test {cand['test_acc']:.1%}, "
                        f"Gap {cand['gap']:.1%}, Score {cand['score']:.3f}, "
                        f"Params {cand['params']}"
                    )
        else:
            self.primary_model = self._build_primary_model(use_early_stopping=True)
            self.primary_params = {}
            self.primary_model.fit(**fit_kwargs)

            # Log training info
            best_iteration = self.primary_model.best_iteration
            logger.info(f"  XGBoost stopped at iteration {best_iteration} (early stopping)")

            # =============================================================================
            # EVALUATE AND CHECK FOR OVERFITTING
            # =============================================================================
            train_preds = self.primary_model.predict(X_train_scaled)
            test_preds = self.primary_model.predict(X_test_scaled)

            train_bal_acc = balanced_accuracy_score(y_train_encoded, train_preds)
            test_bal_acc = balanced_accuracy_score(y_test_encoded, test_preds)
            gap = train_bal_acc - test_bal_acc

        logger.info(f"  Primary Model (Balanced Acc) - Train: {train_bal_acc:.1%}, Test: {test_bal_acc:.1%}, Gap: {gap:.1%}")

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

        # Create meta labels on TRAINING data (OOF to reduce leakage)
        meta_labels_train = None
        meta_train_index = X_train.index
        if samples_info is not None:
            X_train_sorted = X_train.sort_index()
            samples_info_train = samples_info.loc[X_train_sorted.index]
            y_train_encoded_series = pd.Series(y_train_encoded, index=X_train.index)
            y_train_encoded_series = y_train_encoded_series.loc[X_train_sorted.index]
            expected_classes = set(range(len(self.label_encoder.classes_)))
            oof_preds_encoded = None
            try:
                oof_preds_encoded = self._oof_primary_predictions(
                    X_train_sorted,
                    y_train_encoded_series,
                    samples_info_train,
                    expected_classes=expected_classes
                )
            except Exception as e:
                logger.warning(f"OOF primary predictions failed: {e}")

            if oof_preds_encoded is not None:
                oof_valid = oof_preds_encoded.notna()
                min_oof = max(50, int(0.5 * len(oof_preds_encoded)))
                if oof_valid.sum() >= min_oof:
                    logger.info("Creating meta labels from OOF primary predictions")
                    meta_train_index = oof_valid[oof_valid].index
                    oof_decoded = self.label_encoder.inverse_transform(
                        oof_preds_encoded[oof_valid].astype(int)
                    )
                    meta_labels_train = self.meta_labeler.create_meta_labels_from_predictions(
                        primary_predictions=oof_decoded,
                        actual_labels=y_train.loc[meta_train_index].values
                    )
                else:
                    logger.warning(
                        f"OOF predictions too sparse ({oof_valid.sum()}/{len(oof_preds_encoded)}); "
                        "falling back to in-sample meta labels"
                    )

        if meta_labels_train is None:
            logger.info("Creating meta labels from in-sample primary predictions")
            primary_train_preds = self.primary_model.predict(X_train_scaled)
            primary_train_preds_decoded = self.label_encoder.inverse_transform(primary_train_preds)
            meta_labels_train = self.meta_labeler.create_meta_labels_from_predictions(
                primary_predictions=primary_train_preds_decoded,
                actual_labels=y_train.values
            )
            meta_train_index = X_train.index

        X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train.index)
        X_train_meta = X_train_scaled_df.loc[meta_train_index].to_numpy()
        meta_labels_train = np.asarray(meta_labels_train)

        # Tune meta model regularization if requested
        best_meta_params = {"C": 0.1, "penalty": "l2", "solver": "lbfgs"}
        if self.meta_params_override:
            logger.info("  Using meta params override (skipping tuning)")
            best_meta_params = dict(self.meta_params_override)
        elif self.tune_meta and len(meta_labels_train) >= 100:
            logger.info("  Tuning meta model regularization...")
            if self.meta_search_depth == "deep":
                c_candidates = self.meta_c_candidates or [
                    0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0
                ]
            else:
                c_candidates = self.meta_c_candidates or [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

            meta_candidates = []
            if self.meta_search_depth == "deep":
                l1_ratios = self.meta_l1_ratios or [0.1, 0.5, 0.9]
                for c_val in c_candidates:
                    meta_candidates.append({"C": c_val, "penalty": "l2", "solver": "lbfgs"})
                    meta_candidates.append({"C": c_val, "penalty": "l1", "solver": "saga"})
                    for l1_ratio in l1_ratios:
                        meta_candidates.append({
                            "C": c_val,
                            "penalty": "elasticnet",
                            "solver": "saga",
                            "l1_ratio": l1_ratio
                        })
            else:
                for c_val in c_candidates:
                    meta_candidates.append({"C": c_val, "penalty": "l2", "solver": "lbfgs"})

            logger.info(
                f"  Meta search depth: {self.meta_search_depth} "
                f"({len(meta_candidates)} candidates)"
            )
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            best_score = None
            for cand in meta_candidates:
                fold_scores = []
                for train_idx, val_idx in skf.split(X_train_meta, meta_labels_train):
                    model_params = {
                        "C": cand["C"],
                        "penalty": cand["penalty"],
                        "solver": cand["solver"],
                        "max_iter": 4000,
                        "class_weight": "balanced",
                        "random_state": 42
                    }
                    if "l1_ratio" in cand:
                        model_params["l1_ratio"] = cand["l1_ratio"]

                    model = LogisticRegression(**model_params)
                    model.fit(X_train_meta[train_idx], meta_labels_train[train_idx])
                    preds = model.predict(X_train_meta[val_idx])
                    fold_scores.append(
                        balanced_accuracy_score(meta_labels_train[val_idx], preds)
                    )

                mean_score = float(np.mean(fold_scores))
                cand_desc = f"C={cand['C']:.3g}, penalty={cand['penalty']}, solver={cand['solver']}"
                if "l1_ratio" in cand:
                    cand_desc += f", l1_ratio={cand['l1_ratio']:.2f}"
                logger.info(f"  Meta {cand_desc}: cv_balanced_accuracy={mean_score:.3f}")
                if best_score is None or mean_score > best_score:
                    best_score = mean_score
                    best_meta_params = cand

            best_desc = (
                f"C={best_meta_params['C']:.3g}, penalty={best_meta_params['penalty']}, "
                f"solver={best_meta_params['solver']}"
            )
            if "l1_ratio" in best_meta_params:
                best_desc += f", l1_ratio={best_meta_params['l1_ratio']:.2f}"
            logger.info(f"  Selected meta params: {best_desc} (cv_balanced_accuracy={best_score:.3f})")

        # Train meta model
        meta_params = {
            "C": best_meta_params["C"],
            "penalty": best_meta_params["penalty"],
            "solver": best_meta_params["solver"],
            "max_iter": 4000,
            "class_weight": "balanced",
            "random_state": 42
        }
        if "l1_ratio" in best_meta_params:
            meta_params["l1_ratio"] = best_meta_params["l1_ratio"]

        self.meta_model = LogisticRegression(**meta_params)

        self.meta_model.fit(X_train_meta, meta_labels_train)

        # Evaluate meta model on test set
        primary_test_preds = self.primary_model.predict(X_test_scaled)
        primary_test_preds_decoded = self.label_encoder.inverse_transform(primary_test_preds)
        meta_labels_test = (primary_test_preds_decoded == y_test.values).astype(int)

        # Meta model performance
        meta_train_pred = self.meta_model.predict(X_train_meta)
        meta_test_pred = self.meta_model.predict(X_test_scaled)

        meta_train_acc = (meta_train_pred == meta_labels_train).mean()
        meta_test_acc = (meta_test_pred == meta_labels_test).mean()
        meta_train_bal_acc = balanced_accuracy_score(meta_labels_train, meta_train_pred)
        meta_test_bal_acc = balanced_accuracy_score(meta_labels_test, meta_test_pred)
        meta_test_baseline = max(meta_labels_test.mean(), 1 - meta_labels_test.mean())

        logger.info("")
        logger.info("=" * 60)
        logger.info("META MODEL PERFORMANCE")
        logger.info("=" * 60)
        logger.info(f"Meta model accuracy:")
        logger.info(f"  Train: {meta_train_acc:.1%} (balanced: {meta_train_bal_acc:.1%})")
        logger.info(f"  Test:  {meta_test_acc:.1%} (balanced: {meta_test_bal_acc:.1%}, baseline: {meta_test_baseline:.1%})")
        logger.info("=" * 60)

        # Sanity check - meta model should beat baseline and ~50% balanced accuracy
        if meta_test_acc < meta_test_baseline:
            logger.warning("")
            logger.warning("⚠️  META MODEL UNDERPERFORMING")
            logger.warning(f"    Current: {meta_test_acc:.1%} (baseline: {meta_test_baseline:.1%})")
            logger.warning("    ISSUE: Meta model worse than majority-class baseline")
            logger.warning("    ACTION: Improve features or retrain with stronger signal")
            logger.warning("")
        elif meta_test_bal_acc < 0.50:
            logger.warning("")
            logger.warning("⚠️  META MODEL BELOW RANDOM (BALANCED)")
            logger.warning(f"    Current balanced accuracy: {meta_test_bal_acc:.1%}")
            logger.warning("    ACTION: Check feature usefulness or meta-label quality")
            logger.warning("")
        elif meta_test_acc > 0.70 and meta_test_bal_acc > 0.70:
            logger.warning("")
            logger.warning("⚠️  META MODEL SUSPICIOUSLY HIGH")
            logger.warning(f"    Current: {meta_test_acc:.1%} (balanced: {meta_test_bal_acc:.1%})")
            logger.warning(f"    Expected: 45-60% (modestly better than random)")
            logger.warning("    POSSIBLE ISSUE: Data leakage or overfitting")
            logger.warning("")
        else:
            logger.info(f"✓ Good meta model accuracy: {meta_test_acc:.1%}")
            logger.info(f"  Meta model can help filter weak primary predictions")

        meta_scores_mean = meta_test_bal_acc

        # Step 7: Purged CV validation (tick-level event timestamps)
        logger.info("Step 7: Purged CV validation...")
        cv_mean = None
        cv_std = None
        try:
            if samples_info is None:
                raise ValueError("samples_info unavailable for purged CV")
            samples_info_cv = samples_info.loc[features.index]
            features_cv = features

            cv_model = self._build_primary_model(
                use_early_stopping=False,
                params_override=self.primary_params
            )
            cv_pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", cv_model)
            ])

            y_encoded_full = self.label_encoder.transform(y_direction.loc[features_cv.index])
            y_encoded_full = pd.Series(y_encoded_full, index=features_cv.index)

            cv_scores = self.cv.cross_val_score_purged_stratified(
                model=cv_pipeline,
                X=features_cv,
                y=y_encoded_full,
                samples_info=samples_info_cv,
                scoring='balanced_accuracy'
            )
            cv_mean = float(cv_scores.mean())
            cv_std = float(cv_scores.std())
            logger.info(f"Purged CV balanced accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
        except Exception as e:
            logger.warning(f"Purged CV failed: {e}")

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)

        return {
            'success': True,
            'n_samples': len(features),
            'primary_accuracy': test_bal_acc,
            'meta_accuracy': meta_scores_mean,
            'purged_cv_mean': cv_mean,
            'purged_cv_std': cv_std,
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
        primary_params = self.primary_model.get_params() if self.primary_model else {}
        meta_params = self.meta_model.get_params() if self.meta_model else {}
        model_data = {
            'primary_model': self.primary_model,
            'meta_model': self.meta_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,  # CRITICAL: Save encoder
            'feature_names': self.feature_names,
            'important_features': self.important_features,
            'frac_diff_d': self.frac_diff.d,
            'model_type': 'XGBoost_primary_LR_meta',
            'hyperparameters': {
                'xgb_max_depth': primary_params.get('max_depth'),
                'xgb_learning_rate': primary_params.get('learning_rate'),
                'xgb_reg_alpha': primary_params.get('reg_alpha'),
                'xgb_reg_lambda': primary_params.get('reg_lambda'),
                'xgb_gamma': primary_params.get('gamma'),
                'xgb_min_child_weight': primary_params.get('min_child_weight'),
                'xgb_subsample': primary_params.get('subsample'),
                'xgb_colsample_bytree': primary_params.get('colsample_bytree'),
                'xgb_colsample_bylevel': primary_params.get('colsample_bylevel'),
                'xgb_n_estimators': primary_params.get('n_estimators'),
                'lr_C': meta_params.get('C'),
                'lr_class_weight': meta_params.get('class_weight')
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
        frac_diff_d = data.get('frac_diff_d')
        if frac_diff_d is not None:
            self.frac_diff.d = frac_diff_d
            self.frac_diff._optimal_d = frac_diff_d

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
        min_samples: int = 100,
        event_window_seconds: Optional[int] = None
    ) -> Dict:
        """
        Train models from tick data stored in database.

        ARCHITECTURE (C2 FIX):
        This method applies CUSUM filtering at the TICK level (not bar level):
        1. Loads tick data from the SQLite database
        2. Applies CUSUM filter to tick prices (~35% filter rate)
        3. Generates imbalance bars from CUSUM-filtered ticks
        4. Converts bars to DataFrame
        5. Applies fractional differentiation (dynamic per symbol)
        6. Triple-barrier labeling
        7. Trains primary + meta models, then purged CV validation

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

        # Filter ticks to those within a window around CUSUM events
        if event_window_seconds is None:
            event_window_seconds = CUSUM_EVENT_WINDOW_SECONDS
        filtered_ticks = self._filter_ticks_by_event_window(
            ticks=ticks,
            tick_timestamps=tick_timestamps,
            cusum_events=cusum_events,
            window_seconds=event_window_seconds
        )
        logger.info(f"  Filtered ticks: {len(filtered_ticks):,} (window: ±{event_window_seconds}s)")

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
        results = self.train(bars_df, min_samples=min_samples, symbol=symbol)

        if results['success']:
            results['returns'] = bars_df['close'].pct_change().dropna()
            results['bars_count'] = len(bars_df)
            logger.info("=" * 80)
            logger.info(f"✓ TRAINING SUCCESSFUL FROM TICK DATA")
            logger.info("=" * 80)
            logger.info(f"  Tick data: {len(ticks):,} ticks")
            logger.info(f"  Bars generated: {len(bars_list)}")
            logger.info(f"  Samples used: {results['n_samples']}")
            logger.info(f"  Primary balanced accuracy: {results['primary_accuracy']:.3f}")
            logger.info(f"  Meta balanced accuracy: {results['meta_accuracy']:.3f}")
            logger.info("=" * 80)

        return results
