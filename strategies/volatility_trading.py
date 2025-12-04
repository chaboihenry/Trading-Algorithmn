"""
Volatility Regime Trading with Incremental Learning (M1 Optimized)
===================================================================
Uses vectorized GARCH models and XGBoost for M1-optimized volatility forecasting
import sqlite3
with incremental training support for daily model updates.

HYBRID FIX APPLIED:
1. Lower regime thresholds: ±15% -> ±10%
2. Class weights for imbalanced data
3. Lower confidence threshold: 0.60 -> 0.50
4. Extreme percentile signals for stable regime

ENHANCEMENT: Smart Feature Selection (further optimization)
"""

import pandas as pd
import numpy as np
from base_strategy import BaseStrategy
from incremental_trainer import IncrementalTrainer
from feature_selector import SmartFeatureSelector
import logging
import joblib

# M1-optimized imports
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Vectorized GARCH
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logging.warning("arch package not available - using simplified GARCH")

logger = logging.getLogger(__name__)


class VolatilityTradingStrategy:
    """Trade based on GARCH forecasts and XGBoost regime prediction with incremental learning"""

    # HYBRID FIX: Adjusted thresholds and parameters
    REGIME_THRESHOLD = 0.10  # Changed from 0.15 to 0.10 (more sensitive)
    CONFIDENCE_THRESHOLD = 0.50  # Changed from 0.60 to 0.50 (more signals)
    EXTREME_VOL_PERCENTILE_LOW = 15  # For stable regime signals
    EXTREME_VOL_PERCENTILE_HIGH = 85  # For stable regime signals

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db",
                 lookback_days: int = 90,
                 min_samples: int = 30,
                 use_feature_selection: bool = True,
                 n_features: int = 20,
                 api=None):
        """
        Initialize volatility strategy with optional feature selection

        Args:
            db_path: Path to database
            lookback_days: Days of historical data to use
            min_samples: Minimum samples required for training
            use_feature_selection: Enable smart feature selection (default True)
            n_features: Number of top features to keep (default 20, as we already have fewer features)
            api: Alpaca API object
        """
        self.db_path = db_path
        self.lookback_days = lookback_days
        self.min_samples = min_samples
        self.model = None
        self.scaler = StandardScaler()
        self.name = "VolatilityTradingStrategy"
        self.trainer = IncrementalTrainer(db_path=db_path)
        self.feature_cols = None
        self.api = api

        # Feature selection configuration
        self.use_feature_selection = use_feature_selection
        self.n_features = n_features
        self.feature_selector = None

        if use_feature_selection:
            logger.info(f"✨ Feature selection ENABLED (top {n_features} features)")
        else:
            logger.info(f"Feature selection DISABLED (using all features)")

        # Try to load existing model
        self._load_model()

    def _get_model_path(self) -> Path:
        """Get path to saved model"""
        models_dir = Path(__file__).parent / 'models'
        models_dir.mkdir(exist_ok=True)
        return models_dir / 'volatility_xgboost.joblib'

    def _get_scaler_path(self) -> Path:
        """Get path to saved scaler"""
        models_dir = Path(__file__).parent / 'models'
        return models_dir / 'volatility_scaler.joblib'

    def _get_features_path(self) -> Path:
        """Get path to saved feature columns"""
        models_dir = Path(__file__).parent / 'models'
        return models_dir / 'volatility_features.joblib'

    def _load_model(self) -> bool:
        """Load saved model, scaler, and feature columns"""
        model_path = self._get_model_path()
        scaler_path = self._get_scaler_path()
        features_path = self._get_features_path()

        if model_path.exists() and scaler_path.exists() and features_path.exists():
            try:
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.feature_cols = joblib.load(features_path)
                logger.info(f"Loaded existing volatility model from {model_path}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
                return False
        return False

    def _save_model(self) -> None:
        """Save model, scaler, and feature columns"""
        if self.model is not None:
            try:
                joblib.dump(self.model, self._get_model_path())
                joblib.dump(self.scaler, self._get_scaler_path())
                joblib.dump(self.feature_cols, self._get_features_path())
                logger.info(f"Saved volatility model to {self._get_model_path()}")
            except Exception as e:
                logger.error(f"Failed to save model: {e}")

    def train_model(self, force_full_retrain: bool = False) -> bool:
        """
        Train volatility model with incremental learning

        Args:
            force_full_retrain: Force complete retrain instead of incremental

        Returns:
            True if training successful
        """
        # Check if full retrain needed
        if force_full_retrain or self.trainer.should_full_retrain(self.name, days_threshold=90):
            logger.info("=" * 80)
            logger.info("FULL RETRAIN: Loading all historical volatility data")
            logger.info("=" * 80)

            # Get all historical volatility features
            df = self._get_volatility_features()

            if df.empty:
                logger.warning("No volatility data available")
                return False

            logger.info(f"Loaded {len(df)} volatility records with options and economic data")

            # Calculate GARCH features
            df = self._calculate_garch_features_vectorized(df)

            # Create regime labels
            df = self._create_regime_labels(df)

            # Prepare ML features
            X, y, df_clean, feature_cols = self._prepare_ml_features(df)

            if len(X) == 0:
                logger.warning("No valid features after cleaning")
                return False

            logger.info(f"Using {len(feature_cols)} features for volatility regime prediction")

            # Store feature columns
            self.feature_cols = feature_cols

            # Train model (this handles train/test split internally)
            self._train_xgboost(X, y, incremental=False)

            if self.model is None:
                logger.warning("Model training failed")
                return False

            # The model is already saved by _train_xgboost, but we need to get test accuracy
            # Re-split the data to compute final metrics (matching _train_xgboost split)
            from sklearn.model_selection import train_test_split
            from sklearn.utils.class_weight import compute_class_weight

            classes = np.unique(y)
            class_weights = compute_class_weight('balanced', classes=classes, y=y)
            weight_dict = dict(zip(classes, class_weights))
            sample_weights = np.array([weight_dict[label] for label in y])

            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
            )

            # Apply feature selection if used
            if self.use_feature_selection and self.feature_selector is not None:
                X_train_df = pd.DataFrame(X_train, columns=feature_cols)
                X_test_df = pd.DataFrame(X_test, columns=feature_cols)
                X_train_df = self.feature_selector.transform(X_train_df)
                X_test_df = self.feature_selector.transform(X_test_df)
                X_train = X_train_df.values
                X_test = X_test_df.values

            # Scale and compute accuracies
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            train_acc = self.model.score(X_train_scaled, y_train)
            test_acc = self.model.score(X_test_scaled, y_test)

            logger.info(f"✅ Full retrain complete: Train accuracy={train_acc:.3f}, Test accuracy={test_acc:.3f}")

            # Record training in metadata
            training_start = df_clean['vol_date'].min()
            training_end = df_clean['vol_date'].max()

            self.trainer.save_model(
                strategy_name=self.name,
                model=self.model,
                scaler=self.scaler,
                training_start_date=str(training_start),
                training_end_date=str(training_end),
                num_training_samples=len(X),
                num_new_samples=len(X),
                is_full_retrain=True,
                train_accuracy=train_acc,
                test_accuracy=test_acc,  # Proper train/test split
                feature_names=self.feature_cols,
                hyperparameters=self.model.get_params(),
                notes="Full retrain with all historical volatility data"
            )

            return True

        else:
            logger.info("=" * 80)
            logger.info("INCREMENTAL UPDATE: Loading new volatility data")
            logger.info("=" * 80)

            # Get last training date
            last_training_date = self.trainer.get_last_training_date(self.name)

            if last_training_date:
                logger.info(f"Last training: {last_training_date}")
                # Get new data since last training
                df = self._get_volatility_features(lookback_days=30)  # Last 30 days

                if df.empty:
                    logger.info("No new data available")
                    return True

                # Filter to new data only
                df = df[df['vol_date'] > last_training_date]

                if df.empty:
                    logger.info("No new data since last training")
                    return True

                logger.info(f"Found {len(df)} new volatility records")
            else:
                logger.warning("No training history found, performing full retrain")
                return self.train_model(force_full_retrain=True)

            # Calculate GARCH features
            df = self._calculate_garch_features_vectorized(df)

            # Create regime labels
            df = self._create_regime_labels(df)

            # Prepare ML features
            X, y, df_clean, feature_cols = self._prepare_ml_features(df)

            if len(X) == 0:
                logger.warning("No valid new features")
                return True
            
            # FIX: Check for single-class data BEFORE modifying instance state
            if len(np.unique(y)) < 2:
                logger.warning("⚠️  Only one class present in new data. Skipping incremental training for today.")
                return True

            # Check if we have enough samples for training
            min_samples_required = 50
            if len(X) < min_samples_required:
                logger.info(f"Only {len(X)} new samples (need {min_samples_required}), keeping existing model")
                return True

            # Incremental update (retrain with new data)
            # For volatility, we do a full retrain as incremental XGBoost is complex
            logger.info(f"Retraining with {len(X)} new samples")

            # Set feature_cols now that we are committed to training
            self.feature_cols = feature_cols
            self._train_xgboost(X, y, incremental=False)

            if self.model is None:
                logger.warning("Model training failed")
                return False

            # Re-split data to compute proper train/test accuracies
            from sklearn.model_selection import train_test_split
            from sklearn.utils.class_weight import compute_class_weight

            classes = np.unique(y)
            class_weights = compute_class_weight('balanced', classes=classes, y=y)
            weight_dict = dict(zip(classes, class_weights))
            sample_weights = np.array([weight_dict[label] for label in y])

            X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
            )

            # Apply feature selection if used
            if self.use_feature_selection and self.feature_selector is not None:
                X_train_df = pd.DataFrame(X_train, columns=feature_cols)
                X_test_df = pd.DataFrame(X_test, columns=feature_cols)
                X_train_df = self.feature_selector.transform(X_train_df)
                X_test_df = self.feature_selector.transform(X_test_df)
                X_train = X_train_df.values
                X_test = X_test_df.values

            # Scale and compute accuracies
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            train_acc = self.model.score(X_train_scaled, y_train)
            test_acc = self.model.score(X_test_scaled, y_test)

            logger.info(f"✅ Incremental update complete: Train accuracy={train_acc:.3f}, Test accuracy={test_acc:.3f}")

            # Record training in metadata
            training_start = df_clean['vol_date'].min()
            training_end = df_clean['vol_date'].max()

            # Get previous total samples if available
            last_metadata = self.trainer.get_latest_model_metadata(self.name)
            prev_samples = last_metadata.get('num_training_samples', 0) if last_metadata else 0

            self.trainer.save_model(
                strategy_name=self.name,
                model=self.model,
                scaler=self.scaler,
                training_start_date=str(training_start),
                training_end_date=str(training_end),
                num_training_samples=prev_samples + len(X),
                num_new_samples=len(X),
                is_full_retrain=False,
                train_accuracy=train_acc,
                test_accuracy=test_acc,  # Proper train/test split
                feature_names=self.feature_cols,
                hyperparameters=self.model.get_params(),
                notes=f"Incremental update with {len(X)} new volatility samples"
            )

            return True

    def _get_volatility_features(self, lookback_days: int = None) -> pd.DataFrame:
        """Get comprehensive volatility features for ML model with OPTIONS and ECONOMIC data"""
        if self.api:
            return self._get_live_volatility_features(lookback_days)
        
        if lookback_days is None:
            lookback_days = self.lookback_days

        conn = sqlite3.connect(self.db_path)
        query = f"""
            SELECT
                vm.symbol_ticker,
                vm.vol_date,
                -- Multiple volatility estimators
                vm.close_to_close_vol_10d,
                vm.close_to_close_vol_20d,
                vm.close_to_close_vol_60d,
                vm.parkinson_vol_10d,
                vm.parkinson_vol_20d,
                vm.garman_klass_vol_10d,
                vm.garman_klass_vol_20d,
                vm.yang_zhang_vol_10d,
                vm.yang_zhang_vol_20d,
                -- Volatility dynamics
                vm.realized_vol_percentile_1y,
                vm.realized_vol_percentile_3y,
                vm.volatility_of_volatility_20d,
                vm.vol_clustering_index,
                vm.volatility_trend,
                vm.volatility_acceleration,
                -- Volume features
                vm.volume_weighted_volatility,
                vm.abnormal_volume_count_20d,
                -- Gap features
                vm.gap_frequency_60d,
                vm.avg_gap_size_60d,
                vm.gap_volatility_contribution,
                -- Intraday features
                vm.overnight_vol_ratio,
                vm.intraday_vol_ratio,
                -- Technical indicators
                ti.rsi_14,
                ti.atr_14,
                ti.atr_20,
                ti.bb_width,
                ti.adx_14,
                -- Price features
                mf.return_1d,
                mf.return_5d,
                mf.return_20d,
                rpd.close as current_price,
                -- OPTIONS DATA
                od.implied_volatility_30d,
                od.implied_volatility_60d,
                od.implied_volatility_90d,
                od.iv_percentile_1y,
                od.iv_percentile_3y,
                od.put_call_ratio,
                od.put_call_ratio_volume,
                od.put_call_ratio_oi,
                od.put_volume,
                od.call_volume,
                od.total_options_volume,
                od.atm_iv,
                od.skew_25delta,
                -- ECONOMIC INDICATORS
                ei.vix,
                ei.treasury_10y,
                ei.treasury_2y,
                ei.yield_curve_10y_2y,
                ei.dollar_index
            FROM volatility_metrics vm
            LEFT JOIN technical_indicators ti
                ON vm.symbol_ticker = ti.symbol_ticker
                AND vm.vol_date = ti.indicator_date
            LEFT JOIN ml_features mf
                ON vm.symbol_ticker = mf.symbol_ticker
                AND vm.vol_date = mf.feature_date
            LEFT JOIN raw_price_data rpd
                ON vm.symbol_ticker = rpd.symbol_ticker
                AND vm.vol_date = rpd.price_date
            LEFT JOIN (
                SELECT symbol_ticker, options_date,
                       implied_volatility_30d, implied_volatility_60d, implied_volatility_90d,
                       iv_percentile_1y, iv_percentile_3y,
                       put_call_ratio, put_call_ratio_volume, put_call_ratio_oi,
                       put_volume, call_volume, total_options_volume,
                       atm_iv, skew_25delta
                FROM options_data
                WHERE (symbol_ticker, options_date) IN (
                    SELECT symbol_ticker, MAX(options_date)
                    FROM options_data
                    GROUP BY symbol_ticker
                )
            ) od ON vm.symbol_ticker = od.symbol_ticker
            LEFT JOIN (
                SELECT indicator_date, vix, treasury_10y, treasury_2y,
                       yield_curve_10y_2y, dollar_index
                FROM economic_indicators
                WHERE (indicator_date) IN (
                    SELECT MAX(indicator_date) FROM economic_indicators
                )
            ) ei ON vm.vol_date = ei.indicator_date
            WHERE vm.vol_date >= date('now', '-{lookback_days} days')
            ORDER BY vm.symbol_ticker, vm.vol_date
        """
        df = pd.read_sql(query, conn)
        conn.close()
        logger.info(f"Loaded {len(df)} volatility records with options and economic data")
        return df

    def _get_live_volatility_features(self, lookback_days: int = None) -> pd.DataFrame:
        """Get simplified volatility features from live Alpaca data."""
        if lookback_days is None:
            lookback_days = self.lookback_days

        logger.info("Fetching live price data from Alpaca for volatility features...")
        assets = self.api.list_assets(status='active', tradable=True)
        us_equities = [a.symbol for a in assets if a.exchange in ['NASDAQ', 'NYSE']]
        
        end_dt = pd.Timestamp.now(tz='America/New_York')
        start_dt = end_dt - pd.Timedelta(days=lookback_days + 100) # Fetch more data for calculations

        try:
            price_data = self.api.get_bars(
                us_equities,
                "1D",
                start=start_dt.isoformat(),
                end=end_dt.isoformat()
            ).df

            if price_data.empty:
                logger.warning("No price data fetched from Alpaca.")
                return pd.DataFrame()

            df = price_data.reset_index()
            df = df[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df.columns = ['symbol_ticker', 'vol_date', 'open', 'high', 'low', 'close', 'volume']
            df['vol_date'] = df['vol_date'].dt.date

            # Calculate simplified features
            df['return_1d'] = df.groupby('symbol_ticker')['close'].pct_change(1)
            df['return_5d'] = df.groupby('symbol_ticker')['close'].pct_change(5)
            df['return_20d'] = df.groupby('symbol_ticker')['close'].pct_change(20)

            # Close-to-close volatility
            df['close_to_close_vol_10d'] = df.groupby('symbol_ticker')['return_1d'].rolling(10).std().reset_index(0, drop=True)
            df['close_to_close_vol_20d'] = df.groupby('symbol_ticker')['return_1d'].rolling(20).std().reset_index(0, drop=True)
            df['close_to_close_vol_60d'] = df.groupby('symbol_ticker')['return_1d'].rolling(60).std().reset_index(0, drop=True)

            # RSI
            delta = df.groupby('symbol_ticker')['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))

            # ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df.groupby('symbol_ticker')['close'].shift())
            low_close = np.abs(df['low'] - df.groupby('symbol_ticker')['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['atr_14'] = true_range.rolling(14).mean()
            
            df['current_price'] = df['close']

            return df

        except Exception as e:
            logger.error(f"Failed to fetch or process live volatility features: {e}")
            return pd.DataFrame()


    def _calculate_garch_features_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized GARCH calculation for M1 optimization"""
        if ARCH_AVAILABLE:
            for ticker in df['symbol_ticker'].unique():
                mask = df['symbol_ticker'] == ticker
                ticker_data = df[mask].sort_values('vol_date')

                returns = ticker_data['return_1d'].fillna(0).infer_objects(copy=False) * 100

                if len(returns) < 30:
                    continue

                try:
                    model = arch_model(
                        returns,
                        vol='GARCH',
                        p=1,
                        q=1,
                        rescale=False,
                        mean='Zero'
                    )
                    result = model.fit(
                        disp='off',
                        options={'maxiter': 100},
                        show_warning=False
                    )

                    cond_vol = result.conditional_volatility / 100

                    df.loc[mask, 'garch_cond_vol'] = cond_vol.values
                    df.loc[mask, 'garch_forecast'] = df.loc[mask, 'garch_cond_vol'].shift(1)
                    df.loc[mask, 'vol_surprise'] = (
                        df.loc[mask, 'close_to_close_vol_10d'] - df.loc[mask, 'garch_forecast']
                    )

                except Exception as e:
                    logger.warning(f"GARCH fit failed for {ticker}: {e}")
                    self._calculate_simple_garch(df, mask)
        else:
            for ticker in df['symbol_ticker'].unique():
                mask = df['symbol_ticker'] == ticker
                self._calculate_simple_garch(df, mask)

        return df

    def _calculate_simple_garch(self, df: pd.DataFrame, mask) -> None:
        """Vectorized GARCH approximation (no loops)"""
        ticker_data = df[mask].sort_values('vol_date')
        returns = ticker_data['return_1d'].fillna(0)

        alpha = 0.06
        variance = (returns ** 2).ewm(alpha=alpha, adjust=False).mean()
        cond_vol = np.sqrt(variance)

        df.loc[mask, 'garch_cond_vol'] = cond_vol.values
        df.loc[mask, 'garch_forecast'] = df.loc[mask, 'garch_cond_vol'].shift(1)
        df.loc[mask, 'vol_surprise'] = (
            df.loc[mask, 'close_to_close_vol_10d'] - df.loc[mask, 'garch_forecast']
        )

    def _create_regime_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility regime labels with ADJUSTED THRESHOLDS"""
        df = df.sort_values(['symbol_ticker', 'vol_date'])

        df['future_vol_change'] = df.groupby('symbol_ticker')['close_to_close_vol_20d'].shift(-5)
        df['vol_change_pct'] = (
            (df['future_vol_change'] - df['close_to_close_vol_20d']) /
            df['close_to_close_vol_20d']
        )

        # HYBRID FIX: Use adjusted threshold (0.10 instead of 0.15)
        conditions = [
            df['vol_change_pct'] < -self.REGIME_THRESHOLD,  # Significant decrease
            df['vol_change_pct'] > self.REGIME_THRESHOLD,    # Significant increase
        ]
        choices = [0, 2]
        df['regime_label'] = np.select(conditions, choices, default=1)

        return df

    def _prepare_ml_features(self, df: pd.DataFrame) -> tuple:
        """Prepare feature matrix with a simplified feature set for live trading."""
        feature_cols = [
            'close_to_close_vol_10d', 'close_to_close_vol_20d', 'close_to_close_vol_60d',
            'return_1d', 'return_5d', 'return_20d',
            'rsi_14', 'atr_14'
        ]

        df_clean = df.dropna(subset=['regime_label'])

        for col in feature_cols:
            if col in df_clean.columns:
                if df_clean[col].notna().any():
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val if not pd.isna(median_val) else 0)
                else:
                    df_clean[col] = 0

        available_cols = [col for col in feature_cols if col in df_clean.columns]

        if len(available_cols) == 0:
            return np.array([]), np.array([]), pd.DataFrame(), []

        X = df_clean[available_cols].values
        y = df_clean['regime_label'].values

        logger.info(f"Using {len(available_cols)} features for volatility regime prediction")
        return X, y, df_clean, available_cols

    def _train_xgboost(self, X: np.ndarray, y: np.ndarray, incremental: bool = False) -> None:
        """
        Train XGBoost with CLASS WEIGHTS for imbalanced data

        HYBRID FIX: Added class weights to handle imbalanced regime distribution
        """
        # FIX: Cannot train a model if all data belongs to a single class.
        if len(np.unique(y)) < 2:
            logger.warning("⚠️  Only one class present in training data for this run.")
            logger.warning("   Model training will be skipped as it's impossible to train on single-class data.")
            return
            
        if len(X) < self.min_samples:
            logger.warning(f"Not enough samples ({len(X)}) to train model")
            return

        # HYBRID FIX: Compute class weights for imbalanced data
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        weight_dict = dict(zip(classes, class_weights))
        sample_weights = np.array([weight_dict[label] for label in y])

        logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        logger.info(f"Class weights: {weight_dict}")

        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
        )

        # SMART FEATURE SELECTION (if enabled)
        if self.use_feature_selection:
            logger.info("\n" + "="*80)
            logger.info("APPLYING SMART FEATURE SELECTION")
            logger.info("="*80)

            # Convert to DataFrame first (volatility passes numpy arrays)
            X_train_df = pd.DataFrame(X_train, columns=self.feature_cols)
            X_test_df = pd.DataFrame(X_test, columns=self.feature_cols)

            self.feature_selector = SmartFeatureSelector(
                strategy_name=self.name,
                n_features=self.n_features
            )

            # Fit and transform
            X_train_df = self.feature_selector.fit_transform(
                X_train_df, y_train,
                eval_set=(X_test_df, y_test)
            )
            X_test_df = self.feature_selector.transform(X_test_df)

            # Update feature_cols to selected features only
            self.feature_cols = self.feature_selector.selected_features_

            logger.info(f"✅ Feature selection complete: {len(X_train_df.columns)} features selected")

            # Convert back to numpy for scaling
            X_train = X_train_df.values
            X_test = X_test_df.values

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # XGBoost with M1 optimization + class weights
        self.model = xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method='hist',
            device='cpu',
            n_jobs=-1,
            enable_categorical=False,
            early_stopping_rounds=15,
            eval_metric='mlogloss',
            random_state=42,
            verbosity=0
        )

        # Train with sample weights
        self.model.fit(
            X_train_scaled, y_train,
            sample_weight=w_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )

        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        logger.info(f"XGBoost trained (with class weights): Train accuracy={train_score:.3f}, Test accuracy={test_score:.3f}")

        # Save the model
        self._save_model()

    def incremental_train(self, force_full: bool = False) -> dict:
        """
        Incrementally train or update the model with new data

        Args:
            force_full: If True, retrain from scratch

        Returns:
            Training results dict
        """
        logger.info("="*60)
        logger.info("VOLATILITY STRATEGY INCREMENTAL TRAINING")
        logger.info("="*60)

        # Check if we need full retrain
        needs_full_retrain = force_full or self.model is None

        if needs_full_retrain:
            logger.info("Performing FULL model training...")
            lookback = 365  # Use 1 year of data for full training
        else:
            logger.info("Performing INCREMENTAL model update...")
            lookback = 30  # Just recent data for incremental

        # Get data
        df = self._get_volatility_features(lookback_days=lookback)

        if df.empty:
            return {'success': False, 'error': 'No volatility data available'}

        # Calculate features
        df = self._calculate_garch_features_vectorized(df)
        df = self._create_regime_labels(df)
        X, y, df_clean, feature_cols = self._prepare_ml_features(df)

        if len(X) == 0:
            return {'success': False, 'error': 'No valid features after cleaning'}

        self.feature_cols = feature_cols

        # Train model
        self._train_xgboost(X, y, incremental=not needs_full_retrain)

        if self.model is None:
            return {'success': False, 'error': 'Model training failed'}

        # Get class distribution after training
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique.astype(int), counts.astype(int)))

        return {
            'success': True,
            'training_type': 'full' if needs_full_retrain else 'incremental',
            'samples_used': len(X),
            'features_used': len(feature_cols),
            'class_distribution': class_dist,
            'model_saved': True
        }

    def _get_feature_importance(self) -> dict:
        """Get feature importance from XGBoost"""
        if self.model is None or self.feature_cols is None:
            return {}

        importance = dict(zip(self.feature_cols, self.model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def generate_signals(self) -> pd.DataFrame:
        """Generate volatility trading signals with HYBRID FIX applied"""
        # Get model version for tracking
        model_info = self.trainer.get_latest_model_info(self.name)
        model_version = f"v{model_info.get('model_version', 'N/A')}" if model_info else "Unknown"

        # Get volatility features
        df = self._get_volatility_features()

        if df.empty:
            logger.warning("No volatility data available")
            return pd.DataFrame()

        # Calculate GARCH features
        df = self._calculate_garch_features_vectorized(df)

        # Create regime labels
        df = self._create_regime_labels(df)

        # Prepare ML features
        X, y, df_clean, feature_cols = self._prepare_ml_features(df)

        if len(X) == 0:
            logger.warning("No valid features after cleaning")
            return pd.DataFrame()

        # Train if no model exists
        if self.model is None:
            self.feature_cols = feature_cols
            self._train_xgboost(X, y)

            if self.model is None:
                logger.warning("Model training failed")
                return pd.DataFrame()
            # Update version after training
            model_info = self.trainer.get_latest_model_info(self.name)
            model_version = f"v{model_info.get('model_version', 'N/A')}" if model_info else "v1"
        else:
            # Use saved feature columns
            if self.feature_cols is None:
                self.feature_cols = feature_cols

        # Log feature importance
        importance = self._get_feature_importance()
        logger.info("Top 5 volatility features:")
        for feat, imp in list(importance.items())[:5]:
            logger.info(f"  {feat}: {imp:.3f}")

        # Get latest data for prediction
        latest_data = df_clean.groupby('symbol_ticker').tail(1).copy()

        # Use saved feature columns, fill missing with 0
        X_latest_list = []
        for _, row in latest_data.iterrows():
            row_features = [row.get(col, 0) for col in self.feature_cols]
            X_latest_list.append(row_features)
        X_latest = np.array(X_latest_list)
        X_latest_scaled = self.scaler.transform(X_latest)

        # Predict regime
        regime_pred = self.model.predict(X_latest_scaled)
        regime_proba = self.model.predict_proba(X_latest_scaled)

        confidences = np.max(regime_proba, axis=1)

        # HYBRID FIX: Lower confidence threshold (0.50 instead of 0.60)
        high_conf_mask = confidences >= self.CONFIDENCE_THRESHOLD
        buy_mask = (regime_pred == 2) & high_conf_mask  # Vol increasing
        sell_mask = (regime_pred == 0) & high_conf_mask  # Vol decreasing

        # HYBRID FIX: Add signals for stable regime at extreme percentiles
        stable_mask = regime_pred == 1
        vol_percentiles = latest_data['realized_vol_percentile_1y'].values

        # Stable + extreme low vol -> expect vol increase -> BUY volatility
        extreme_low_vol_mask = stable_mask & (vol_percentiles < self.EXTREME_VOL_PERCENTILE_LOW) & high_conf_mask
        # Stable + extreme high vol -> expect vol decrease -> SELL volatility
        extreme_high_vol_mask = stable_mask & (vol_percentiles > self.EXTREME_VOL_PERCENTILE_HIGH) & high_conf_mask

        # Combine masks
        buy_mask = buy_mask | extreme_low_vol_mask
        sell_mask = sell_mask | extreme_high_vol_mask

        # Extract arrays
        symbols = latest_data['symbol_ticker'].values
        dates = latest_data['vol_date'].values
        prices = latest_data['current_price'].values
        atrs = latest_data['atr_14'].values
        garch_forecasts = latest_data['garch_forecast'].values if 'garch_forecast' in latest_data.columns else np.zeros(len(latest_data))

        signals = []

        # Process BUY signals
        buy_indices = np.where(buy_mask)[0]
        if len(buy_indices) > 0:
            buy_prices = prices[buy_indices]
            buy_atrs = atrs[buy_indices]
            # Handle NaN ATRs
            buy_atrs = np.where(np.isnan(buy_atrs), buy_prices * 0.02, buy_atrs)
            buy_stops = buy_prices * (1 - 2*buy_atrs/buy_prices)
            buy_targets = buy_prices * (1 + 3*buy_atrs/buy_prices)

            for i, idx in enumerate(buy_indices):
                signal_source = "extreme_low_vol" if extreme_low_vol_mask[idx] else "regime_2"
                signals.append({
                    'symbol_ticker': symbols[idx],
                    'signal_date': dates[idx],
                    'signal_type': 'BUY',
                    'strength': float(confidences[idx]),
                    'entry_price': float(buy_prices[i]),
                    'stop_loss': float(buy_stops[i]),
                    'take_profit': float(buy_targets[i]),
                    'metadata': f'{{"predicted_regime": {int(regime_pred[idx])}, "confidence": {confidences[idx]:.3f}, "signal_source": "{signal_source}", "model": "XGBoost_Incremental", "model_version": "{model_version}"}}'
                })

        # Process SELL signals
        sell_indices = np.where(sell_mask)[0]
        if len(sell_indices) > 0:
            sell_prices = prices[sell_indices]
            sell_atrs = atrs[sell_indices]
            # Handle NaN ATRs
            sell_atrs = np.where(np.isnan(sell_atrs), sell_prices * 0.02, sell_atrs)
            sell_stops = sell_prices * (1 + 2*sell_atrs/sell_prices)
            sell_targets = sell_prices * (1 - 3*sell_atrs/sell_prices)

            for i, idx in enumerate(sell_indices):
                signal_source = "extreme_high_vol" if extreme_high_vol_mask[idx] else "regime_0"
                signals.append({
                    'symbol_ticker': symbols[idx],
                    'signal_date': dates[idx],
                    'signal_type': 'SELL',
                    'strength': float(confidences[idx]),
                    'entry_price': float(sell_prices[i]),
                    'stop_loss': float(sell_stops[i]),
                    'take_profit': float(sell_targets[i]),
                    'metadata': f'{{"predicted_regime": {int(regime_pred[idx])}, "confidence": {confidences[idx]:.3f}, "signal_source": "{signal_source}", "model": "XGBoost_Incremental", "model_version": "{model_version}"}}'
                })

        logger.info(f"Generated {len(signals)} volatility signals (threshold={self.CONFIDENCE_THRESHOLD}, regime_threshold={self.REGIME_THRESHOLD})")
        return pd.DataFrame(signals)


