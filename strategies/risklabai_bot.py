import logging
import pandas as pd
import numpy as np
import os
import boto3
import io
import sys
from pathlib import Path
from dotenv import load_dotenv
from ta import momentum, volatility 
from datetime import datetime, timedelta
from typing import Dict, Tuple

# Load Env
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent
env_path = project_root / "config" / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# Lumibot
from lumibot.strategies.strategy import Strategy
from lumibot.entities import Asset

# Machine Learning
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import clone

# Financial ML Utils
from utils.financial_ml import (
    CUSUMEventFilter, TripleBarrierLabeler, MetaLabeler, 
    FractionalDifferentiator, AdvancedFeatures, ModelTuner, 
    PurgedCrossValidator, HRPPortfolio
)

# Data & Config
from config.all_symbols import SYMBOLS
from config.aws_config import AWS_REGION, S3_MODEL_BUCKET
from data.model_storage import ModelStorage
from data.tick_to_bars import ImbalanceBarGenerator

logger = logging.getLogger(__name__)

class RiskLabAIModel:
    def __init__(self, symbol: str = None):
        self.symbol = symbol
        self.primary_model = None
        self.meta_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.cusum = None 
        
        self.labeler = TripleBarrierLabeler(
            profit_taking_mult=2.0, 
            stop_loss_mult=2.0,
            max_holding_period=20
        )
        self.meta_labeler = MetaLabeler()
        self.frac_diff = FractionalDifferentiator(d=0.4)
        self.tuner = ModelTuner(n_splits=5, embargo_pct=0.01)

    def _get_s3_client(self):
        key_id = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        if not key_id or not secret_key: return None
        return boto3.client(
            's3',
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key_id=key_id,
            aws_secret_access_key=secret_key
        )

    def generate_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        try:
            # 1. Returns & Volatility
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            df['ewma_vol'] = df['log_ret'].ewm(span=50).std()
            
            # 2. Technical Indicators
            df['rsi'] = momentum.rsi(df['close'], window=14)
            bb_high = volatility.bollinger_hband(df['close'], window=20, window_dev=2)
            bb_low = volatility.bollinger_lband(df['close'], window=20, window_dev=2)
            df['bb_width'] = (bb_high - bb_low) / df['close']
            df['bb_position'] = (df['close'] - bb_low) / (bb_high - bb_low)

            # 3. Advanced Features
            try: df['frac_close'] = self.frac_diff.transform(df['close'])
            except: df['frac_close'] = np.nan
            try: df['entropy'] = AdvancedFeatures.get_entropy(df['close'], window=100)
            except: df['entropy'] = np.nan
            
            # 4. Microstructure
            df['tick_rule'] = np.where(df['close'] > df['close'].shift(1), 1, -1)
            df['imbalance'] = df['tick_rule'] * df['volume']
            df['cumulative_imbalance'] = df['imbalance'].rolling(20).sum()
            df['ma_fast'] = df['close'].rolling(50).mean()
            df['ma_slow'] = df['close'].rolling(200).mean()
            df['trend_strength'] = (df['ma_fast'] - df['ma_slow']) / df['ma_slow']

            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Robust Cleaning
            nan_cols = df.columns[df.isna().all()].tolist()
            if nan_cols: df = df.drop(columns=nan_cols)
            df = df.dropna()
            
        except Exception as e:
            logger.error(f"[{self.symbol}] Feature Gen Error: {e}")
            return pd.DataFrame()

        excluded_cols = ['timestamp', 'bar_start', 'bar_end', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'regime', 'sq_ret']
        self.feature_names = [c for c in df.columns if c not in excluded_cols]
        return df

    def train(self, symbol: str, days=365, min_samples=200) -> Dict:
        self.symbol = symbol
        bars = pd.DataFrame()
        
        # 1. Load Data (S3)
        bucket = os.getenv("S3_MODEL_BUCKET", "trading-agent-models")
        s3_key = f"{symbol}/bars/{symbol}_dollar_bars.parquet"
        s3 = self._get_s3_client()
        if s3:
            try:
                logger.info(f"[{symbol}] Downloading {s3_key}...")
                obj = s3.get_object(Bucket=bucket, Key=s3_key)
                with io.BytesIO(obj['Body'].read()) as buffer:
                    bars = pd.read_parquet(buffer)
                logger.info(f"[{symbol}] Loaded {len(bars):,} bars.")
            except Exception as e: logger.error(f"[{symbol}] S3 Error: {e}")
        
        if bars.empty: return {'success': False, 'reason': 'no_data_s3'}

        if not isinstance(bars.index, pd.DatetimeIndex):
            if 'timestamp' in bars.columns: bars.set_index('timestamp', inplace=True)
            else: bars.index = pd.to_datetime(bars.index)
        if bars.index.tz is not None: bars.index = bars.index.tz_localize(None)
        
        start_date = pd.Timestamp.now() - pd.Timedelta(days=days)
        bars = bars[bars.index >= start_date]
        
        if len(bars) < min_samples: 
            return {'success': False, 'reason': f'insufficient_bars_{len(bars)}'}

        # Feature Gen
        logger.info(f"[{symbol}] Generating Features (FracDiff, Entropy)...")
        features_df = self.generate_features(bars)
        if features_df.empty: return {'success': False, 'reason': 'no_features'}
        close_prices = bars.loc[features_df.index, 'close']
        
        # 2. CUSUM Filter
        logger.info(f"[{symbol}] Applying Symmetric CUSUM Filter...")
        daily_vol = close_prices.pct_change().ewm(span=100).std()
        h_threshold = daily_vol.mean() * 2.0 
        self.cusum = CUSUMEventFilter(threshold=h_threshold)
        t_events = self.cusum.get_events(close_prices)
        logger.info(f"[{symbol}] CUSUM filtered {len(close_prices):,} bars down to {len(t_events):,} significant events.")
        
        if len(t_events) < 50: return {'success': False, 'reason': 'cusum_filtered_too_aggressive'}

        # 3. Labeling
        logger.info(f"[{symbol}] Applying Triple Barrier Method...")
        labels = self.labeler.label(close_prices, pd.DataFrame(index=t_events))
        if labels.empty: return {'success': False, 'reason': 'no_labels'}
        
        common_idx = labels.index.intersection(features_df.index)
        X = features_df.loc[common_idx, self.feature_names]
        y_raw = labels.loc[common_idx, 'bin']
        
        y_encoded = self.label_encoder.fit_transform(y_raw)
        y = pd.Series(y_encoded, index=common_idx)
        t1 = labels.loc[common_idx, 't1'] if 't1' in labels.columns else pd.Series(index=common_idx, data=common_idx)
        if hasattr(t1, 'dt') and getattr(t1.dt, 'tz', None) is not None: t1 = t1.dt.tz_localize(None)

        # 4. Primary Model Configuration
        base_estimator = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()), 
            ('clf', XGBClassifier(
                n_estimators=100, 
                max_depth=4, 
                learning_rate=0.05, 
                eval_metric='mlogloss', 
                n_jobs=1,
                # Initial hint, but we override this dynamically per fold
                objective='multi:softprob', 
                num_class=3                 
            ))
        ])

        # 5. Meta-Labeling with ROBUST FOLD HANDLING
        logger.info(f"[{symbol}] Training Meta-Model (Purged CV)...")
        cv = PurgedCrossValidator(n_splits=5, embargo_pct=0.01)
        meta_features = np.zeros(len(y))
        valid_indices_mask = np.zeros(len(y), dtype=bool)
        
        try:
            cv_gen = cv.get_cv(t1)
            for train_idx, val_idx in cv_gen.split(X, y):
                # Data for this fold
                X_train_f, y_train_f = X.iloc[train_idx], y.iloc[train_idx]
                X_val_f = X.iloc[val_idx]

                # --- ROBUST LABEL HANDLING ---
                # 1. Remap labels locally (e.g. [0, 2] -> [0, 1])
                fold_le = LabelEncoder()
                y_train_fold_encoded = fold_le.fit_transform(y_train_f)
                n_classes_fold = len(fold_le.classes_)
                
                # 2. Configure Model for this specific fold
                fold_model = clone(base_estimator)
                
                if n_classes_fold < 2:
                    # Edge Case: Training data has only 1 class. 
                    # We cannot train XGBoost. Predict that single class for everything.
                    # Map local class 0 -> global class
                    single_class = fold_le.classes_[0]
                    val_preds = np.full(len(X_val_f), single_class)
                else:
                    # Update parameters to match LOCAL class count
                    # We keep multi:softprob to get consistent (N, k) probability matrix
                    fold_model.set_params(
                        clf__num_class=n_classes_fold,
                        clf__objective='multi:softprob'
                    )
                    
                    fold_model.fit(X_train_f, y_train_fold_encoded)
                    
                    # 3. Predict & Map Back to Global
                    probs_fold = fold_model.predict_proba(X_val_f)
                    
                    # Create Global Probability Matrix (N, 3)
                    probs_global = np.zeros((len(X_val_f), 3))
                    
                    for local_idx, global_class in enumerate(fold_le.classes_):
                        if global_class < 3: # Safety check
                            probs_global[:, global_class] = probs_fold[:, local_idx]
                    
                    # Argmax on GLOBAL probabilities
                    val_preds = np.argmax(probs_global, axis=1)

                meta_features[val_idx] = val_preds
                valid_indices_mask[val_idx] = True

        except Exception as e:
            logger.error(f"[{symbol}] CV Split Failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'success': False, 'reason': 'cv_failed'}

        # Train Final Primary Model (On ALL data, usually safe from gaps if timeline is long)
        self.primary_model = clone(base_estimator)
        # Ensure final model has correct global params
        self.primary_model.set_params(clf__num_class=3, clf__objective='multi:softprob')
        self.primary_model.fit(X, y)
        
        # 6. Meta Model
        X_meta = X.iloc[valid_indices_mask]
        y_meta_target = (meta_features[valid_indices_mask] == y.iloc[valid_indices_mask]).astype(int)
        
        if len(np.unique(y_meta_target)) < 2:
             self.meta_model = None
        else:
            self.meta_model = LogisticRegression(class_weight='balanced', solver='liblinear')
            self.meta_model.fit(X_meta, y_meta_target)
            
        model_data = {
            'primary_model': self.primary_model,
            'meta_model': self.meta_model,
            'feature_names': self.feature_names,
            'label_encoder': self.label_encoder
        }
        ModelStorage().save_model(symbol, model_data)

        return {'success': True, 'n_samples': len(X), 'n_features': len(self.feature_names)}

    def predict(self, current_bar_df: pd.DataFrame) -> Tuple[int, float]:
        if self.primary_model is None: return 0, 0.0
        features = self.generate_features(current_bar_df)
        if features.empty: return 0, 0.0
        latest = features.iloc[[-1]]
        try: X = latest[self.feature_names]
        except KeyError: return 0, 0.0
        
        try:
            # Predict Probabilities to be safe, then argmax
            probs = self.primary_model.predict_proba(X)[0]
            signal_encoded = np.argmax(probs)
            signal = self.label_encoder.inverse_transform([signal_encoded])[0]
        except: signal = 0
        
        confidence = 0.0
        if self.meta_model:
            if hasattr(self.meta_model, "predict_proba"):
                probs = self.meta_model.predict_proba(X)
                confidence = probs[0][1] if probs.shape[1] > 1 else 0.0
            else: confidence = float(self.meta_model.predict(X)[0])
        
        return int(signal), float(confidence)

class RiskLabAIStrategy(Strategy):
    def initialize(self):
        self.sleeptime = "1M"
        self.symbols = SYMBOLS
        self.models = {}
        self.storage = ModelStorage()
        self.hrp_portfolio = HRPPortfolio()
        self.target_weights = {}
        self.last_rebalance = None
        for sym in self.symbols:
            data = self.storage.load_model(sym)
            if data:
                self.models[sym] = RiskLabAIModel(sym)
                self.models[sym].primary_model = data['primary_model']
                self.models[sym].meta_model = data['meta_model']
                self.models[sym].feature_names = data['feature_names']
                self.models[sym].label_encoder = data['label_encoder']
                logger.info(f"Loaded {sym}")

    def on_trading_iteration(self):
        self._update_hrp_weights()
        for symbol in self.symbols:
            if symbol not in self.models: continue
            bars = self.get_historical_prices(symbol, 100, "minute").df
            if bars.empty: continue
            model = self.models[symbol]
            signal, conf = model.predict(bars)
            pos = self.get_position(symbol)
            if conf > 0.6:
                current_price = bars['close'].iloc[-1]
                vol = model.labeler.get_volatility(bars['close']).iloc[-1]
                if signal == 1 and pos is None:
                    tp_price = current_price * (1 + vol * model.labeler.pt_mult)
                    sl_price = current_price * (1 - vol * model.labeler.sl_mult)
                    base_w = self.target_weights.get(symbol, 1.0/len(self.symbols))
                    scale = self._calculate_kelly(conf)
                    budget = self.get_portfolio_value() * base_w * scale
                    qty = max(1, int(budget // current_price))
                    if qty > 0:
                        self.submit_order(self.create_order(symbol, qty, "buy", take_profit_price=tp_price, stop_loss_price=sl_price))
                elif signal == -1 and pos:
                    self.sell_all(symbol)

    def _update_hrp_weights(self):
        now = datetime.now()
        if self.last_rebalance and (now - self.last_rebalance) < timedelta(hours=24): return
        try:
            price_data = {}
            target_list = self.symbols[:50] 
            for sym in target_list:
                bars = self.get_historical_prices(sym, 120, "day").df
                if not bars.empty: price_data[sym] = bars['close']
            if not price_data: return
            prices_df = pd.DataFrame(price_data)
            returns_df = prices_df.pct_change().dropna()
            self.target_weights = self.hrp_portfolio.optimize(returns_df).to_dict()
            self.last_rebalance = now
            logger.info(f"HRP Weights Updated.")
        except Exception as e: logger.error(f"HRP Failed: {e}")

    def _calculate_kelly(self, confidence, win_loss=1.0):
        if confidence <= 0.5: return 0.0
        p = confidence
        f = p - ((1-p)/win_loss)
        return max(0.0, min(f * 0.5, 1.0))