import logging
import pandas as pd
import numpy as np
from ta import momentum, volatility 
from datetime import datetime, timedelta
from typing import Dict, Tuple

from lumibot.strategies.strategy import Strategy
from lumibot.entities import Asset

# --- RiskLabAI Integration ---
from utils.financial_ml import (
    CUSUMEventFilter, 
    TripleBarrierLabeler, 
    MetaLabeler, 
    FractionalDifferentiator,
    AdvancedFeatures,
    ModelTuner,
    PurgedCrossValidator,
    FeatureImportance,
    HRPPortfolio
)

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import clone

from config.all_symbols import SYMBOLS
from data.model_storage import ModelStorage
from data.tick_storage import TickStorage
from data.tick_to_bars import ImbalanceBarGenerator

logger = logging.getLogger(__name__)

class RiskLabAIModel:
    def __init__(self, symbol: str = None):
        self.symbol = symbol
        self.primary_model = None
        self.meta_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
        self.cusum = CUSUMEventFilter(threshold=None) 
        self.labeler = TripleBarrierLabeler(
            profit_taking_mult=2.0, 
            stop_loss_mult=2.0,
            max_holding_period=100
        )
        self.meta_labeler = MetaLabeler()
        self.frac_diff = FractionalDifferentiator(d=0.4)
        self.tuner = ModelTuner(n_splits=5, embargo_pct=0.01)

    def generate_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        df = bars.copy()
        
        # 1. Returns & Volatility
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['sq_ret'] = df['log_ret'] ** 2
        df['ewma_vol'] = df['sq_ret'].ewm(alpha=0.03, adjust=False).mean().apply(np.sqrt)
        
        # 2. Technical Indicators
        df['rsi'] = momentum.rsi(df['close'], window=14)
        bb_high = volatility.bollinger_hband(df['close'], window=20, window_dev=2)
        bb_low = volatility.bollinger_lband(df['close'], window=20, window_dev=2)
        df['bb_width'] = (bb_high - bb_low) / df['close']
        df['bb_position'] = (df['close'] - bb_low) / (bb_high - bb_low)

        # 3. Advanced Features
        df['frac_close'] = self.frac_diff.transform(df['close'])
        df['entropy'] = AdvancedFeatures.get_entropy(df['close'], window=20)
        df['regime'] = AdvancedFeatures.get_regime(df['close'], window=50)
        
        # 4. Microstructure
        df['tick_rule'] = np.where(df['close'] > df['close'].shift(1), 1, -1)
        df['imbalance'] = df['tick_rule'] * df['volume']
        df['cumulative_imbalance'] = df['imbalance'].rolling(10).sum()
        df['ma_fast'] = df['close'].rolling(20).mean()
        df['ma_slow'] = df['close'].rolling(50).mean()
        df['trend_strength'] = (df['ma_fast'] - df['ma_slow']) / df['ma_slow']

        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        excluded_cols = ['timestamp', 'bar_start', 'bar_end', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'regime', 'sq_ret']
        self.feature_names = [c for c in df.columns if c not in excluded_cols]
        return df

    def train_from_ticks(self, symbol: str, min_samples=200, tune=False) -> Dict:
        logger.info(f"[{symbol}] Loading ticks...")
        ticks = TickStorage().load_ticks(symbol)
        if ticks.empty: return {'success': False, 'reason': 'no_data'}

        logger.info(f"[{symbol}] Generating Bars...")
        bars = ImbalanceBarGenerator.process_ticks(ticks, batch_size=1000000)
        if bars.empty or len(bars) < min_samples: return {'success': False, 'reason': 'insufficient_bars'}

        if isinstance(bars.index, pd.DatetimeIndex) and bars.index.tz is not None:
            bars.index = bars.index.tz_localize(None)

        logger.info(f"[{symbol}] Generating Features...")
        features_df = self.generate_features(bars)
        
        close_prices = bars.loc[features_df.index, 'close']
        events = self.cusum.get_events(close_prices)
        labels = self.labeler.label(close_prices, pd.DataFrame(index=events))
        
        common_idx = labels.index.intersection(features_df.index)
        X = features_df.loc[common_idx, self.feature_names]
        y = labels.loc[common_idx, 'bin']
        t1 = labels.loc[common_idx, 't1'] if 't1' in labels.columns else pd.Series(index=common_idx, data=common_idx)

        # Normalize Timezones
        if isinstance(X.index, pd.DatetimeIndex) and X.index.tz is not None: X.index = X.index.tz_localize(None)
        if isinstance(y.index, pd.DatetimeIndex) and y.index.tz is not None: y.index = y.index.tz_localize(None)
        if isinstance(t1.index, pd.DatetimeIndex) and t1.index.tz is not None: t1.index = t1.index.tz_localize(None)
        if pd.api.types.is_datetime64_any_dtype(t1) and hasattr(t1, 'dt') and getattr(t1.dt, 'tz', None) is not None:
            t1 = t1.dt.tz_localize(None)

        if len(y) < 50: return {'success': False, 'reason': 'insufficient_labels'}

        base_estimator = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', n_jobs=1))
        ])

        logger.info(f"[{symbol}] Purged CV for Meta-Labels...")
        cv = PurgedCrossValidator(n_splits=5, embargo_pct=0.01)
        
        # Meta Labels Generation
        meta_features = np.zeros(len(y))
        valid_indices_mask = np.zeros(len(y), dtype=bool)
        
        cv_gen = cv.get_cv(t1)
        for train_idx, val_idx in cv_gen.split(X, y):
            temp_model = clone(base_estimator)
            temp_model.fit(X.iloc[train_idx], y.iloc[train_idx])
            val_preds = temp_model.predict(X.iloc[val_idx])
            meta_features[val_idx] = val_preds
            valid_indices_mask[val_idx] = True

        logger.info(f"[{symbol}] Training Final Models...")
        self.primary_model = clone(base_estimator)
        self.primary_model.fit(X, y)
        
        X_meta = X.iloc[valid_indices_mask]
        y_meta_target = (meta_features[valid_indices_mask] == y.iloc[valid_indices_mask]).astype(int)
        
        self.meta_model = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=5),
            n_estimators=50,
            max_samples=0.6,
            n_jobs=1
        ).fit(X_meta, y_meta_target)

        return {'success': True, 'n_samples': len(X), 'n_features': len(self.feature_names)}

    def predict(self, current_bar_df: pd.DataFrame) -> Tuple[int, float]:
        if self.primary_model is None: return 0, 0.0

        features = self.generate_features(current_bar_df)
        if features.empty: return 0, 0.0
        
        latest = features.iloc[[-1]]
        if latest['regime'].iloc[0] == -1: return 0, 0.0

        X = latest[self.feature_names]
        signal = self.primary_model.predict(X)[0]
        confidence = self.meta_labeler.get_bet_size(self.meta_model, X).iloc[0]
        
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
            
            if signal == 1 and conf > 0.8:
                if pos is None:
                    base_w = self.target_weights.get(symbol, 1.0/len(self.symbols))
                    scale = self._calculate_kelly(conf)
                    
                    budget = self.get_portfolio_value() * base_w * scale
                    qty = max(1, int(budget // bars['close'].iloc[-1]))
                    
                    logger.info(f"[{symbol}] BUY | HRP: {base_w:.2%} | Kelly: {scale:.2f} | Qty: {qty}")
                    if qty > 0:
                        self.submit_order(self.create_order(symbol, qty, "buy"))

            elif signal == -1 and conf > 0.8:
                if pos:
                    logger.info(f"[{symbol}] SELL (Conf: {conf:.2f})")
                    self.sell_all(symbol)

    def _update_hrp_weights(self):
        """
        Runs Hierarchical Risk Parity (HRP) optimization.
        Uses RiskLabAI's portfolio optimization to allocate capital based on 
        cluster-based risk rather than simple mean-variance.
        """
        now = datetime.now()
        # Run daily to avoid excessive API calls
        if self.last_rebalance and (now - self.last_rebalance) < timedelta(hours=24):
            return

        logger.info("Running HRP Optimization...")
        try:
            price_data = {}
            # Limit universe for speed, or use full SYMBOLS list
            target_list = self.symbols[:50] 
            
            for sym in target_list:
                # Fetch recent daily data for correlation analysis
                bars = self.get_historical_prices(sym, 120, "day").df
                if not bars.empty:
                    price_data[sym] = bars['close']
            
            if not price_data:
                logger.warning("No data for HRP. Using Equal Weights.")
                return

            prices_df = pd.DataFrame(price_data)
            returns_df = prices_df.pct_change().dropna()
            
            # RiskLabAI HRP Optimization
            self.target_weights = self.hrp_portfolio.optimize(returns_df).to_dict()
            self.last_rebalance = now
            
            # Log top allocations for verification
            sorted_w = sorted(self.target_weights.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"HRP Weights Updated. Top 5: {sorted_w}")
            
        except Exception as e:
            logger.error(f"HRP Failed: {e}. Defaulting to existing weights.")

    def _calculate_kelly(self, confidence, win_loss=1.0):
        if confidence <= 0.5: return 0.0
        p = confidence
        f = p - ((1-p)/win_loss)
        # Use Half-Kelly for safety
        return max(0.0, min(f * 0.5, 1.0))