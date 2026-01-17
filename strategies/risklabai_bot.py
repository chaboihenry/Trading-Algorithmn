import logging
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
from typing import Dict, Tuple, Optional

from lumibot.strategies.strategy import Strategy
from lumibot.entities import Asset

# --- RiskLabAI Integration ---
from utils.financial_ml import (
    CUSUMEventFilter, 
    TripleBarrierLabeler, 
    MetaLabeler, 
    FractionalDifferentiator,
    AdvancedFeatures,
    ModelTuner
)

# --- ML Models ---
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from config.settings import ALPACA_CONFIG
from data.model_storage import ModelStorage
from data.tick_storage import TickStorage
from data.tick_to_bars import ImbalanceBarGenerator

logger = logging.getLogger(__name__)

class RiskLabAIModel:
    """
    Core ML Logic: Feature Engineering, Training, and Prediction.
    Now enhanced with Fractional Differentiation, Entropy, and Regime Detection.
    """
    def __init__(self, symbol: str = None):
        self.symbol = symbol
        self.primary_model = None
        self.meta_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
        # Tools
        self.cusum = CUSUMEventFilter(threshold=None) # Dynamic
        self.labeler = TripleBarrierLabeler(
            profit_taking_mult=2.0, 
            stop_loss_mult=2.0,
            max_holding_period=20
        )
        self.meta_labeler = MetaLabeler()
        self.frac_diff = FractionalDifferentiator(d=0.4)
        self.tuner = ModelTuner(n_splits=5, embargo_pct=0.01)

    def generate_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        Creates institutional-grade features from raw candles.
        """
        df = bars.copy()
        
        # 1. Log Returns (Standard)
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        
        # 2. Fractional Differentiation (Stationary + Memory)
        # Replaces standard price with memory-preserving diff
        df['frac_close'] = self.frac_diff.transform(df['close'])
        
        # 3. Market Regime & Entropy (Crash Detection)
        df['entropy'] = AdvancedFeatures.get_entropy(df['close'], window=20)
        df['regime'] = AdvancedFeatures.get_regime(df['close'], window=50)
        
        # 4. Volatility (Parkinson/Garman-Klass approximation via Range)
        df['volatility'] = (df['high'] - df['low']) / df['close']
        
        # 5. Microstructure (Tick Rule proxy)
        df['tick_rule'] = np.where(df['close'] > df['close'].shift(1), 1, -1)
        df['imbalance'] = df['tick_rule'] * df['volume']
        df['cumulative_imbalance'] = df['imbalance'].rolling(10).sum()

        # 6. Moving Averages (Trends)
        df['ma_fast'] = df['close'].rolling(20).mean()
        df['ma_slow'] = df['close'].rolling(50).mean()
        df['trend_strength'] = (df['ma_fast'] - df['ma_slow']) / df['ma_slow']

        # Cleanup
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        self.feature_names = [c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'regime']]
        return df

    def train_from_ticks(self, symbol: str, min_samples=200, tune=False) -> Dict:
        """
        Full Training Pipeline:
        Ticks -> Bars -> Features -> Labeling -> Tuning -> Training
        """
        logger.info(f"[{symbol}] Loading ticks from database...")
        ticks = TickStorage().load_ticks(symbol)
        if ticks.empty:
            return {'success': False, 'reason': 'no_data'}

        # 1. Generate Bars
        logger.info(f"[{symbol}] Generating Imbalance Bars...")
        bars = ImbalanceBarGenerator.process_ticks(ticks, batch_size=1000000)
        if bars.empty or len(bars) < min_samples:
            return {'success': False, 'reason': 'insufficient_bars'}

        # 2. Feature Engineering
        logger.info(f"[{symbol}] Generating Advanced Features...")
        features_df = self.generate_features(bars)
        
        # 3. Regime Filtering (Optional but recommended)
        # We can exclude 'Crash' regimes from training to prevent learning noise
        # features_df = features_df[features_df['regime'] != -1]

        # 4. Triple Barrier Labeling
        close_prices = bars.loc[features_df.index, 'close']
        
        # Get Events (CUSUM)
        events = self.cusum.get_events(close_prices)
        
        # Get Labels
        labels = self.labeler.label(close_prices, pd.DataFrame(index=events))
        
        # Align Features X and Labels y
        common_idx = labels.index.intersection(features_df.index)
        X = features_df.loc[common_idx, self.feature_names]
        y = labels.loc[common_idx, 'bin']
        
        # Store barrier touch times for Purged CV
        t1 = labels.loc[common_idx, 't1'] if 't1' in labels.columns else pd.Series(index=common_idx, data=common_idx)

        if len(y) < 50:
            return {'success': False, 'reason': 'insufficient_labels'}

        # 5. Train Primary Model (Direction)
        logger.info(f"[{symbol}] Training Primary Model (RF)...")
        
        # Use simple Imputer+RF pipeline
        base_estimator = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample'))
        ])

        if tune:
            logger.info(f"[{symbol}] Tuning Hyperparameters...")
            param_grid = {
                'clf__n_estimators': [50, 100], 
                'clf__max_depth': [3, 5, 10],
                'clf__min_samples_leaf': [5, 10]
            }
            self.primary_model, params = self.tuner.tune(base_estimator, X, y, t1, param_grid)
            logger.info(f"[{symbol}] Best Params: {params}")
        else:
            self.primary_model = base_estimator.fit(X, y)

        # 6. Train Meta Model (Confidence)
        logger.info(f"[{symbol}] Training Meta Model...")
        primary_preds = self.primary_model.predict(X)
        meta_y = self.meta_labeler.create_labels(primary_preds, y.values)
        
        # Meta model uses Bagging for robustness
        self.meta_model = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=5),
            n_estimators=50,
            max_samples=0.6
        ).fit(X, meta_y)

        return {'success': True, 'n_samples': len(X), 'n_features': len(self.feature_names)}

    def predict(self, current_bar_df: pd.DataFrame) -> Tuple[int, float]:
        """
        Live Prediction.
        Returns: (Signal [-1, 0, 1], Confidence [0.0 - 1.0])
        """
        if self.primary_model is None:
            return 0, 0.0

        # Feature Engineering
        features = self.generate_features(current_bar_df)
        if features.empty: return 0, 0.0
        
        latest = features.iloc[[-1]]
        
        # REGIME FILTER: If market is crashing/volatile, HOLD.
        if latest['regime'].iloc[0] == -1:
            return 0, 0.0

        X = latest[self.feature_names]
        
        # Primary Prediction
        signal = self.primary_model.predict(X)[0]
        
        # Meta Prediction (Bet Size)
        confidence = self.meta_labeler.get_bet_size(self.meta_model, X).iloc[0]
        
        return int(signal), float(confidence)


class RiskLabAIStrategy(Strategy):
    """
    Lumibot Strategy Implementation
    """
    def initialize(self):
        self.sleeptime = "1M" 
        self.symbols = ["SPY", "QQQ"] # Default
        self.models = {}
        self.storage = ModelStorage()
        
        # Load pre-trained models
        for sym in self.symbols:
            data = self.storage.load_model(sym)
            if data:
                self.models[sym] = RiskLabAIModel(sym)
                self.models[sym].primary_model = data['primary_model']
                self.models[sym].meta_model = data['meta_model']
                self.models[sym].feature_names = data['feature_names']
                logger.info(f"Loaded model for {sym}")
            else:
                logger.warning(f"No model found for {sym}")

    def on_trading_iteration(self):
        for symbol in self.symbols:
            if symbol not in self.models: continue
            
            # Get recent bars (enough to compute features)
            bars = self.get_historical_prices(symbol, 100, "minute").df
            if bars.empty: continue
            
            # Predict
            model = self.models[symbol]
            signal, conf = model.predict(bars)
            
            # Execution Logic
            pos = self.get_position(symbol)
            if signal == 1 and conf > 0.6:
                if pos is None:
                    order = self.create_order(symbol, quantity=10, side="buy")
                    self.submit_order(order)
            elif signal == -1 and conf > 0.6:
                if pos:
                    self.sell_all()