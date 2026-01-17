import logging
import pandas as pd
import numpy as np
from typing import Optional, Union, List
from sklearn.model_selection import GridSearchCV

# --- RiskLabAI Imports ---
from RiskLabAI.data.labeling.labeling import (
    cusum_filter_events_dynamic_threshold, 
    triple_barrier, 
    vertical_barrier
)
from RiskLabAI.data.differentiation.differentiation import fractional_difference_fixed
from RiskLabAI.optimization.nco import get_optimal_portfolio_weights
from RiskLabAI.backtest.validation.purged_kfold import PurgedKFold
from RiskLabAI.features.feature_importance.feature_importance_mdi import FeatureImportanceMDI
from RiskLabAI.features.feature_importance.feature_importance_mda import FeatureImportanceMDA
from RiskLabAI.features.entropy_features.shannon import shannon_entropy

logger = logging.getLogger(__name__)


class CUSUMEventFilter:
    def __init__(self, threshold: Optional[float] = None):
        self.threshold = threshold

    def get_events(self, close: pd.Series, threshold: float = None) -> pd.DatetimeIndex:
        if threshold is None:
            threshold = self.threshold if self.threshold else close.pct_change().std()
        
        if isinstance(threshold, (int, float)):
            h = pd.Series(threshold, index=close.index)
        else:
            h = threshold
            
        return cusum_filter_events_dynamic_threshold(prices=close, threshold=h)


class TripleBarrierLabeler:
    def __init__(self, profit_taking_mult=2.0, stop_loss_mult=2.0, max_holding_period=30):
        self.pt_mult = profit_taking_mult
        self.sl_mult = stop_loss_mult
        self.max_holding = max_holding_period

    def get_volatility(self, close: pd.Series, span=20) -> pd.Series:
        return close.pct_change().ewm(span=span).std()

    def label(self, close: pd.Series, events: pd.DataFrame = None) -> pd.DataFrame:
        if events is None:
            events = pd.DataFrame(index=close.index)
            
        vol = self.get_volatility(close)
        
        # Vertical Barrier (Time Limit)
        if 't1' not in events.columns:
            events['t1'] = vertical_barrier(
                close=close, 
                time_events=events.index, 
                number_days=self.max_holding
            )

        events_ready = events[['t1']].copy()
        events_ready['trgt'] = vol.loc[events_ready.index]
        events_ready = events_ready.dropna()

        ptsl = [self.pt_mult, self.sl_mult]
        
        labels = triple_barrier(
            close=close,
            events=events_ready,
            ptsl=ptsl,
            molecule=list(events_ready.index)
        )
        
        # Ensure 'bin' column exists and preserve Neutral class (0)
        if 'bin' not in labels.columns:
            labels['bin'] = 0
            target_vol = events_ready.loc[labels.index, 'trgt']
            
            upper = target_vol * self.pt_mult
            lower = -target_vol * self.sl_mult
            
            labels.loc[labels['ret'] > upper, 'bin'] = 1
            labels.loc[labels['ret'] < lower, 'bin'] = -1

        labels['bin'] = labels['bin'].fillna(0).astype(int)
        return labels


class FractionalDifferentiator:
    def __init__(self, d=0.4, threshold=1e-5):
        self.d = d
        self.threshold = threshold
        
    def transform(self, series: pd.Series, d: float = None) -> pd.Series:
        d_val = d if d is not None else self.d
        df = pd.DataFrame(series)
        
        res = fractional_difference_fixed(
            series=df, 
            degree=d_val, 
            threshold=self.threshold
        )
        return res.iloc[:, 0]


class AdvancedFeatures:
    """
    Institutional-grade features for robust model training.
    """
    @staticmethod
    def get_entropy(series: pd.Series, window=20) -> pd.Series:
        """
        Calculates Rolling Shannon Entropy.
        High entropy indicates market chaos/crash risk.
        """
        def _calc(window_data):
            try:
                # Discretize data into bins to form a "message"
                bins = pd.qcut(window_data, q=5, labels=False, duplicates='drop')
                message = "".join(map(str, bins))
                return shannon_entropy(message)
            except ValueError:
                return 0.0

        return series.rolling(window=window).apply(_calc, raw=False)

    @staticmethod
    def get_regime(close: pd.Series, window=50) -> pd.Series:
        """
        Detects Market Regime: 1 (Safe/Bull), -1 (Risk/Bear).
        Uses Volatility and Trend logic to filter trades during crashes.
        """
        # 1. Volatility Regime
        returns = close.pct_change()
        vol = returns.rolling(window=window).std()
        avg_vol = vol.rolling(window=window*2).mean()
        
        # 2. Trend Regime
        ma_fast = close.rolling(window=int(window/2)).mean()
        ma_slow = close.rolling(window=window).mean()
        
        regime = pd.Series(0, index=close.index)
        
        # Safe: Low Vol + Uptrend
        regime[(vol < avg_vol) & (ma_fast > ma_slow)] = 1
        
        # Risky: High Vol OR Downtrend
        regime[(vol > avg_vol) | (ma_fast < ma_slow)] = -1
        
        return regime


class ModelTuner:
    """
    Hyperparameter tuning using Purged Cross-Validation.
    """
    def __init__(self, n_splits=5, embargo_pct=0.01):
        self.cv_engine = PurgedCrossValidator(n_splits, embargo_pct)

    def tune(self, model, X, y, t1, param_grid):
        # Generate the purged CV splits
        cv_gen = self.cv_engine.get_cv(t1)
        
        # Grid Search with F1 score (balance precision/recall)
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='f1_weighted',
            cv=cv_gen,
            n_jobs=-1
        )
        search.fit(X, y)
        return search.best_estimator_, search.best_params_


class HRPPortfolio:
    def optimize(self, returns: pd.DataFrame) -> pd.Series:
        cov = returns.cov().values
        weights = get_optimal_portfolio_weights(covariance=cov, mu=None)
        return pd.Series(weights, index=returns.columns)


class PurgedCrossValidator:
    def __init__(self, n_splits=5, embargo_pct=0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        
    def get_cv(self, samples_info: Union[pd.DataFrame, pd.Series]):
        times = samples_info['t1'] if isinstance(samples_info, pd.DataFrame) else samples_info
        
        return PurgedKFold(
            n_splits=self.n_splits, 
            times=times, 
            embargo=self.embargo_pct
        )


class FeatureImportance:
    @staticmethod
    def get_mdi(model, feature_names: List[str]) -> pd.DataFrame:
        calculator = FeatureImportanceMDI()
        return calculator.compute(model, feature_names)

    @staticmethod
    def get_mda(model, X, y, cv_gen) -> pd.DataFrame:
        calculator = FeatureImportanceMDA()
        return calculator.compute(model, X, y, cv_gen)


class MetaLabeler:
    def __init__(self): 
        pass

    def create_labels(self, primary_preds: np.ndarray, actual_labels: np.ndarray) -> np.ndarray:
        return (primary_preds == actual_labels).astype(int)
    
    def get_bet_size(self, meta_model, features: pd.DataFrame, min_prob=0.5) -> pd.Series:
        if hasattr(meta_model, "predict_proba"):
            probs = meta_model.predict_proba(features)[:, 1]
        else:
            probs = meta_model.predict(features)
        
        bets = pd.Series(probs, index=features.index)
        bets[bets < min_prob] = 0
        return bets