import logging
import pandas as pd
import numpy as np
from typing import Optional, Union, List
from sklearn.model_selection import GridSearchCV

# --- RiskLabAI Imports ---
from RiskLabAI.data.labeling.labeling import (
    cusum_filter_events_dynamic_threshold, 
    vertical_barrier
)
from RiskLabAI.data.differentiation.differentiation import fractional_difference_fixed
from RiskLabAI.optimization.nco import get_optimal_portfolio_weights
from RiskLabAI.backtest.validation.purged_kfold import PurgedKFold
from RiskLabAI.features.feature_importance.feature_importance_mdi import FeatureImportanceMDI
from RiskLabAI.features.feature_importance.feature_importance_mda import FeatureImportanceMDA
from RiskLabAI.features.entropy_features.shannon import shannon_entropy

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. TRIPLE BARRIER PATCH (Fixes library crash)
# ==============================================================================

def triple_barrier_patch(close: pd.Series, events: pd.DataFrame, ptsl: list, molecule: list) -> pd.DataFrame:
    """
    Robust implementation of the Triple Barrier Method.
    replaces the buggy library function to ensure stability.
    """
    out = events.loc[molecule].copy()
    out['side'] = 0
    
    if 't1' not in out.columns:
        out['t1'] = pd.NaT

    # Calculate explicit exit prices for the bot
    current_prices = close.loc[out.index]
    out['profit_taking'] = current_prices * (1 + out['trgt'] * ptsl[0])
    out['stop_loss'] = current_prices * (1 - out['trgt'] * ptsl[1])

    for t0, row in out.iterrows():
        try:
            t1_vertical = row['t1']
            trgt = row['trgt']
            
            # Slice path from entry to vertical barrier
            price_path = close.loc[t0:t1_vertical]
            path_returns = (price_path / close.loc[t0]) - 1
            
            # Check Barriers
            pt_level = trgt * ptsl[0]
            sl_level = -trgt * ptsl[1]
            
            touch_pt = path_returns[path_returns > pt_level].index.min()
            touch_sl = path_returns[path_returns < sl_level].index.min()
            
            if pd.isna(touch_pt) and pd.isna(touch_sl):
                out.loc[t0, 'side'] = 0
                out.loc[t0, 't1'] = t1_vertical
                
            elif pd.isna(touch_sl):
                out.loc[t0, 'side'] = 1
                out.loc[t0, 't1'] = touch_pt
                
            elif pd.isna(touch_pt):
                out.loc[t0, 'side'] = -1
                out.loc[t0, 't1'] = touch_sl
                
            else:
                # Touched both; take earliest
                if touch_pt <= touch_sl:
                    out.loc[t0, 'side'] = 1
                    out.loc[t0, 't1'] = touch_pt
                else:
                    out.loc[t0, 'side'] = -1
                    out.loc[t0, 't1'] = touch_sl
                    
        except Exception:
            continue
            
    return out

# ==============================================================================
# 2. CLASS WRAPPERS
# ==============================================================================

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
    def __init__(self, profit_taking_mult=2.0, stop_loss_mult=2.0, max_holding_period=100):
        self.pt_mult = profit_taking_mult
        self.sl_mult = stop_loss_mult
        self.max_holding = max_holding_period

    def get_volatility(self, close: pd.Series, span=100) -> pd.Series:
        return close.pct_change().ewm(span=span).std()

    def label(self, close: pd.Series, events: pd.DataFrame = None) -> pd.DataFrame:
        if events is None:
            events = pd.DataFrame(index=close.index)
            
        vol = self.get_volatility(close)
        
        # 1. Vertical Barrier (Using RiskLabAI)
        if 't1' not in events.columns:
            events['t1'] = vertical_barrier(
                close=close, 
                time_events=events.index, 
                number_days=self.max_holding
            )

        # 2. Setup
        events_ready = events[['t1']].copy().dropna()
        events_ready['trgt'] = vol.loc[events_ready.index]
        events_ready = events_ready.dropna()
        
        # Add Aliases for compatibility
        events_ready['End Time'] = events_ready['t1']
        events_ready['Base Width'] = events_ready['trgt']
        
        ptsl = [self.pt_mult, self.sl_mult]
        
        # 3. Apply Barriers (Using Patch)
        labels = triple_barrier_patch(
            close=close,
            events=events_ready,
            ptsl=ptsl,
            molecule=list(events_ready.index)
        )
        
        # 4. Finalize
        p_start = close.loc[labels.index]
        p_end = close.loc[labels['t1']]
        labels['ret'] = (p_end.values / p_start.values) - 1
        labels['bin'] = labels['side'].astype(int)
        
        return labels

# ==============================================================================
# 3. UTILITIES
# ==============================================================================

class PurgedCrossValidator:
    def __init__(self, n_splits=5, embargo_pct=0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        
    def get_cv(self, samples_info: Union[pd.DataFrame, pd.Series]):
        times = samples_info['t1'] if isinstance(samples_info, pd.DataFrame) else samples_info
        
        if isinstance(times.index, pd.DatetimeIndex) and times.index.tz is not None:
            times.index = times.index.tz_localize(None)
        if pd.api.types.is_datetime64_any_dtype(times) and getattr(times.dt, 'tz', None) is not None:
             times = times.dt.tz_localize(None)
        
        return PurgedKFold(n_splits=self.n_splits, times=times, embargo=self.embargo_pct)

class FractionalDifferentiator:
    def __init__(self, d=0.4, threshold=1e-5):
        self.d = d
        self.threshold = threshold
        
    def transform(self, series: pd.Series, d: float = None) -> pd.Series:
        d_val = d if d is not None else self.d
        df = pd.DataFrame(series)
        res = fractional_difference_fixed(series=df, degree=d_val, threshold=self.threshold)
        return res.iloc[:, 0]

class AdvancedFeatures:
    @staticmethod
    def get_entropy(series: pd.Series, window=20) -> pd.Series:
        def _calc(window_data):
            try:
                bins = pd.qcut(window_data, q=5, labels=False, duplicates='drop')
                message = "".join(map(str, bins))
                return shannon_entropy(message)
            except ValueError: return 0.0
        return series.rolling(window=window).apply(_calc, raw=False)

    @staticmethod
    def get_regime(close: pd.Series, window=50) -> pd.Series:
        returns = close.pct_change()
        vol = returns.rolling(window=window).std()
        avg_vol = vol.rolling(window=window*2).mean()
        ma_fast = close.rolling(window=int(window/2)).mean()
        ma_slow = close.rolling(window=window).mean()
        
        regime = pd.Series(0, index=close.index)
        regime[(vol < avg_vol) & (ma_fast > ma_slow)] = 1
        regime[(vol > avg_vol) | (ma_fast < ma_slow)] = -1
        return regime

class ModelTuner:
    def __init__(self, n_splits=5, embargo_pct=0.01):
        self.cv_engine = PurgedCrossValidator(n_splits, embargo_pct)

    def tune(self, model, X, y, t1, param_grid):
        cv_gen = self.cv_engine.get_cv(t1)
        search = GridSearchCV(
            estimator=model, param_grid=param_grid, scoring='f1_weighted', cv=cv_gen, n_jobs=-1
        )
        search.fit(X, y)
        return search.best_estimator_, search.best_params_

class HRPPortfolio:
    def optimize(self, returns: pd.DataFrame) -> pd.Series:
        cov = returns.cov().values
        weights = get_optimal_portfolio_weights(covariance=cov, mu=None)
        return pd.Series(weights, index=returns.columns)

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
    def __init__(self): pass
    def create_labels(self, primary_preds, actual_labels):
        return (primary_preds == actual_labels).astype(int)
    def get_bet_size(self, meta_model, features, min_prob=0.5):
        if hasattr(meta_model, "predict_proba"):
            probs = meta_model.predict_proba(features)[:, 1]
        else:
            probs = meta_model.predict(features)
        bets = pd.Series(probs, index=features.index)
        bets[bets < min_prob] = 0
        return bets