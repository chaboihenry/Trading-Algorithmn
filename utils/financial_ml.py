import logging
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Optional, Union, Dict, List

# --- PATH SETUP ---
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

logger = logging.getLogger(__name__)

# --- RiskLabAI Library Imports (Safe Import) ---
try:
    from RiskLabAI.data.labeling import (
        cusum_filter_events_dynamic_threshold, 
        triple_barrier,
        vertical_barrier
    )
    from RiskLabAI.data.differentiation import fractional_difference_fixed
    from RiskLabAI.optimization import (
        get_optimal_portfolio_weights
    )
    from RiskLabAI.backtest.validation.purged_kfold import PurgedKFold
    
    # Feature Importance Imports
    from RiskLabAI.features.feature_importance.feature_importance_mdi import mean_decrease_impurity
    from RiskLabAI.features.feature_importance.feature_importance_mda import mean_decrease_accuracy

except ImportError as e:
    logger.warning(f"RiskLabAI library missing or incomplete: {e}. ML features will be limited.")
    # Dummy functions to prevent crash
    def cusum_filter_events_dynamic_threshold(*args, **kwargs): return pd.DatetimeIndex([])
    def triple_barrier(*args, **kwargs): return pd.DataFrame()
    def vertical_barrier(*args, **kwargs): return pd.Series()
    def fractional_difference_fixed(*args, **kwargs): return pd.DataFrame()
    def get_optimal_portfolio_weights(*args, **kwargs): return []
    class PurgedKFold: pass
    def mean_decrease_impurity(*args, **kwargs): return pd.DataFrame()
    def mean_decrease_accuracy(*args, **kwargs): return pd.DataFrame()

# ... (The rest of the file stays exactly the same) ...
# =============================================================================
# 1. EVENT SAMPLING (CUSUM)
# =============================================================================
class CUSUMEventFilter:
    def __init__(self, threshold: Optional[float] = None):
        self.threshold = threshold
    def get_events(self, close: pd.Series, threshold: float = None) -> pd.DatetimeIndex:
        if threshold is None:
            if self.threshold: threshold = self.threshold
            else:
                diff = close.diff().dropna()
                threshold = diff.std() * 0.7
        if isinstance(threshold, (int, float)):
            h = pd.Series(threshold, index=close.index)
        else: h = threshold
        try: return cusum_filter_events_dynamic_threshold(prices=close, threshold=h)
        except Exception as e:
            logger.error(f"CUSUM error: {e}")
            return pd.DatetimeIndex([])

class TripleBarrierLabeler:
    def __init__(self, profit_taking_mult=2.0, stop_loss_mult=2.0, max_holding_period=30):
        self.pt_mult = profit_taking_mult
        self.sl_mult = stop_loss_mult
        self.max_holding = max_holding_period
    def get_volatility(self, close: pd.Series, span=20) -> pd.Series:
        returns = np.log(close / close.shift(1)).dropna()
        return returns.ewm(span=span).std().reindex(close.index).ffill().bfill()
    def label(self, close: pd.Series, events: pd.DataFrame = None) -> pd.DataFrame:
        if events is None: events = pd.DataFrame(index=close.index)
        valid_times = events.index.intersection(close.index)
        events = events.loc[valid_times]
        if events.empty: return pd.DataFrame()
        daily_vol = self.get_volatility(close)
        if 't1' not in events.columns:
            events['t1'] = vertical_barrier(close=close, time_events=events.index, number_days=self.max_holding)
        target = daily_vol.reindex(events.index)
        target = target[target > 0] 
        events = events.loc[target.index]
        events_formatted = pd.DataFrame(index=events.index)
        events_formatted['End Time'] = events['t1']
        events_formatted['Base Width'] = target
        events_formatted['Side'] = pd.Series(1, index=events.index)
        ptsl = [self.pt_mult, self.sl_mult]
        try:
            labels = triple_barrier(close=close, events=events_formatted, ptsl=ptsl, molecule=list(events_formatted.index))
            labels['bin'] = 0
            return labels
        except Exception as e:
            logger.error(f"Labeling error: {e}")
            return pd.DataFrame()

class MetaLabeler:
    def __init__(self): pass
    def create_labels(self, primary_preds: np.ndarray, actual_labels: np.ndarray) -> np.ndarray:
        return (primary_preds == actual_labels).astype(int)
    def get_bet_size(self, meta_model, features: pd.DataFrame, min_prob=0.5) -> pd.Series:
        try: probs = meta_model.predict_proba(features)[:, 1]
        except AttributeError: probs = meta_model.predict(features)
        bets = pd.Series(probs, index=features.index)
        bets[bets < min_prob] = 0
        return bets

class PurgedCrossValidator:
    def __init__(self, n_splits=5, embargo_pct=0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
    def get_cv(self, samples_info: Union[pd.DataFrame, pd.Series]):
        times = samples_info['t1'] if isinstance(samples_info, pd.DataFrame) else samples_info
        return PurgedKFold(n_splits=self.n_splits, times=times, embargo=self.embargo_pct)

class HRPPortfolio:
    def optimize(self, returns: pd.DataFrame) -> pd.Series:
        try:
            cov = returns.cov().values
            weights = get_optimal_portfolio_weights(covariance=cov, mu=None)
            return pd.Series(weights, index=returns.columns)
        except Exception as e:
            logger.error(f"HRP failed: {e}")
            return pd.Series(1.0/len(returns.columns), index=returns.columns)

class FractionalDifferentiator:
    def __init__(self, d=None, threshold=0.01):
        self.d = d
        self.threshold = threshold
    def transform(self, series: pd.Series, d: float = None) -> pd.Series:
        d_val = d if d is not None else (self.d or 1.0)
        try:
            df = pd.DataFrame(series)
            res = fractional_difference_fixed(series=df, degree=d_val, threshold=self.threshold)
            return res.iloc[:, 0]
        except Exception: return pd.Series()

class FeatureImportance:
    @staticmethod
    def get_mdi(model, feature_names: List[str]) -> pd.DataFrame:
        try: return mean_decrease_impurity(model, feature_names)
        except Exception: return pd.DataFrame()
    @staticmethod
    def get_mda(model, X, y, cv_gen) -> pd.DataFrame:
        try: return mean_decrease_accuracy(model, X, y, cv_gen)
        except Exception: return pd.DataFrame()