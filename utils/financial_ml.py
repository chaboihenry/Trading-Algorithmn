import logging
import pandas as pd
import numpy as np
from typing import Optional, Union, List
from sklearn.model_selection import GridSearchCV

# --- RiskLabAI Imports ---
# We use the library for everything EXCEPT the triple_barrier function
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
# 1. TRIPLE BARRIER LOGIC (Local Copy from Source)
# ==============================================================================

def triple_barrier_local(close: pd.Series, events: pd.DataFrame, ptsl: List[float], molecule: List[pd.Timestamp]) -> pd.DataFrame:
    """
    Apply the triple-barrier method.
    
    COPIED FROM RISKLABAI SOURCE with two fixes:
    1. Initialize columns to prevent 'drop' crash.
    2. Keep columns in output so the bot knows exit prices.
    """
    # Filter for events this worker owns
    events_filtered = events.loc[molecule]
    output = pd.DataFrame(index=events_filtered.index)
    output["End Time"] = events_filtered["End Time"]

    # --- FIX 1: Initialize columns to NaT/NaN to prevent crash ---
    output["stop_loss"] = pd.NaT
    output["profit_taking"] = pd.NaT

    # 1. Set horizontal barriers
    if ptsl[0] > 0:
        profit_taking = ptsl[0] * events_filtered["Base Width"]
    else:
        profit_taking = pd.Series(np.inf, index=events_filtered.index)

    if ptsl[1] > 0:
        stop_loss = -ptsl[1] * events_filtered["Base Width"]
    else:
        stop_loss = pd.Series(-np.inf, index=events_filtered.index)

    # Get side if it exists, otherwise default to 1 (long)
    side = events_filtered.get("Side", pd.Series(1.0, index=events_filtered.index))

    # 2. Find first touch time
    # (Iterate through valid events)
    for location, vertical_barrier_time in events_filtered["End Time"].fillna(close.index[-1]).items():
        try:
            # Path prices from event start to vertical barrier
            path_prices = close.loc[location:vertical_barrier_time]
            
            # Calculate path returns, adjusted by side
            path_returns = (
                np.log(path_prices / close[location]) * side.at[location]
            )

            # Check Stop Loss Touch
            sl_touches = path_returns[path_returns < stop_loss.at[location]].index
            if not sl_touches.empty:
                output.loc[location, "stop_loss"] = sl_touches.min()
            
            # Check Profit Taking Touch
            pt_touches = path_returns[path_returns > profit_taking.at[location]].index
            if not pt_touches.empty:
                output.loc[location, "profit_taking"] = pt_touches.min()
                
        except Exception:
            continue

    # The 'End Time' column in output now holds the *first* barrier touched (Vertical, SL, or PT)
    # We take the min of all three times.
    output["End Time"] = output[["End Time", "stop_loss", "profit_taking"]].min(axis=1)
    
    # --- FIX 2: Do NOT drop columns. Return them so the bot can use them. ---
    return output


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

        # 2. Prepare Inputs
        events_ready = events[['t1']].copy().dropna()
        events_ready['trgt'] = vol.loc[events_ready.index]
        events_ready = events_ready.dropna()
        
        # Add Aliases required by the source code logic
        events_ready['End Time'] = events_ready['t1']
        events_ready['Base Width'] = events_ready['trgt']
        events_ready['Side'] = 1  # Default to Long
        
        ptsl = [self.pt_mult, self.sl_mult]
        
        # 3. Apply Barriers (Using LOCAL COPY of the source code)
        labels = triple_barrier_local(
            close=close,
            events=events_ready,
            ptsl=ptsl,
            molecule=list(events_ready.index)
        )
        
        # 4. Finalize
        # Map library output names back to our standard names
        labels.rename(columns={"End Time": "t1"}, inplace=True)
        
        # Calculate Returns based on the touch time found
        p_start = close.loc[labels.index]
        p_end = close.loc[labels['t1']]
        labels['ret'] = (p_end.values / p_start.values) - 1
        
        # Calculate Bin (1=Win, -1=Loss, 0=Timeout)
        # We can infer this from which barrier was hit first
        labels['bin'] = 0
        
        # If t1 == profit_taking -> 1
        # If t1 == stop_loss -> -1
        # If t1 == original vertical barrier -> 0
        
        # (Simplified sign check on return is often robust enough)
        labels['bin'] = np.sign(labels['ret']).astype(int)
        
        # Explicitly calculate price levels for the bot using the multipliers
        labels['profit_taking_price'] = close.loc[labels.index] * (1 + events_ready['trgt'] * self.pt_mult)
        labels['stop_loss_price'] = close.loc[labels.index] * (1 - events_ready['trgt'] * self.sl_mult)

        return labels

# ==============================================================================
# 3. UTILITIES (Timezone Safe)
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