"""
CUSUM Filter for Event Sampling

Instead of predicting every bar (wasteful, many bars have no signal),
CUSUM detects when cumulative changes exceed a threshold.

This gives you EVENT-DRIVEN sampling:
- Only generate labels when something significant happens
- Reduces noise in training data
- Focuses model on meaningful price movements
"""

from RiskLabAI.filters import cusum_filter
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class CUSUMEventFilter:
    """
    Filters events using CUSUM (Cumulative Sum) algorithm.

    How it works:
    1. Track cumulative sum of changes: S_t = max(0, S_{t-1} + y_t - threshold)
    2. Trigger event when S_t exceeds threshold
    3. Reset S_t to 0 after trigger

    Result: Events only at meaningful price movements.

    Attributes:
        threshold: Cumulative change needed to trigger event
    """

    def __init__(self, threshold: Optional[float] = None):
        """
        Initialize CUSUM filter.

        Args:
            threshold: Event trigger threshold.
                      If None, uses daily volatility.
        """
        self.threshold = threshold
        logger.info(f"CUSUMEventFilter initialized: threshold={threshold}")

    def get_events(
        self,
        close: pd.Series,
        threshold: Optional[float] = None
    ) -> pd.DatetimeIndex:
        """
        Get event timestamps using CUSUM filter.

        Args:
            close: Closing price series
            threshold: Override default threshold

        Returns:
            DatetimeIndex of event timestamps
        """
        h = threshold or self.threshold

        # If no threshold, use daily volatility
        if h is None:
            daily_returns = close.pct_change().dropna()
            h = daily_returns.rolling(20).std().mean()
            logger.info(f"Auto-calculated threshold from volatility: {h:.6f}")

        logger.info(f"Filtering events with threshold={h:.6f}")

        try:
            events = cusum_filter(close, threshold=h)

            logger.info(f"Found {len(events)} CUSUM events from {len(close)} prices")

            return events

        except Exception as e:
            logger.error(f"Error in CUSUM filtering: {e}")
            raise

    def get_events_with_returns(
        self,
        close: pd.Series,
        threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Get events with associated price changes.

        Args:
            close: Closing price series
            threshold: Override default threshold

        Returns:
            DataFrame with columns:
            - event_time: Timestamp of event
            - price: Price at event
            - price_change: Change since last event
            - pct_change: Percentage change since last event
        """
        events = self.get_events(close, threshold)

        if len(events) == 0:
            return pd.DataFrame()

        # Get prices at events
        event_prices = close.loc[events]

        # Calculate changes
        price_changes = event_prices.diff()
        pct_changes = event_prices.pct_change()

        result = pd.DataFrame({
            'event_time': events,
            'price': event_prices.values,
            'price_change': price_changes.values,
            'pct_change': pct_changes.values
        })

        result.set_index('event_time', inplace=True)

        return result

    def estimate_optimal_threshold(
        self,
        close: pd.Series,
        target_events_per_day: int = 10,
        lookback_days: int = 20
    ) -> float:
        """
        Estimate optimal threshold based on desired event frequency.

        Args:
            close: Price series
            target_events_per_day: Desired number of events per trading day
            lookback_days: Period for volatility estimation

        Returns:
            Estimated threshold
        """
        # Calculate daily volatility
        daily_returns = close.pct_change().dropna()
        daily_vol = daily_returns.rolling(lookback_days).std().mean()

        # Estimate threshold
        # Higher target events = lower threshold
        threshold = daily_vol / np.sqrt(target_events_per_day)

        logger.info(f"Estimated threshold: {threshold:.6f} for "
                   f"{target_events_per_day} events/day")

        return threshold

    def adaptive_threshold(
        self,
        close: pd.Series,
        window: int = 20,
        multiplier: float = 1.0
    ) -> pd.Series:
        """
        Calculate adaptive threshold based on rolling volatility.

        Args:
            close: Price series
            window: Rolling window for volatility
            multiplier: Threshold multiplier

        Returns:
            Series of adaptive thresholds
        """
        daily_returns = close.pct_change()
        rolling_vol = daily_returns.rolling(window).std()

        adaptive_thresh = rolling_vol * multiplier

        logger.debug(f"Adaptive threshold: mean={adaptive_thresh.mean():.6f}, "
                    f"std={adaptive_thresh.std():.6f}")

        return adaptive_thresh

    def symmetric_cusum(
        self,
        close: pd.Series,
        threshold: Optional[float] = None
    ) -> dict:
        """
        Apply symmetric CUSUM to detect both upward and downward events.

        Args:
            close: Price series
            threshold: Event threshold

        Returns:
            Dictionary with 'up_events' and 'down_events'
        """
        if threshold is None:
            threshold = self.threshold or self.estimate_optimal_threshold(close)

        # Get returns
        returns = close.pct_change().dropna()

        # Upward CUSUM
        s_pos = pd.Series(0.0, index=returns.index)
        up_events = []

        for i in range(1, len(returns)):
            s_pos.iloc[i] = max(0, s_pos.iloc[i-1] + returns.iloc[i])
            if s_pos.iloc[i] > threshold:
                up_events.append(returns.index[i])
                s_pos.iloc[i] = 0

        # Downward CUSUM
        s_neg = pd.Series(0.0, index=returns.index)
        down_events = []

        for i in range(1, len(returns)):
            s_neg.iloc[i] = min(0, s_neg.iloc[i-1] + returns.iloc[i])
            if s_neg.iloc[i] < -threshold:
                down_events.append(returns.index[i])
                s_neg.iloc[i] = 0

        logger.info(f"Symmetric CUSUM: {len(up_events)} up events, "
                   f"{len(down_events)} down events")

        return {
            'up_events': pd.DatetimeIndex(up_events),
            'down_events': pd.DatetimeIndex(down_events)
        }
