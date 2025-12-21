"""
CUSUM Filter for Event Sampling

Uses RiskLabAI's CUSUM filter implementation for event-driven sampling.
"""

from RiskLabAI.data.labeling import cusum_filter_events_dynamic_threshold, symmetric_cusum_filter
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class CUSUMEventFilter:
    """
    Filters events using CUSUM (Cumulative Sum) algorithm via RiskLabAI.

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
            h = daily_returns.rolling(20).std()
            logger.info(f"Auto-calculated dynamic threshold from volatility")
        elif isinstance(h, (int, float)):
            # Convert scalar to Series
            h = pd.Series(h, index=close.index)

        logger.info(f"Filtering events with CUSUM")

        try:
            events = cusum_filter_events_dynamic_threshold(
                prices=close,
                threshold=h
            )

            logger.info(f"Found {len(events)} CUSUM events from {len(close)} prices")
            return events

        except Exception as e:
            logger.error(f"Error in CUSUM filtering: {e}")
            raise

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
        h = threshold or self.threshold

        if h is None:
            daily_returns = close.pct_change().dropna()
            h = daily_returns.rolling(20).std().mean()

        logger.info(f"Running symmetric CUSUM with threshold={h:.6f}")

        try:
            # RiskLabAI's symmetric CUSUM returns combined events
            events = symmetric_cusum_filter(prices=close, threshold=h)

            logger.info(f"Symmetric CUSUM: {len(events)} total events")

            return {
                'events': events,
                'threshold': h
            }

        except Exception as e:
            logger.error(f"Error in symmetric CUSUM: {e}")
            raise
