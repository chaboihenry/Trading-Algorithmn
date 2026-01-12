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
                      If None, uses daily volatility * 2.5.
        """
        self.threshold = threshold
        logger.info(f"CUSUMEventFilter initialized: threshold={threshold}")

    def _calculate_threshold(self, prices: pd.Series) -> float:
        """
        Calculate CUSUM threshold to filter ~35-40% of events.

        Args:
            prices: Price series

        Returns:
            Threshold value (volatility * multiplier)
        """
        daily_returns = prices.pct_change().dropna()
        daily_vol = daily_returns.std()

        # Multiply by 2.5 to get ~35% filter rate (was ~1.0)
        threshold = daily_vol * 2.5

        logger.info(f"CUSUM threshold: {threshold:.6f} (vol={daily_vol:.6f})")
        return threshold

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

        # If no threshold, use daily volatility * 2.5
        if h is None:
            h = self._calculate_threshold(close)
            logger.info(f"Auto-calculated threshold from volatility")

        # Convert scalar to Series if needed
        if isinstance(h, (int, float)):
            h = pd.Series(h, index=close.index)

        logger.info(f"Filtering events with CUSUM")

        try:
            events = cusum_filter_events_dynamic_threshold(
                prices=close,
                threshold=h
            )

            # Log filter rate with warnings
            filter_rate = len(events) / len(close)
            logger.info(f"CUSUM filter rate: {filter_rate*100:.1f}% ({len(events)}/{len(close)})")

            if filter_rate > 0.50:
                logger.warning(f"⚠️ Filter rate {filter_rate:.1%} too high - increase threshold")
            if filter_rate < 0.20:
                logger.warning(f"⚠️ Filter rate {filter_rate:.1%} too low - decrease threshold")

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
            Dictionary with 'events' and 'down_events'
        """
        h = threshold or self.threshold

        if h is None:
            h = self._calculate_threshold(close)
            logger.info(f"Auto-calculated threshold for symmetric CUSUM")

        logger.info(f"Running symmetric CUSUM with threshold={h:.6f}")

        try:
            # RiskLabAI's symmetric CUSUM returns combined events
            events = symmetric_cusum_filter(prices=close, threshold=h)

            # Log filter rate with warnings
            filter_rate = len(events) / len(close)
            logger.info(f"Symmetric CUSUM filter rate: {filter_rate*100:.1f}% ({len(events)}/{len(close)})")

            if filter_rate > 0.50:
                logger.warning(f"⚠️ Filter rate {filter_rate:.1%} too high - increase threshold")
            if filter_rate < 0.20:
                logger.warning(f"⚠️ Filter rate {filter_rate:.1%} too low - decrease threshold")

            return {
                'events': events,
                'threshold': h
            }

        except Exception as e:
            logger.error(f"Error in symmetric CUSUM: {e}")
            raise
