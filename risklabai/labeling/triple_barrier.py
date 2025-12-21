"""
Triple-Barrier Method for Labeling

Instead of labeling based on fixed-time returns (which ignores volatility),
the triple-barrier method sets dynamic barriers:
- Upper barrier (take-profit): Triggered when price rises X%
- Lower barrier (stop-loss): Triggered when price falls Y%
- Vertical barrier (timeout): Triggered after T periods

Labels: +1 (hit upper), -1 (hit lower), 0 (hit vertical/timeout)

This creates labels that match how real traders actually operate.
"""

from RiskLabAI.labeling import (
    get_events,
    get_bins,
    drop_labels,
    get_vertical_barrier,
    get_3_barriers,
    get_daily_vol
)
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TripleBarrierLabeler:
    """
    Labels price data using the triple-barrier method.

    The beauty of this approach:
    1. Barriers adapt to volatility (wider in volatile periods)
    2. Timeout prevents holding losing positions forever
    3. Labels match real trading mechanics (stop-loss, take-profit)

    Attributes:
        profit_taking_mult: Multiplier for upper barrier (based on volatility)
        stop_loss_mult: Multiplier for lower barrier (based on volatility)
        max_holding_period: Maximum bars before timeout
        volatility_lookback: Lookback for volatility estimation
    """

    def __init__(
        self,
        profit_taking_mult: float = 2.0,
        stop_loss_mult: float = 2.0,
        max_holding_period: int = 10,
        volatility_lookback: int = 20
    ):
        """
        Initialize triple-barrier labeler.

        Args:
            profit_taking_mult: Upper barrier = volatility * this value
            stop_loss_mult: Lower barrier = volatility * this value
            max_holding_period: Maximum periods before vertical barrier
            volatility_lookback: Days for volatility calculation
        """
        self.profit_taking_mult = profit_taking_mult
        self.stop_loss_mult = stop_loss_mult
        self.max_holding_period = max_holding_period
        self.volatility_lookback = volatility_lookback

        logger.info(f"TripleBarrierLabeler initialized: pt={profit_taking_mult}, "
                   f"sl={stop_loss_mult}, max_hold={max_holding_period}")

    def get_daily_volatility(self, close: pd.Series, span: Optional[int] = None) -> pd.Series:
        """
        Calculate daily volatility using exponentially weighted std dev.

        Args:
            close: Series of closing prices
            span: Lookback period (default: self.volatility_lookback)

        Returns:
            Series of daily volatility estimates
        """
        if span is None:
            span = self.volatility_lookback

        # Use RiskLabAI's get_daily_vol function
        vol = get_daily_vol(close=close, span=span)

        logger.debug(f"Volatility calculated: mean={vol.mean():.4f}, std={vol.std():.4f}")
        return vol

    def label(
        self,
        close: pd.Series,
        events: Optional[pd.DataFrame] = None,
        side: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Apply triple-barrier labeling to price series.

        Args:
            close: Series of closing prices
            events: Optional pre-computed event timestamps (from CUSUM filter)
                   Should have 't1' column with vertical barrier times
            side: Optional side predictions from primary model (+1 long, -1 short)
                 When provided, enables meta-labeling

        Returns:
            DataFrame with columns:
            - t1: Time when first barrier was touched
            - ret: Return at barrier touch
            - trgt: Target (volatility at event time)
            - bin: Label (-1, 0, +1)
        """
        logger.info(f"Labeling {len(close)} price points")

        # Get daily volatility
        daily_vol = self.get_daily_volatility(close)

        # If no events provided, use all timestamps
        if events is None:
            logger.debug("No events provided, using all timestamps")
            events = pd.DataFrame(index=close.index)
            events['t1'] = get_vertical_barrier(
                close.index,
                close,
                num_days=self.max_holding_period
            )
        elif 't1' not in events.columns:
            # Add vertical barrier if not present
            events['t1'] = get_vertical_barrier(
                events.index,
                close,
                num_days=self.max_holding_period
            )

        # Prepare target (volatility)
        target = daily_vol.reindex(events.index)
        target = target[target > 0]  # Remove zero volatility periods

        # Align events with target
        events = events.loc[target.index]

        logger.debug(f"Processing {len(events)} events")

        try:
            # Get triple barrier events
            barriers = get_3_barriers(
                close=close,
                events=events,
                pt_sl=[self.profit_taking_mult, self.stop_loss_mult],
                target=target,
                min_ret=0.01,  # Minimum return threshold
                num_threads=1,
                vertical_barrier_times=events['t1'],
                side=side
            )

            # Get binary labels
            labels = get_bins(barriers, close)

            # Drop rare labels if severely imbalanced
            labels = drop_labels(labels, min_pct=0.05)

            logger.info(f"Generated {len(labels)} labels")
            logger.info(f"Label distribution: {labels['bin'].value_counts().to_dict()}")

            return labels

        except Exception as e:
            logger.error(f"Error in triple-barrier labeling: {e}")
            raise

    def get_label_stats(self, labels: pd.DataFrame) -> dict:
        """
        Get statistics about labels.

        Args:
            labels: Output from self.label()

        Returns:
            Dictionary with label statistics
        """
        stats = {
            'total_labels': len(labels),
            'label_counts': labels['bin'].value_counts().to_dict(),
            'label_percentages': (labels['bin'].value_counts(normalize=True) * 100).to_dict(),
            'mean_return': labels['ret'].mean(),
            'std_return': labels['ret'].std(),
            'mean_target': labels['trgt'].mean() if 'trgt' in labels.columns else None
        }

        return stats
