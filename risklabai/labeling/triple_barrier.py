"""
Triple-Barrier Method for Labeling

Instead of labeling based on fixed-time returns (which ignores volatility),
the triple-barrier method sets dynamic barriers using RiskLabAI's implementation.

Barriers:
- Upper barrier (take-profit): Triggered when price rises X%
- Lower barrier (stop-loss): Triggered when price falls Y%
- Vertical barrier (timeout): Triggered after T periods

Labels: +1 (hit upper), -1 (hit lower), 0 (hit vertical/timeout)

================================================================================
CRITICAL: Understanding Triple-Barrier Parameters
================================================================================

Triple Barrier Labeling Parameters:
-----------------------------------
- profit_taking_mult: Volatility MULTIPLIER for upper barrier (e.g., 2.0 = 2x daily vol)
- stop_loss_mult:     Volatility MULTIPLIER for lower barrier (e.g., 2.0 = 2x daily vol)
- max_holding_period: Maximum bars before timeout (neutral label)

IMPORTANT - These are NOT percentages, they are MULTIPLIERS:
------------------------------------------------------------
profit_taking_mult=2.0 means "2 x daily_volatility profit target"

Example with 1% daily volatility:
- profit_taking_mult = 2.0 → profit barrier at +2.0% (2 x 1%)
- stop_loss_mult = 2.0     → loss barrier at -2.0% (2 x 1%)

Example with 3% daily volatility (volatile market):
- profit_taking_mult = 2.0 → profit barrier at +6.0% (2 x 3%)
- stop_loss_mult = 2.0     → loss barrier at -6.0% (2 x 3%)

This adaptive approach is superior because:
1. Barriers automatically widen in volatile markets (avoiding noise)
2. Barriers automatically tighten in calm markets (capturing small moves)
3. Risk-reward ratio stays constant across market regimes

Understanding max_holding_period:
---------------------------------
This parameter controls how long we wait before giving up (neutral label). A shorter period may lead to more neutral labels as there's less time for prices to hit the profit-taking or stop-loss barriers.
================================================================================
"""

from RiskLabAI.data.labeling import (
    triple_barrier,
    vertical_barrier,
    daily_volatility_with_log_returns
)
import pandas as pd
import numpy as np
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class TripleBarrierLabeler:
    """
    Labels price data using the triple-barrier method via RiskLabAI.

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
        max_holding_period: int = 30,
        volatility_lookback: int = 20
    ):
        """
        Initialize triple-barrier labeler.

        Args:
            profit_taking_mult: Upper barrier = volatility * this value
            stop_loss_mult: Lower barrier = volatility * this value
            max_holding_period: Maximum periods before vertical barrier
            volatility_lookback: Days for volatility calculation

        IMPORTANT - Parameter Interpretation:
        =====================================
        profit_taking_mult and stop_loss_mult are MULTIPLIERS of volatility, NOT percentages.

        Example:
        - If daily volatility is 1%, profit_taking_mult=2.0 means 2% profit target
        - If daily volatility is 2%, profit_taking_mult=2.0 means 4% profit target

        This adaptive approach ensures barriers scale with market conditions:
        - Volatile periods → wider barriers (avoid premature exits)
        - Calm periods → tighter barriers (capture smaller moves)

        max_holding_period controls neutral label rate:
        - Too small (e.g., 10) → 52% neutral labels (insufficient time to hit barriers)
        - Optimal (e.g., 30) → 30-35% neutral labels (healthy balance)
        - Too large (e.g., 100) → low neutral rate but stale signals
        """
        self.profit_taking_mult = profit_taking_mult
        self.stop_loss_mult = stop_loss_mult
        self.max_holding_period = max_holding_period
        self.volatility_lookback = volatility_lookback

        logger.info(f"TripleBarrierLabeler initialized: pt={profit_taking_mult}, "
                   f"sl={stop_loss_mult}, max_hold={max_holding_period}")

    def get_daily_volatility(self, close: pd.Series, span: Optional[int] = None) -> pd.Series:
        """
        Calculate daily volatility using exponentially weighted standard deviation.

        This is a robust alternative to RiskLabAI's daily_volatility_with_log_returns
        that handles various data formats better.

        Args:
            close: Series of closing prices
            span: Lookback period (default: self.volatility_lookback)

        Returns:
            Series of daily volatility estimates
        """
        if span is None:
            span = self.volatility_lookback

        # Calculate log returns
        returns = np.log(close / close.shift(1)).dropna()

        # Calculate exponentially weighted volatility
        vol = returns.ewm(span=span).std()

        # Reindex to match close series (fill forward for missing values)
        vol = vol.reindex(close.index).ffill()

        # Fill any remaining NaNs at the beginning with the first valid value
        vol = vol.bfill()

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
            DataFrame with triple-barrier labels
        """
        logger.info(f"Labeling {len(close)} price points")

        # If no events provided, use all timestamps
        if events is None:
            logger.debug("No events provided, using all timestamps as events")
            events = pd.DataFrame(index=close.index)
        else:
            # CRITICAL FIX: Ensure event timestamps exist in close index
            # RiskLabAI's daily_volatility function expects events to be a subset of close
            valid_times = events.index.intersection(close.index)
            if len(valid_times) < len(events):
                logger.warning(f"Filtering events: {len(events)} -> {len(valid_times)} (matching close index)")
            events = events.loc[valid_times]

            if len(events) == 0:
                logger.error("No valid event timestamps found in close index!")
                return pd.DataFrame()

        # Get daily volatility (now that events match close index)
        daily_vol = self.get_daily_volatility(close)

        # Add vertical barrier if not present
        if 't1' not in events.columns:
            events['t1'] = vertical_barrier(
                close=close,
                time_events=events.index,
                number_days=self.max_holding_period
            )

        # Prepare target (volatility-based barriers)
        target = daily_vol.reindex(events.index)
        target = target[target > 0]  # Remove zero volatility periods

        # Align events with target
        events = events.loc[target.index]

        # Create events DataFrame with required columns for triple_barrier
        # RiskLabAI expects specific column names: 'End Time', 'Base Width', 'Side'
        events_for_labeling = pd.DataFrame(index=events.index)
        events_for_labeling['End Time'] = events['t1']
        events_for_labeling['Base Width'] = target
        events_for_labeling['Side'] = side if side is not None else pd.Series(1, index=events.index)

        logger.debug(f"Processing {len(events_for_labeling)} events")

        try:
            # Apply triple-barrier labeling
            # ptsl = [profit_taking, stop_loss] as multiples of target
            ptsl = [self.profit_taking_mult, self.stop_loss_mult]

            # molecule is list of timestamps to process
            molecule = list(events_for_labeling.index)

            labels = triple_barrier(
                close=close,
                events=events_for_labeling,
                ptsl=ptsl,
                molecule=molecule
            )

            # Add 'bin' and 'ret' columns based on which barrier was hit
            # bin: -1 (stop loss), 0 (vertical/timeout), 1 (profit taking)
            # ret: actual return at barrier touch

            labels['ret'] = 0.0
            labels['bin'] = 0

            # Track statistics for force_directional debugging
            barrier_stats = {
                'profit': 0, 'loss': 0, 'timeout': 0,
                'timeout_positive': 0, 'timeout_negative': 0, 'timeout_flat': 0,
                'no_threshold': 0, 'missing_end_time': 0, 'end_time_not_in_close': 0,
                'total_labels': len(labels)
            }

            for idx in labels.index:
                # Get start and end prices
                start_price = close.loc[idx]
                end_time = labels.loc[idx, 'End Time']

                # Handle missing/invalid end times
                actual_end_time = None
                if pd.isna(end_time):
                    barrier_stats['missing_end_time'] += 1
                    # Calculate intended vertical barrier time (max_holding_period bars from event)
                    future_bars = close.index[close.index > idx]
                    if len(future_bars) >= self.max_holding_period:
                        actual_end_time = future_bars[self.max_holding_period - 1]
                    elif len(future_bars) > 0:
                        # Less than max_holding_period bars available, use last available
                        actual_end_time = future_bars[-1]
                    else:
                        # No future data, skip this label
                        continue
                elif end_time not in close.index:
                    barrier_stats['end_time_not_in_close'] += 1
                    # Find closest available bar BEFORE or AT the intended end_time
                    # Don't go beyond max_holding_period from event start
                    future_bars = close.index[close.index > idx]
                    if len(future_bars) >= self.max_holding_period:
                        actual_end_time = future_bars[self.max_holding_period - 1]
                    elif len(future_bars) > 0:
                        actual_end_time = future_bars[-1]
                    else:
                        continue
                else:
                    actual_end_time = end_time

                # Now we have a valid end_time (either original or fallback)
                if actual_end_time is not None and actual_end_time in close.index:
                    end_price = close.loc[actual_end_time]
                    ret = (end_price / start_price) - 1
                    labels.loc[idx, 'ret'] = ret

                    # Determine which barrier was hit
                    threshold = target.loc[idx] if idx in target.index else 0
                    if threshold > 0:
                        profit_barrier = self.profit_taking_mult * threshold
                        loss_barrier = -self.stop_loss_mult * threshold

                        if ret >= profit_barrier:
                            labels.loc[idx, 'bin'] = 1  # Profit taking
                            barrier_stats['profit'] += 1
                        elif ret <= loss_barrier:
                            labels.loc[idx, 'bin'] = -1  # Stop loss
                            barrier_stats['loss'] += 1
                        else:
                            # Vertical barrier / timeout hit - label as neutral (0)
                            # Neutral represents timeout and can be used as trading signal
                            barrier_stats['timeout'] += 1
                            labels.loc[idx, 'bin'] = 0
                    else:
                        barrier_stats['no_threshold'] += 1

            # Log labeling statistics
            logger.info(f"Triple-barrier labeling complete: {len(labels)} labels generated")
            logger.info(f"  Profit hits: {barrier_stats['profit']}")
            logger.info(f"  Loss hits: {barrier_stats['loss']}")
            logger.info(f"  Timeouts (neutral): {barrier_stats['timeout']}")
            if 'bin' in labels.columns:
                logger.info(f"Label distribution: {labels['bin'].value_counts().to_dict()}")

            # Calculate and log label distribution percentages
            if 'bin' in labels.columns and len(labels) > 0:
                n_total = len(labels)
                n_long = (labels["bin"] == 1).sum()
                n_short = (labels["bin"] == -1).sum()
                n_neutral = (labels["bin"] == 0).sum()

                logger.info("")
                logger.info("=" * 60)
                logger.info("LABEL DISTRIBUTION ANALYSIS")
                logger.info("=" * 60)
                logger.info(f"  Long (profit):   {n_long:5d} ({n_long/n_total*100:5.1f}%)")
                logger.info(f"  Short (loss):    {n_short:5d} ({n_short/n_total*100:5.1f}%)")
                logger.info(f"  Neutral (timeout): {n_neutral:5d} ({n_neutral/n_total*100:5.1f}%)")
                logger.info(f"  Total:           {n_total:5d}")
                logger.info("=" * 60)

                # Warn if neutral rate is too high
                neutral_pct = n_neutral / n_total
                
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
            'label_counts': labels['bin'].value_counts().to_dict() if 'bin' in labels.columns else {},
            'label_percentages': (labels['bin'].value_counts(normalize=True) * 100).to_dict() if 'bin' in labels.columns else {},
            'mean_return': labels['ret'].mean() if 'ret' in labels.columns else None,
            'std_return': labels['ret'].std() if 'ret' in labels.columns else None,
            'mean_target': labels['trgt'].mean() if 'trgt' in labels.columns else None
        }

        return stats
