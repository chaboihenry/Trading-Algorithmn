"""
RiskLabAI Data Structures Module

Implements alternative bar types that sample based on market activity
rather than clock time. Uses the actual RiskLabAI library API.

Bar Types:
- Standard Bars: Dollar, volume, tick bars
- Imbalance Bars: Fixed imbalance bars
- Run Bars: Fixed run bars
"""

from RiskLabAI.data.structures import StandardBars, FixedImbalanceBars, FixedRunBars
import pandas as pd
import numpy as np
from typing import Literal, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class BarGenerator:
    """
    Generates information-driven bars from tick/trade data using RiskLabAI.

    Why this matters:
    - Time bars oversample during quiet periods, undersample during volatile periods
    - Information-driven bars sample based on actual market activity
    - Result: More normal returns, better ML model performance

    Attributes:
        bar_type: Type of bar to generate
        threshold: Sampling threshold (varies by bar type)
    """

    def __init__(
        self,
        bar_type: Literal[
            'dollar', 'volume', 'tick',
            'dollar_imbalance', 'volume_imbalance', 'tick_imbalance',
            'dollar_run', 'volume_run', 'tick_run'
        ] = 'dollar',
        threshold: float = 50000
    ):
        """
        Initialize bar generator.

        Args:
            bar_type: Which bar type to generate
            threshold: Sampling threshold
                - For dollar bars: dollars per bar (e.g., 50,000)
                - For volume bars: shares per bar (e.g., 10,000)
                - For tick bars: trades per bar (e.g., 100)
        """
        self.bar_type = bar_type
        self.threshold = threshold

        # Map bar type to RiskLabAI class and type string
        self.bar_class_map = {
            'dollar': (StandardBars, 'dollar_bars'),
            'volume': (StandardBars, 'volume_bars'),
            'tick': (StandardBars, 'tick_bars'),
            'dollar_imbalance': (FixedImbalanceBars, 'dollar_imbalance'),
            'volume_imbalance': (FixedImbalanceBars, 'volume_imbalance'),
            'tick_imbalance': (FixedImbalanceBars, 'tick_imbalance'),
            'dollar_run': (FixedRunBars, 'dollar_run'),
            'volume_run': (FixedRunBars, 'volume_run'),
            'tick_run': (FixedRunBars, 'tick_run'),
        }

        if bar_type not in self.bar_class_map:
            raise ValueError(f"Invalid bar_type: {bar_type}")

        bar_class, bar_type_str = self.bar_class_map[bar_type]
        self.bar_generator = bar_class(bar_type=bar_type_str, threshold=threshold)

        logger.info(f"BarGenerator initialized: type={bar_type}, threshold={threshold}")

    def generate_bars(self, tick_data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert tick data to information-driven bars.

        Args:
            tick_data: DataFrame with columns [date_time, price, volume]
                      Expected format from RiskLabAI: List of tuples/arrays

        Returns:
            DataFrame with OHLCV bars
        """
        logger.info(f"Generating {self.bar_type} bars from {len(tick_data)} ticks")

        if tick_data.empty:
            logger.warning("Empty tick data provided")
            return pd.DataFrame()

        try:
            # Convert DataFrame to list of tuples format expected by RiskLabAI
            # Expected: [(date_time, price, volume), ...]
            if isinstance(tick_data, pd.DataFrame):
                # Ensure we have the required columns
                required_cols = ['date_time', 'price', 'volume']
                if not all(col in tick_data.columns for col in required_cols):
                    # Try alternative column names
                    if 'close' in tick_data.columns:
                        tick_data = tick_data.rename(columns={'close': 'price'})
                    if 'timestamp' in tick_data.columns:
                        tick_data = tick_data.rename(columns={'timestamp': 'date_time'})

                # Convert to list of tuples
                data_list = tick_data[['date_time', 'price', 'volume']].values.tolist()
            else:
                data_list = tick_data

            # Generate bars using RiskLabAI
            bars_list = self.bar_generator.construct_bars_from_data(data_list)

            # Convert to DataFrame
            if len(bars_list) == 0:
                logger.warning("No bars generated")
                return pd.DataFrame()

            # RiskLabAI returns: [date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value]
            bars_df = pd.DataFrame(bars_list, columns=[
                'date_time', 'open', 'high', 'low', 'close', 'volume',
                'cum_buy_volume', 'cum_ticks', 'cum_dollar_value'
            ])

            # Set index
            bars_df['date_time'] = pd.to_datetime(bars_df['date_time'])
            bars_df.set_index('date_time', inplace=True)

            logger.info(f"Generated {len(bars_df)} bars")
            return bars_df

        except Exception as e:
            logger.error(f"Error generating bars: {e}")
            raise

    def estimate_threshold(
        self,
        tick_data: pd.DataFrame,
        target_bars_per_day: int = 50
    ) -> float:
        """
        Estimate optimal threshold for bar generation.

        Args:
            tick_data: Sample tick data
            target_bars_per_day: Desired number of bars per trading day

        Returns:
            Estimated threshold value
        """
        if 'price' in tick_data.columns and 'volume' in tick_data.columns:
            total_value = (tick_data['price'] * tick_data['volume']).sum()
        elif 'cum_dollar_value' in tick_data.columns:
            total_value = tick_data['cum_dollar_value'].iloc[-1]
        else:
            raise ValueError("Cannot estimate threshold: missing price/volume data")

        # Assume data is from one trading day
        estimated_threshold = total_value / target_bars_per_day

        logger.info(f"Estimated threshold: {estimated_threshold:,.0f}")
        return estimated_threshold
