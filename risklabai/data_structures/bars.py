"""
RiskLabAI Data Structures Module

Implements alternative bar types that sample based on market activity
rather than clock time. These have better statistical properties
(closer to normal distribution, less serial correlation).

Types:
- Dollar Bars: Sample every $X traded
- Volume Bars: Sample every X shares traded
- Tick Bars: Sample every X trades
- Imbalance Bars: Sample when buy/sell imbalance exceeds threshold
- Run Bars: Sample when run of buys/sells is unusually long
"""

from RiskLabAI.data import (
    dollar_bars,
    volume_bars,
    tick_bars,
    tick_imbalance_bars,
    volume_imbalance_bars,
    dollar_imbalance_bars,
    tick_run_bars,
    volume_run_bars,
    dollar_run_bars
)
import pandas as pd
import numpy as np
from typing import Literal, Optional
import logging

logger = logging.getLogger(__name__)


class BarGenerator:
    """
    Generates information-driven bars from tick/trade data.

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
        ] = 'dollar_imbalance',
        threshold: Optional[float] = None
    ):
        """
        Initialize bar generator.

        Args:
            bar_type: Which bar type to generate
            threshold: Sampling threshold. If None, auto-calculated.
                - For dollar bars: dollars per bar (e.g., 1_000_000)
                - For volume bars: shares per bar (e.g., 10_000)
                - For tick bars: trades per bar (e.g., 100)
                - For imbalance/run bars: auto-calculated from expected imbalance
        """
        self.bar_type = bar_type
        self.threshold = threshold

        logger.info(f"BarGenerator initialized: type={bar_type}, threshold={threshold}")

    def generate_bars(self, tick_data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert tick data to information-driven bars.

        Args:
            tick_data: DataFrame with columns [datetime, price, volume]
                      Index should be datetime

        Returns:
            DataFrame with OHLCV bars based on information sampling
        """
        logger.info(f"Generating {self.bar_type} bars from {len(tick_data)} ticks")

        # Validate input data
        if tick_data.empty:
            logger.warning("Empty tick data provided")
            return pd.DataFrame()

        # Map bar type to RiskLabAI function
        bar_functions = {
            'dollar': dollar_bars,
            'volume': volume_bars,
            'tick': tick_bars,
            'dollar_imbalance': dollar_imbalance_bars,
            'volume_imbalance': volume_imbalance_bars,
            'tick_imbalance': tick_imbalance_bars,
            'dollar_run': dollar_run_bars,
            'volume_run': volume_run_bars,
            'tick_run': tick_run_bars,
        }

        bar_func = bar_functions[self.bar_type]

        # Generate bars
        try:
            if self.threshold is not None:
                bars = bar_func(tick_data, threshold=self.threshold)
            else:
                # Auto-calculate threshold if not provided
                bars = bar_func(tick_data)

            logger.info(f"Generated {len(bars)} bars")
            return bars

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
        if 'dollar_value' in tick_data.columns:
            total_value = tick_data['dollar_value'].sum()
        elif 'price' in tick_data.columns and 'volume' in tick_data.columns:
            total_value = (tick_data['price'] * tick_data['volume']).sum()
        else:
            raise ValueError("Cannot estimate threshold: missing price/volume data")

        # Assume data is from one trading day
        estimated_threshold = total_value / target_bars_per_day

        logger.info(f"Estimated threshold: ${estimated_threshold:,.0f}")
        return estimated_threshold
