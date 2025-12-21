"""
Fractional Differentiation for Stationary Features

Uses RiskLabAI's fractional differentiation implementation.

d=0: Original price (non-stationary, full memory)
d=1: Returns (stationary, almost no memory)
d≈0.4: Sweet spot (stationary, retains memory)
"""

from RiskLabAI.data.differentiation import (
    fractional_difference_fixed,
    find_optimal_ffd_simple
)
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FractionalDifferentiator:
    """
    Applies fractional differentiation using RiskLabAI.

    Attributes:
        d: Differentiation order (0 < d < 1, typically 0.3-0.6)
        threshold: Weight cutoff threshold
    """

    def __init__(self, d: Optional[float] = None, threshold: float = 0.01):
        """
        Initialize fractional differentiator.

        Args:
            d: Differentiation order. If None, auto-calculated.
            threshold: Cutoff threshold for weights
        """
        self.d = d
        self.threshold = threshold
        self._optimal_d = None

        logger.info(f"FractionalDifferentiator initialized: d={d}, threshold={threshold}")

    def find_optimal_d(
        self,
        series: pd.Series,
        max_d: float = 1.0,
        p_value: float = 0.05
    ) -> float:
        """
        Find minimum d that achieves stationarity.

        Args:
            series: Price series to analyze
            max_d: Maximum d to test
            p_value: Significance level for ADF test

        Returns:
            Optimal d value
        """
        logger.info(f"Finding optimal d for series of length {len(series)}")

        try:
            # Convert to DataFrame (RiskLabAI expects DataFrame)
            df = pd.DataFrame(series)

            result_df = find_optimal_ffd_simple(
                input_series=df,
                p_value_threshold=p_value
            )

            # Extract the optimal d value from the result
            # find_optimal_ffd_simple returns a DataFrame with the optimal d column
            if 'd_value' in result_df.columns:
                self._optimal_d = result_df['d_value'].iloc[0]
            elif 'Optimal d' in result_df.columns:
                self._optimal_d = result_df['Optimal d'].iloc[0]
            else:
                # If we can't find the d value, use default
                self._optimal_d = 0.4
                logger.warning(f"Could not extract d value, using default {self._optimal_d}")

            logger.info(f"Optimal d found: {self._optimal_d:.3f}")
            return self._optimal_d

        except Exception as e:
            logger.error(f"Error finding optimal d: {e}")
            self._optimal_d = 0.4
            logger.warning(f"Using default d={self._optimal_d}")
            return self._optimal_d

    def transform(self, series: pd.Series) -> pd.Series:
        """
        Apply fractional differentiation.

        Args:
            series: Price series

        Returns:
            Fractionally differentiated series
        """
        d = self.d if self.d is not None else self._optimal_d

        if d is None:
            logger.info("No d specified, finding optimal d...")
            d = self.find_optimal_d(series)

        logger.debug(f"Transforming series with d={d:.3f}")

        try:
            # Convert to DataFrame (RiskLabAI expects DataFrame)
            df = pd.DataFrame(series)
            
            result_df = fractional_difference_fixed(
                series=df,
                degree=d,
                threshold=self.threshold
            )

            # Convert back to Series
            result = result_df.iloc[:, 0]
            result.index = series.index[:len(result)]

            logger.debug(f"Transform complete: {len(result)} values")
            return result

        except Exception as e:
            logger.error(f"Error in fractional differentiation: {e}")
            raise
