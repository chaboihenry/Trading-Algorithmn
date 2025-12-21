"""
Fractional Differentiation for Stationary Features

The Problem:
- ML models need stationary features (constant mean/variance)
- Price series are non-stationary (trending)
- Normal differentiation (returns) loses too much information

The Solution:
- Fractionally differentiate with d < 1
- Find minimum d that achieves stationarity
- Preserve maximum memory while being stationary

d=0: Original price (non-stationary, full memory)
d=1: Returns (stationary, almost no memory)
d=0.4: Sweet spot (stationary, retains memory)
"""

from RiskLabAI.features import (
    frac_diff_ffd,
    get_opt_d,
    plot_min_ffd
)
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FractionalDifferentiator:
    """
    Applies fractional differentiation to price series.

    Why this matters:
    - Price is non-stationary → violates ML assumptions
    - Returns throw away information → poor predictions
    - Fractional diff finds the sweet spot

    The math (simplified):
    - FFD uses fixed-width window to approximate infinite weights
    - Weight decay: w_k = w_{k-1} * (k-1-d) / k
    - Threshold tau cuts off weights below certain value

    Attributes:
        d: Differentiation order (0 < d < 1, typically 0.3-0.6)
        tau: Weight cutoff threshold
    """

    def __init__(self, d: Optional[float] = None, tau: float = 1e-5):
        """
        Initialize fractional differentiator.

        Args:
            d: Differentiation order. If None, auto-calculated.
            tau: Cutoff threshold for weights
        """
        self.d = d
        self.tau = tau
        self._optimal_d = None

        logger.info(f"FractionalDifferentiator initialized: d={d}, tau={tau}")

    def find_optimal_d(
        self,
        series: pd.Series,
        max_d: float = 1.0,
        p_value: float = 0.05
    ) -> float:
        """
        Find minimum d that achieves stationarity.

        Uses ADF test to check stationarity at each d value.

        Args:
            series: Price series to analyze
            max_d: Maximum d to test
            p_value: Significance level for ADF test

        Returns:
            Optimal d value
        """
        logger.info(f"Finding optimal d for series of length {len(series)}")

        try:
            self._optimal_d = get_opt_d(
                series,
                d_range=np.arange(0, max_d + 0.1, 0.1),
                p_value=p_value
            )

            logger.info(f"Optimal d found: {self._optimal_d:.3f}")
            return self._optimal_d

        except Exception as e:
            logger.error(f"Error finding optimal d: {e}")
            # Default to a reasonable value
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
            result = frac_diff_ffd(
                series,
                d=d,
                thres=self.tau
            )

            logger.debug(f"Transform complete: {len(result)} values")
            return result

        except Exception as e:
            logger.error(f"Error in fractional differentiation: {e}")
            raise

    def transform_multiple(
        self,
        df: pd.DataFrame,
        columns: Optional[list] = None,
        d_per_column: Optional[dict] = None
    ) -> pd.DataFrame:
        """
        Apply fractional differentiation to multiple columns.

        Args:
            df: DataFrame with price series
            columns: List of columns to transform (None = all)
            d_per_column: Dictionary mapping column names to d values
                         (None = use self.d for all)

        Returns:
            DataFrame with fractionally differentiated series
        """
        if columns is None:
            columns = df.columns.tolist()

        result = pd.DataFrame(index=df.index)

        for col in columns:
            logger.debug(f"Processing column: {col}")

            # Get d value for this column
            if d_per_column and col in d_per_column:
                d_value = d_per_column[col]
            else:
                d_value = self.d

            # Save current d and set column-specific d
            original_d = self.d
            self.d = d_value

            # Transform
            result[col] = self.transform(df[col])

            # Restore original d
            self.d = original_d

        return result

    def check_stationarity(self, series: pd.Series, p_value: float = 0.05) -> dict:
        """
        Check if a series is stationary using ADF test.

        Args:
            series: Time series to test
            p_value: Significance level

        Returns:
            Dictionary with test results
        """
        from statsmodels.tsa.stattools import adfuller

        result = adfuller(series.dropna(), autolag='AIC')

        is_stationary = result[1] < p_value

        stats = {
            'adf_statistic': result[0],
            'p_value': result[1],
            'is_stationary': is_stationary,
            'critical_values': result[4],
            'used_lag': result[2]
        }

        logger.info(f"Stationarity test: p-value={result[1]:.4f}, "
                   f"stationary={is_stationary}")

        return stats

    def optimize_and_transform(
        self,
        series: pd.Series,
        max_d: float = 1.0,
        p_value: float = 0.05
    ) -> tuple[pd.Series, float]:
        """
        Find optimal d and transform in one step.

        Args:
            series: Price series
            max_d: Maximum d to test
            p_value: Significance level for ADF test

        Returns:
            Tuple of (transformed_series, optimal_d)
        """
        optimal_d = self.find_optimal_d(series, max_d, p_value)
        transformed = self.transform(series)

        return transformed, optimal_d
