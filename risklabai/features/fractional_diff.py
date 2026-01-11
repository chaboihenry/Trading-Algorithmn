"""
Fractional Differentiation for Stationary Features

Uses RiskLabAI's fractional differentiation implementation with robust
optimal d finding using ADF tests.

d=0: Original price (non-stationary, full memory)
d=1: Returns (stationary, almost no memory)
d≈0.2-0.8: Sweet spot (stationary, retains memory) - varies by symbol

The optimal d is found by testing values from 0.0 to 1.0 in steps of 0.05,
selecting the minimum d that achieves stationarity (ADF p-value < 0.05).
"""

from RiskLabAI.data.differentiation import fractional_difference_fixed
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

    def _apply_frac_diff(self, series: pd.Series, d: float) -> pd.Series:
        """
        Apply fractional differentiation for a given d value.

        Args:
            series: Price series
            d: Differentiation order

        Returns:
            Fractionally differentiated series
        """
        try:
            df = pd.DataFrame(series)
            result_df = fractional_difference_fixed(
                series=df,
                degree=d,
                threshold=self.threshold
            )
            result = result_df.iloc[:, 0]
            result.index = series.index[:len(result)]
            return result
        except Exception as e:
            logger.debug(f"Error applying frac diff with d={d}: {e}")
            return pd.Series()

    def find_optimal_d(self, series: pd.Series, d_range=(0.0, 1.0), step=0.05) -> float:
        """
        Find minimum d for stationarity using ADF test.

        Args:
            series: Price series to analyze
            d_range: Range of d values to test (min, max)
            step: Step size for testing d values

        Returns:
            Optimal d value
        """
        from statsmodels.tsa.stattools import adfuller

        series = series.dropna()

        if len(series) < 100:
            logger.warning(f"Series too short ({len(series)} < 100), defaulting to d=1.0")
            self._optimal_d = 1.0
            return 1.0

        logger.info(f"Finding optimal d for series of length {len(series)}")

        for d in np.arange(d_range[0], d_range[1] + step, step):
            try:
                diff_series = self._apply_frac_diff(series, d)
                diff_clean = diff_series.dropna()

                if len(diff_clean) < 50:
                    continue

                adf_result = adfuller(diff_clean, maxlag=12, autolag='AIC')
                adf_stat, pvalue = adf_result[0], adf_result[1]

                if pvalue < 0.05:  # Stationary at 5% significance
                    self._optimal_d = round(d, 2)
                    logger.info(f"✓ Optimal d={d:.2f} (ADF p-value={pvalue:.4f}, stat={adf_stat:.4f})")
                    return self._optimal_d

            except Exception as e:
                logger.debug(f"ADF test failed for d={d}: {e}")
                continue

        logger.warning("No d achieved stationarity, using d=1.0 (standard returns)")
        self._optimal_d = 1.0
        return 1.0

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

        logger.debug(f"Applying fractional differentiation with d={d:.3f}")

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
