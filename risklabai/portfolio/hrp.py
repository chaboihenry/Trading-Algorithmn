"""
Hierarchical Risk Parity (HRP) Portfolio Optimization

Uses RiskLabAI's HRP implementation for stable portfolio optimization.
"""

from RiskLabAI.optimization import (
    get_optimal_portfolio_weights,
    get_optimal_portfolio_weights_nco
)
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class HRPPortfolio:
    """
    Hierarchical Risk Parity portfolio optimizer using RiskLabAI.

    Advantages:
    - No matrix inversion needed
    - Works with singular matrices
    - More stable than mean-variance
    - Naturally diversified

    Attributes:
        weights: Current portfolio weights
    """

    def __init__(self):
        """Initialize HRP optimizer."""
        self.weights = None
        logger.info("HRPPortfolio initialized")

    def optimize(
        self,
        returns: pd.DataFrame,
        mu: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Calculate optimal HRP portfolio weights.

        Args:
            returns: DataFrame of asset returns (assets in columns)
            mu: Optional expected returns (for NCO variant)

        Returns:
            Series of portfolio weights
        """
        logger.info(f"Optimizing HRP portfolio with {returns.shape[1]} assets")

        try:
            # Calculate covariance matrix
            cov = returns.cov().values

            # Convert mu to numpy array if provided
            mu_array = mu.values if mu is not None else None

            # Run HRP optimization
            weights_array = get_optimal_portfolio_weights(
                covariance=cov,
                mu=mu_array
            )

            # Convert to Series
            self.weights = pd.Series(weights_array, index=returns.columns)

            logger.info(f"HRP optimization complete")
            logger.info(f"Weight distribution: min={self.weights.min():.4f}, "
                       f"max={self.weights.max():.4f}, mean={self.weights.mean():.4f}")

            return self.weights

        except Exception as e:
            logger.error(f"HRP optimization failed: {e}")
            # Fallback to equal weights
            logger.warning("Falling back to equal weights")
            self.weights = pd.Series(1.0 / len(returns.columns), index=returns.columns)
            return self.weights

    def optimize_nco(
        self,
        returns: pd.DataFrame,
        mu: Optional[pd.Series] = None,
        n_clusters: Optional[int] = None
    ) -> pd.Series:
        """
        Nested Clustered Optimization (NCO) - Advanced HRP variant.

        Args:
            returns: Asset returns
            mu: Expected returns
            n_clusters: Number of clusters (None = auto)

        Returns:
            Series of NCO weights
        """
        logger.info(f"Running NCO with {returns.shape[1]} assets")

        try:
            cov = returns.cov().values
            mu_array = mu.values if mu is not None else None

            weights_array = get_optimal_portfolio_weights_nco(
                covariance=cov,
                mu=mu_array,
                number_clusters=n_clusters
            )

            weights = pd.Series(weights_array, index=returns.columns)

            logger.info(f"NCO complete: {(weights > 0).sum()}/{len(weights)} assets allocated")
            return weights

        except Exception as e:
            logger.error(f"NCO failed: {e}, falling back to HRP")
            return self.optimize(returns, mu)

    def get_portfolio_stats(
        self,
        weights: pd.Series,
        returns: pd.DataFrame
    ) -> dict:
        """
        Calculate portfolio statistics.

        Args:
            weights: Portfolio weights
            returns: Asset returns

        Returns:
            Dictionary with portfolio stats
        """
        portfolio_returns = (returns * weights).sum(axis=1)

        ann_return = portfolio_returns.mean() * 252
        ann_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        stats = {
            'ann_return': ann_return,
            'ann_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'weight_concentration': (weights ** 2).sum()
        }

        logger.info(f"Portfolio stats: Sharpe={sharpe:.3f}, Vol={ann_vol:.3f}")
        return stats
