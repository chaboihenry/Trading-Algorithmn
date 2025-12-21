"""
Hierarchical Risk Parity (HRP) Portfolio Optimization

Problems with traditional mean-variance optimization:
- Requires inverting covariance matrix (unstable)
- Small estimation errors → wildly different portfolios
- Fails with singular matrices

HRP solution:
1. Cluster assets by correlation
2. Quasi-diagonalize covariance matrix
3. Recursively allocate weights based on variance

Result: Stable, diversified portfolios that work in practice.
"""

from RiskLabAI.portfolio import (
    hrp,
    hrp_allocation,
    nco
)
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class HRPPortfolio:
    """
    Hierarchical Risk Parity portfolio optimizer.

    The three steps:
    1. Tree Clustering: Group similar assets hierarchically
    2. Quasi-Diagonalization: Reorder covariance matrix by clusters
    3. Recursive Bisection: Allocate weights from top-down

    Advantages:
    - No matrix inversion needed
    - Works with singular matrices
    - More stable than mean-variance
    - Naturally diversified

    Attributes:
        linkage_method: How to compute cluster distances
    """

    def __init__(self, linkage_method: str = 'single'):
        """
        Initialize HRP optimizer.

        Args:
            linkage_method: 'single', 'complete', 'average', or 'ward'
        """
        self.linkage_method = linkage_method
        self.weights = None

        logger.info(f"HRPPortfolio initialized: linkage={linkage_method}")

    def optimize(
        self,
        returns: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate optimal HRP portfolio weights.

        Args:
            returns: DataFrame of asset returns (assets in columns)

        Returns:
            Series of portfolio weights
        """
        logger.info(f"Optimizing HRP portfolio with {returns.shape[1]} assets")

        # Calculate covariance and correlation
        cov = returns.cov()
        corr = returns.corr()

        try:
            # Run HRP optimization
            self.weights = hrp(
                cov=cov,
                corr=corr,
                link_method=self.linkage_method
            )

            # Convert to Series if needed
            if isinstance(self.weights, np.ndarray):
                self.weights = pd.Series(self.weights, index=returns.columns)

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

    def rebalance(
        self,
        current_weights: pd.Series,
        new_returns: pd.DataFrame,
        threshold: float = 0.05
    ) -> tuple[pd.Series, bool]:
        """
        Rebalance only if weights drift significantly.

        Args:
            current_weights: Current portfolio weights
            new_returns: Recent returns for new optimization
            threshold: Minimum drift to trigger rebalance

        Returns:
            Tuple of (new_weights, rebalanced_flag)
        """
        new_weights = self.optimize(new_returns)

        # Check drift
        drift = (new_weights - current_weights).abs().sum()

        logger.info(f"Weight drift: {drift:.4f}")

        if drift > threshold:
            logger.info(f"Drift {drift:.4f} > threshold {threshold}, rebalancing")
            return new_weights, True
        else:
            logger.info(f"Drift {drift:.4f} <= threshold {threshold}, no rebalance")
            return current_weights, False

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
        # Portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)

        # Annualized metrics (assuming daily returns)
        ann_return = portfolio_returns.mean() * 252
        ann_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        stats = {
            'ann_return': ann_return,
            'ann_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'weight_concentration': self._calculate_concentration(weights)
        }

        logger.info(f"Portfolio stats: Sharpe={sharpe:.3f}, Vol={ann_vol:.3f}")

        return stats

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()

    def _calculate_concentration(self, weights: pd.Series) -> float:
        """
        Calculate weight concentration using Herfindahl index.

        Returns value between 1/N (perfectly diversified) and 1 (concentrated).
        """
        return (weights ** 2).sum()

    def backtest_portfolio(
        self,
        returns: pd.DataFrame,
        rebalance_freq: str = 'M',
        initial_weights: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Backtest HRP strategy with periodic rebalancing.

        Args:
            returns: Historical returns
            rebalance_freq: Rebalancing frequency ('D', 'W', 'M', 'Q')
            initial_weights: Starting weights (None = optimize from start)

        Returns:
            DataFrame with portfolio performance
        """
        logger.info(f"Backtesting HRP with {rebalance_freq} rebalancing")

        # Initialize
        if initial_weights is None:
            current_weights = self.optimize(returns.iloc[:60])  # Use first 60 days
        else:
            current_weights = initial_weights

        portfolio_values = []
        rebalance_dates = []

        # Get rebalance dates
        rebal_dates = returns.resample(rebalance_freq).last().index

        for date in returns.index:
            # Check if rebalance date
            if date in rebal_dates:
                lookback_returns = returns.loc[:date].tail(60)
                current_weights = self.optimize(lookback_returns)
                rebalance_dates.append(date)

            # Calculate portfolio return
            daily_return = (returns.loc[date] * current_weights).sum()
            portfolio_values.append(daily_return)

        # Create results DataFrame
        results = pd.DataFrame({
            'returns': portfolio_values,
            'cumulative': (1 + pd.Series(portfolio_values, index=returns.index)).cumprod()
        })

        logger.info(f"Backtest complete: {len(rebalance_dates)} rebalances")

        return results

    def nco_allocation(
        self,
        returns: pd.DataFrame,
        min_weight: float = 0.01,
        max_weight: float = 0.5
    ) -> pd.Series:
        """
        Nested Clustered Optimization (NCO) - Advanced HRP variant.

        NCO combines clustering with optimization within clusters.

        Args:
            returns: Asset returns
            min_weight: Minimum asset weight
            max_weight: Maximum asset weight

        Returns:
            Series of NCO weights
        """
        logger.info(f"Running NCO allocation with {returns.shape[1]} assets")

        cov = returns.cov()
        corr = returns.corr()
        mu = returns.mean()

        try:
            weights = nco(
                cov=cov,
                mu=mu,
                corr=corr,
                max_num_clusters=10,
                min_weight=min_weight,
                max_weight=max_weight
            )

            if isinstance(weights, np.ndarray):
                weights = pd.Series(weights, index=returns.columns)

            logger.info(f"NCO complete: {(weights > 0).sum()}/{len(weights)} assets allocated")

            return weights

        except Exception as e:
            logger.error(f"NCO failed: {e}, falling back to HRP")
            return self.optimize(returns)
