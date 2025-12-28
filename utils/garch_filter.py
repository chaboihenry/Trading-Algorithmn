"""
GARCH(1,1) Volatility Filter

Uses GARCH(1,1) model to identify high-volatility regimes where
the RiskLabAI model should be activated.

This complements the CUSUM filter:
- CUSUM: Used in training to prevent overfitting
- GARCH: Used in prediction to identify trading opportunities
"""

import numpy as np
import pandas as pd
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class GARCHVolatilityFilter:
    """
    GARCH(1,1) filter to identify high-volatility trading regimes.

    GARCH(1,1) Model:
    σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1)

    Where:
    - σ²(t) = conditional variance at time t
    - ε²(t-1) = squared residual (shock)
    - ω, α, β = model parameters

    Activation Logic:
    - Activate RiskLabAI when forecasted volatility > threshold
    - Threshold based on historical volatility percentile
    """

    def __init__(
        self,
        lookback_period: int = 100,
        volatility_percentile: float = 0.60,  # Activate when volatility > 60th percentile
        min_observations: int = 50
    ):
        """
        Initialize GARCH filter.

        Args:
            lookback_period: Number of periods for volatility estimation
            volatility_percentile: Percentile threshold for activation (0-1)
            min_observations: Minimum observations required
        """
        self.lookback_period = lookback_period
        self.volatility_percentile = volatility_percentile
        self.min_observations = min_observations

        # GARCH parameters (fitted online or use reasonable defaults)
        self.omega = 0.000001  # Constant term
        self.alpha = 0.10      # ARCH term (reaction to shocks)
        self.beta = 0.85       # GARCH term (persistence)

        # Running state
        self.last_variance = None
        self.historical_volatilities = []

    def simple_garch_forecast(
        self,
        returns: np.ndarray
    ) -> Tuple[float, float]:
        """
        Simple GARCH(1,1) one-step-ahead forecast.

        Uses simplified estimation for real-time prediction without
        full maximum likelihood estimation.

        Args:
            returns: Array of recent returns

        Returns:
            Tuple of (forecasted_volatility, threshold)
        """
        if len(returns) < self.min_observations:
            return 0.0, 0.0

        # Calculate variance using exponential weighting (simplified GARCH)
        # This approximates GARCH without full MLE estimation

        # Initialize variance with sample variance
        if self.last_variance is None:
            self.last_variance = np.var(returns[-self.min_observations:])

        # Get last return (shock)
        last_return = returns[-1]
        last_shock_squared = last_return ** 2

        # GARCH(1,1) one-step forecast
        forecasted_variance = (
            self.omega +
            self.alpha * last_shock_squared +
            self.beta * self.last_variance
        )

        # Update state
        self.last_variance = forecasted_variance

        # Convert to volatility (standard deviation)
        forecasted_vol = np.sqrt(forecasted_variance)

        # Store historical volatility
        self.historical_volatilities.append(forecasted_vol)
        if len(self.historical_volatilities) > self.lookback_period:
            self.historical_volatilities.pop(0)

        # Calculate threshold (percentile of historical volatility)
        if len(self.historical_volatilities) >= self.min_observations:
            threshold = np.percentile(
                self.historical_volatilities,
                self.volatility_percentile * 100
            )
        else:
            # Not enough data - use median as threshold
            threshold = np.median(returns) if len(returns) > 0 else 0.0

        return forecasted_vol, threshold

    def should_trade(
        self,
        prices: pd.Series,
        lookback: int = None
    ) -> Tuple[bool, dict]:
        """
        Determine if current volatility regime warrants trading.

        Args:
            prices: Series of recent prices
            lookback: Number of periods to use (default: self.lookback_period)

        Returns:
            Tuple of (should_trade, info_dict)
        """
        if lookback is None:
            lookback = self.lookback_period

        # Need at least min_observations
        if len(prices) < self.min_observations:
            return False, {
                'reason': 'insufficient_data',
                'n_prices': len(prices),
                'required': self.min_observations
            }

        # Calculate returns
        returns = prices.pct_change().dropna().values

        if len(returns) < self.min_observations:
            return False, {
                'reason': 'insufficient_returns',
                'n_returns': len(returns)
            }

        # Use recent returns for forecast
        recent_returns = returns[-lookback:]

        # Get GARCH forecast
        forecasted_vol, threshold = self.simple_garch_forecast(recent_returns)

        # Decide whether to trade
        should_trade = forecasted_vol > threshold

        # Calculate realized volatility for comparison
        realized_vol = np.std(recent_returns) * np.sqrt(252)  # Annualized

        info = {
            'forecasted_vol': forecasted_vol,
            'threshold': threshold,
            'realized_vol': realized_vol,
            'vol_percentile': self.volatility_percentile,
            'should_trade': should_trade,
            'n_observations': len(recent_returns)
        }

        return should_trade, info

    def reset(self):
        """Reset filter state."""
        self.last_variance = None
        self.historical_volatilities = []
        logger.info("GARCH filter state reset")


def test_garch_filter():
    """Test GARCH filter with synthetic data."""
    import matplotlib.pyplot as plt

    # Generate synthetic price data with regime changes
    np.random.seed(42)

    # Low volatility regime
    low_vol = np.random.normal(0, 0.005, 50)

    # High volatility regime
    high_vol = np.random.normal(0, 0.02, 50)

    # Back to low volatility
    low_vol2 = np.random.normal(0, 0.005, 50)

    # Combine
    returns = np.concatenate([low_vol, high_vol, low_vol2])
    prices = pd.Series(100 * np.exp(np.cumsum(returns)))

    # Test filter
    garch = GARCHVolatilityFilter(
        lookback_period=50,
        volatility_percentile=0.60
    )

    signals = []
    vols = []
    thresholds = []

    for i in range(50, len(prices)):
        should_trade, info = garch.should_trade(prices[:i+1])
        signals.append(should_trade)
        vols.append(info['forecasted_vol'])
        thresholds.append(info['threshold'])

    print(f"Trade signals: {sum(signals)} / {len(signals)} ({sum(signals)/len(signals):.1%})")
    print(f"High vol regime correctly identified: {sum(signals[40:90]) / 50:.1%}")

    return prices, signals, vols, thresholds


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_garch_filter()
    print("✓ GARCH filter test complete")
