"""
RiskLabAI Trading System Package

This package contains the state-of-the-art RiskLabAI trading strategy implementing
cutting-edge financial machine learning techniques from Marcos López de Prado:

- Information-driven data structures (dollar bars, volume bars, imbalance bars)
- Triple-barrier labeling with volatility-adaptive targets
- Meta-labeling for bet sizing
- Fractional differentiation for stationary features
- CUSUM event filtering
- Purged K-fold cross-validation
- Hierarchical Risk Parity (HRP) portfolio optimization

Usage:
    from core import RiskLabAICombined

    # For live/paper trading
    strategy = RiskLabAICombined(broker=broker, parameters={'symbols': ['SPY', 'QQQ']})
    strategy.run_all()
"""

from .risklabai_combined import RiskLabAICombined

__all__ = ['RiskLabAICombined']
__version__ = '2.0.0'  # RiskLabAI-powered
