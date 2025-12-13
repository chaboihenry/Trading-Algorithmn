"""
Lumibot Trading Strategies Package

This package contains three trading strategies implemented using the Lumibot framework:

1. SentimentStrategy - News sentiment analysis using FinBERT
2. PairsStrategy - Statistical arbitrage with cointegrated pairs
3. CombinedStrategy - Meta-learner ensemble combining both strategies

Usage:
    from lumibot_strategies import SentimentStrategy, PairsStrategy, CombinedStrategy

    # For backtesting
    from lumibot_strategies.run_backtest import BacktestRunner

    # For live trading
    from lumibot_strategies.live_trader import LiveTrader
"""

from .core.sentiment_strategy import SentimentStrategy
from .core.pairs_strategy import PairsStrategy
from .core.combined_strategy import CombinedStrategy

__all__ = ['SentimentStrategy', 'PairsStrategy', 'CombinedStrategy']
__version__ = '1.0.0'
