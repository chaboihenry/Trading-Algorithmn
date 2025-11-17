"""
Backtesting Framework for Trading Strategies

Provides comprehensive validation, testing, and reporting for trading strategies:
- Individual strategy validation with walk-forward analysis
- Ensemble strategy testing and optimization
- Trade ranking and selection using Kelly Criterion
- Performance metrics calculation
- Daily trade recommendation reports
"""

from backtesting.metrics_calculator import MetricsCalculator
from backtesting.strategy_validator import StrategyValidator
from backtesting.ensemble_validator import EnsembleValidator
from backtesting.trade_ranker import TradeRanker
from backtesting.backtest_engine import BacktestEngine
from backtesting.report_generator import ReportGenerator

__all__ = [
    'MetricsCalculator',
    'StrategyValidator',
    'EnsembleValidator',
    'TradeRanker',
    'BacktestEngine',
    'ReportGenerator'
]

__version__ = '1.0.0'
