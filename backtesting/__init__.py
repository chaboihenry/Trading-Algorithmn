"""
Backtesting Framework
====================
Comprehensive backtesting system for trading strategies

Main Components:
- BacktestEngine: Core backtesting with realistic execution
- PerformanceMetrics: 50+ performance metrics
- SignalAccuracyAnalyzer: Signal quality and reliability analysis
- BacktestVisualizer: Automated chart generation

Quick Start:
    from backtesting import BacktestEngine, PerformanceMetrics

    engine = BacktestEngine(initial_capital=100000)
    results = engine.run_backtest()
    trades_df = engine.get_trades_df()

    metrics = PerformanceMetrics(trades_df, results['portfolio_value'], 100000)
    metrics.print_summary()
"""

from .backtest_engine import BacktestEngine, Trade
from .performance_metrics import PerformanceMetrics
from .signal_accuracy import SignalAccuracyAnalyzer
from .visualizations import BacktestVisualizer

__version__ = '1.0.0'
__author__ = 'Integrated Trading Agent'

__all__ = [
    'BacktestEngine',
    'Trade',
    'PerformanceMetrics',
    'SignalAccuracyAnalyzer',
    'BacktestVisualizer'
]
