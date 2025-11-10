#!/usr/bin/env python3
"""
Example Usage of Backtesting Framework
======================================
Simple examples showing how to use the backtesting system
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from backtest_engine import BacktestEngine
from performance_metrics import PerformanceMetrics
from signal_accuracy import SignalAccuracyAnalyzer
from visualizations import BacktestVisualizer


def example_1_basic_backtest():
    """
    Example 1: Run a basic backtest for all strategies
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Backtest")
    print("="*60)

    # Initialize engine with default parameters
    engine = BacktestEngine(
        initial_capital=100000,
        commission=0.001,  # 0.1%
        slippage=0.0005    # 0.05%
    )

    # Run backtest (all strategies, all dates)
    results = engine.run_backtest()

    # Get trades
    trades_df = engine.get_trades_df()

    if not trades_df.empty:
        print(f"\nExecuted {len(trades_df)} trades")
        print(f"Profitable trades: {len(trades_df[trades_df['pnl'] > 0])}")
        print(f"Total P&L: ${trades_df['pnl'].sum():,.2f}")
    else:
        print("\nNo trades executed")


def example_2_specific_strategy():
    """
    Example 2: Backtest a specific strategy with date range
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Specific Strategy with Date Range")
    print("="*60)

    engine = BacktestEngine(initial_capital=50000)

    # Backtest pairs trading strategy for specific period
    results = engine.run_backtest(
        strategy_name="PairsTradingStrategy",
        start_date="2024-01-01",
        end_date="2024-12-31"
    )

    trades_df = engine.get_trades_df()

    if not trades_df.empty:
        # Calculate and print metrics
        metrics = PerformanceMetrics(
            trades_df=trades_df,
            portfolio_values=results['portfolio_value'],
            initial_capital=50000
        )

        metrics.print_summary()


def example_3_performance_analysis():
    """
    Example 3: Detailed performance analysis
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Detailed Performance Analysis")
    print("="*60)

    engine = BacktestEngine(initial_capital=100000)
    results = engine.run_backtest()
    trades_df = engine.get_trades_df()

    if trades_df.empty:
        print("No trades to analyze")
        return

    # Calculate comprehensive metrics
    metrics = PerformanceMetrics(
        trades_df=trades_df,
        portfolio_values=results['portfolio_value'],
        initial_capital=100000
    )

    all_metrics = metrics.calculate_all_metrics()

    # Print specific metrics
    print("\nKey Performance Indicators:")
    print(f"  Total Return:     {all_metrics['total_return_pct']:.2f}%")
    print(f"  Sharpe Ratio:     {all_metrics['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown:     {all_metrics['max_drawdown_pct']:.2f}%")
    print(f"  Win Rate:         {all_metrics['win_rate']:.2f}%")
    print(f"  Profit Factor:    {all_metrics['profit_factor']:.2f}")


def example_4_signal_accuracy():
    """
    Example 4: Analyze signal accuracy and reliability
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Signal Accuracy Analysis")
    print("="*60)

    engine = BacktestEngine(initial_capital=100000)
    results = engine.run_backtest()
    trades_df = engine.get_trades_df()

    if trades_df.empty:
        print("No trades to analyze")
        return

    # Analyze signal accuracy
    analyzer = SignalAccuracyAnalyzer()

    # Print accuracy report
    analyzer.print_accuracy_report(trades_df)

    # Check signal reliability
    reliability = analyzer.analyze_signal_reliability(lookback_days=90)

    if reliability:
        print("\nSignal Reliability:")
        print(f"  Directional Accuracy (5d): {reliability['directional_accuracy_5d']:.2f}%")
        print(f"  Average Return (5d):       {reliability['avg_return_5d']:.2f}%")


def example_5_visualizations():
    """
    Example 5: Create visualizations
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Create Visualizations")
    print("="*60)

    engine = BacktestEngine(initial_capital=100000)
    results = engine.run_backtest()
    trades_df = engine.get_trades_df()

    if trades_df.empty:
        print("No trades to visualize")
        return

    # Create visualizer
    visualizer = BacktestVisualizer(
        trades_df=trades_df,
        portfolio_values=results['portfolio_value'],
        initial_capital=100000
    )

    # Create comprehensive report
    visualizer.create_full_report('example_backtest_report.png')
    print("\n✓ Saved comprehensive report to: example_backtest_report.png")

    # Create individual charts
    visualizer.plot_individual_charts('./example_charts')
    print("✓ Saved individual charts to: ./example_charts/")


def example_6_compare_strategies():
    """
    Example 6: Compare multiple strategies
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: Compare Multiple Strategies")
    print("="*60)

    strategies = [
        "PairsTradingStrategy",
        "SentimentTradingStrategy",
        "VolatilityTradingStrategy"
    ]

    results_comparison = {}

    for strategy in strategies:
        print(f"\nBacktesting {strategy}...")

        engine = BacktestEngine(initial_capital=100000)
        results = engine.run_backtest(strategy_name=strategy)
        trades_df = engine.get_trades_df()

        if not trades_df.empty:
            metrics = PerformanceMetrics(
                trades_df=trades_df,
                portfolio_values=results['portfolio_value'],
                initial_capital=100000
            )
            all_metrics = metrics.calculate_all_metrics()

            results_comparison[strategy] = {
                'total_return_pct': all_metrics['total_return_pct'],
                'sharpe_ratio': all_metrics['sharpe_ratio'],
                'max_drawdown_pct': all_metrics['max_drawdown_pct'],
                'win_rate': all_metrics['win_rate'],
                'total_trades': all_metrics['total_trades']
            }

    # Print comparison
    print("\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60)
    print(f"{'Strategy':<30} {'Return%':<12} {'Sharpe':<10} {'Max DD%':<12} {'Win Rate%':<12} {'Trades'}")
    print("-"*90)

    for strategy, metrics in results_comparison.items():
        print(f"{strategy:<30} "
              f"{metrics['total_return_pct']:>10.2f}% "
              f"{metrics['sharpe_ratio']:>8.3f}  "
              f"{metrics['max_drawdown_pct']:>10.2f}% "
              f"{metrics['win_rate']:>10.2f}% "
              f"{metrics['total_trades']:>8}")


def example_7_custom_parameters():
    """
    Example 7: Backtest with custom risk parameters
    """
    print("\n" + "="*60)
    print("EXAMPLE 7: Custom Risk Parameters")
    print("="*60)

    # Conservative parameters
    conservative_engine = BacktestEngine(
        initial_capital=100000,
        commission=0.002,        # Higher commission (0.2%)
        slippage=0.001,          # Higher slippage (0.1%)
        max_position_size=0.05,  # Smaller positions (5%)
        max_holding_days=10      # Shorter holding period
    )

    print("\nRunning conservative backtest...")
    results = conservative_engine.run_backtest()
    trades_df = conservative_engine.get_trades_df()

    if not trades_df.empty:
        print(f"Executed {len(trades_df)} trades with conservative parameters")
        print(f"Total P&L: ${trades_df['pnl'].sum():,.2f}")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("BACKTESTING FRAMEWORK - EXAMPLE USAGE")
    print("="*70)

    examples = [
        ("Basic Backtest", example_1_basic_backtest),
        ("Specific Strategy", example_2_specific_strategy),
        ("Performance Analysis", example_3_performance_analysis),
        ("Signal Accuracy", example_4_signal_accuracy),
        ("Visualizations", example_5_visualizations),
        ("Compare Strategies", example_6_compare_strategies),
        ("Custom Parameters", example_7_custom_parameters),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nRunning Example 1 (Basic Backtest)...")
    print("To run other examples, modify the code below or run them individually.")

    # Run example 1 by default
    try:
        example_1_basic_backtest()
        print("\n✓ Example completed successfully!")
        print("\nTo run more examples, uncomment the desired function calls below.")
    except Exception as e:
        print(f"\n✗ Example failed: {e}")

    # Uncomment to run other examples:
    # example_2_specific_strategy()
    # example_3_performance_analysis()
    # example_4_signal_accuracy()
    # example_5_visualizations()
    # example_6_compare_strategies()
    # example_7_custom_parameters()


if __name__ == "__main__":
    main()
