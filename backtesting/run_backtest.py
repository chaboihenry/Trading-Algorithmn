#!/usr/bin/env python3
"""
Comprehensive Backtesting Runner
================================
Run backtests for all strategies and generate performance reports
"""

import argparse
import sys
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from backtest_engine import BacktestEngine
from performance_metrics import PerformanceMetrics
from signal_accuracy import SignalAccuracyAnalyzer
from visualizations import BacktestVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_full_backtest(
    strategy_name=None,
    start_date=None,
    end_date=None,
    initial_capital=100000,
    commission=0.001,
    slippage=0.0005,
    max_position_size=0.1,
    max_holding_days=20,
    output_dir='./backtest_results',
    db_path="/Volumes/Vault/85_assets_prediction.db"
):
    """
    Run comprehensive backtest with all analysis

    Args:
        strategy_name: Strategy to backtest (None = all strategies)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_capital: Starting capital
        commission: Commission rate
        slippage: Slippage rate
        max_position_size: Max position size as fraction
        max_holding_days: Max holding period
        output_dir: Directory for output files
        db_path: Path to database
    """
    logger.info("="*60)
    logger.info("STARTING COMPREHENSIVE BACKTEST")
    logger.info("="*60)

    # Setup output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize backtest engine
    logger.info("\n1. Initializing Backtest Engine...")
    engine = BacktestEngine(
        db_path=db_path,
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage,
        max_position_size=max_position_size,
        max_holding_days=max_holding_days
    )

    # Run backtest
    logger.info(f"\n2. Running Backtest...")
    logger.info(f"   Strategy: {strategy_name or 'All Strategies'}")
    logger.info(f"   Start Date: {start_date or 'Earliest Available'}")
    logger.info(f"   End Date: {end_date or 'Latest Available'}")
    logger.info(f"   Initial Capital: ${initial_capital:,.2f}")

    results = engine.run_backtest(strategy_name, start_date, end_date)

    if not results:
        logger.error("No backtest results generated. Check if signals exist in database.")
        return

    # Get trades DataFrame
    trades_df = engine.get_trades_df()

    if trades_df.empty:
        logger.warning("No trades executed during backtest period.")
        return

    logger.info(f"\n✓ Backtest Complete: {len(trades_df)} trades executed")

    # Calculate performance metrics
    logger.info("\n3. Calculating Performance Metrics...")
    metrics_calculator = PerformanceMetrics(
        trades_df=trades_df,
        portfolio_values=results['portfolio_value'],
        initial_capital=initial_capital
    )

    metrics_calculator.print_summary()

    # Save metrics to file
    metrics = metrics_calculator.calculate_all_metrics()
    metrics_file = f"{output_dir}/performance_metrics.txt"

    with open(metrics_file, 'w') as f:
        f.write("BACKTEST PERFORMANCE METRICS\n")
        f.write("="*60 + "\n\n")
        for key, value in metrics.items():
            if isinstance(value, dict):
                f.write(f"\n{key}:\n")
                for k, v in value.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"{key}: {value}\n")

    logger.info(f"✓ Performance metrics saved to {metrics_file}")

    # Analyze signal accuracy
    logger.info("\n4. Analyzing Signal Accuracy...")
    accuracy_analyzer = SignalAccuracyAnalyzer(db_path=db_path)
    accuracy_analyzer.print_accuracy_report(trades_df)

    # Save accuracy report
    accuracy_file = f"{output_dir}/signal_accuracy.txt"
    import io
    import contextlib

    with open(accuracy_file, 'w') as f:
        with contextlib.redirect_stdout(f):
            accuracy_analyzer.print_accuracy_report(trades_df)

    logger.info(f"✓ Signal accuracy report saved to {accuracy_file}")

    # Analyze signal reliability (directional accuracy)
    logger.info("\n5. Analyzing Signal Reliability...")
    reliability = accuracy_analyzer.analyze_signal_reliability(
        strategy_name=strategy_name,
        lookback_days=90
    )

    if reliability:
        print("\nSignal Reliability (Directional Accuracy):")
        print(f"  1-Day:  {reliability['directional_accuracy_1d']:.2f}%")
        print(f"  5-Day:  {reliability['directional_accuracy_5d']:.2f}%")
        print(f"  10-Day: {reliability['directional_accuracy_10d']:.2f}%")
        print(f"  Avg Return (5d): {reliability['avg_return_5d']:.2f}%")

    # Create visualizations
    logger.info("\n6. Creating Visualizations...")
    visualizer = BacktestVisualizer(
        trades_df=trades_df,
        portfolio_values=results['portfolio_value'],
        initial_capital=initial_capital
    )

    # Create comprehensive report
    report_file = f"{output_dir}/backtest_report.png"
    visualizer.create_full_report(save_path=report_file)
    logger.info(f"✓ Comprehensive report saved to {report_file}")

    # Create individual charts
    charts_dir = f"{output_dir}/charts"
    visualizer.plot_individual_charts(output_dir=charts_dir)
    logger.info(f"✓ Individual charts saved to {charts_dir}")

    # Save trades to CSV
    trades_file = f"{output_dir}/trades_detailed.csv"
    trades_df.to_csv(trades_file, index=False)
    logger.info(f"✓ Detailed trades saved to {trades_file}")

    # Summary statistics
    logger.info("\n" + "="*60)
    logger.info("BACKTEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Total Trades:          {len(trades_df)}")
    logger.info(f"Win Rate:              {metrics['win_rate']:.2f}%")
    logger.info(f"Total Return:          ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:.2f}%)")
    logger.info(f"Sharpe Ratio:          {metrics['sharpe_ratio']:.3f}")
    logger.info(f"Max Drawdown:          {metrics['max_drawdown_pct']:.2f}%")
    logger.info(f"Profit Factor:         {metrics['profit_factor']:.2f}")
    logger.info("="*60)

    logger.info(f"\n✓ All results saved to: {output_dir}")

    return {
        'trades_df': trades_df,
        'metrics': metrics,
        'reliability': reliability
    }


def main():
    """Main entry point with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description='Run comprehensive backtesting for trading strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest all strategies
  python run_backtest.py

  # Backtest specific strategy
  python run_backtest.py --strategy PairsTradingStrategy

  # Backtest with date range
  python run_backtest.py --start-date 2024-01-01 --end-date 2024-12-31

  # Backtest with custom parameters
  python run_backtest.py --capital 50000 --commission 0.002 --max-position 0.05

  # Backtest with custom output directory
  python run_backtest.py --output ./my_backtest_results
        """
    )

    parser.add_argument(
        '--strategy',
        type=str,
        default=None,
        help='Strategy name to backtest (default: all strategies)'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date in YYYY-MM-DD format (default: earliest available)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date in YYYY-MM-DD format (default: latest available)'
    )

    parser.add_argument(
        '--capital',
        type=float,
        default=100000,
        help='Initial capital (default: 100000)'
    )

    parser.add_argument(
        '--commission',
        type=float,
        default=0.001,
        help='Commission rate per trade (default: 0.001 = 0.1%%)'
    )

    parser.add_argument(
        '--slippage',
        type=float,
        default=0.0005,
        help='Slippage rate per trade (default: 0.0005 = 0.05%%)'
    )

    parser.add_argument(
        '--max-position',
        type=float,
        default=0.1,
        help='Maximum position size as fraction of capital (default: 0.1 = 10%%)'
    )

    parser.add_argument(
        '--max-holding-days',
        type=int,
        default=20,
        help='Maximum days to hold a position (default: 20)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./backtest_results',
        help='Output directory for results (default: ./backtest_results)'
    )

    parser.add_argument(
        '--db',
        type=str,
        default="/Volumes/Vault/85_assets_prediction.db",
        help='Database path (default: /Volumes/Vault/85_assets_prediction.db)'
    )

    args = parser.parse_args()

    # Add timestamp to output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{args.output}/{timestamp}"

    # Run backtest
    try:
        run_full_backtest(
            strategy_name=args.strategy,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_capital=args.capital,
            commission=args.commission,
            slippage=args.slippage,
            max_position_size=args.max_position,
            max_holding_days=args.max_holding_days,
            output_dir=output_dir,
            db_path=args.db
        )
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
