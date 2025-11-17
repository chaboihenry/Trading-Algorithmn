#!/Users/henry/miniconda3/envs/trading/bin/python
"""
Simple Backtesting Runner

Usage:
  python run_backtest.py                    # Full backtest workflow
  python run_backtest.py --validate         # Validate strategies only
  python run_backtest.py --performance      # Analyze performance only
  python run_backtest.py --trades           # Select top trades only
  python run_backtest.py --quick <strategy> # Quick validation of single strategy
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backtesting.backtest_engine import BacktestEngine


def main():
    parser = argparse.ArgumentParser(
        description='Streamlined Backtesting System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete backtest workflow
  python run_backtest.py

  # Validate all strategies
  python run_backtest.py --validate

  # Analyze portfolio performance
  python run_backtest.py --performance

  # Select top 5 trades with $100k capital
  python run_backtest.py --trades --num-trades 5 --capital 100000

  # Quick validation of single strategy
  python run_backtest.py --quick pairs_trading

  # Full backtest with custom settings
  python run_backtest.py --num-trades 10 --capital 250000 --export
        """
    )

    # Mode selection
    parser.add_argument('--validate', action='store_true',
                       help='Validate all strategies only')
    parser.add_argument('--performance', action='store_true',
                       help='Analyze portfolio performance only')
    parser.add_argument('--trades', action='store_true',
                       help='Select top trades only')
    parser.add_argument('--quick', type=str, metavar='STRATEGY',
                       help='Quick validation of single strategy')

    # Parameters
    parser.add_argument('--num-trades', type=int, default=5,
                       help='Number of top trades to select (default: 5)')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Total available capital (default: 100000)')
    parser.add_argument('--db', type=str,
                       default='/Volumes/Vault/85_assets_prediction.db',
                       help='Path to database')
    parser.add_argument('--export', action='store_true',
                       help='Export trades to CSV (default: True for full workflow)')
    parser.add_argument('--output-dir', type=str, default='backtesting/results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Create backtest engine
    print(f"\nInitializing Backtesting Engine...")
    print(f"Database: {args.db}")
    engine = BacktestEngine(db_path=args.db)

    # Execute based on mode
    try:
        # Quick validation of single strategy
        if args.quick:
            print(f"\n{'='*80}")
            print(f"QUICK VALIDATION: {args.quick}")
            print(f"{'='*80}\n")
            result = engine.quick_validation(args.quick)

            if result.get('success'):
                print(f"\n✅ Validation completed")
                sys.exit(0)
            else:
                print(f"\n❌ Validation failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)

        # Validate only
        elif args.validate:
            print(f"\n{'='*80}")
            print("VALIDATING ALL STRATEGIES")
            print(f"{'='*80}\n")
            result = engine.validate_all_strategies(quick=True)

            if result['strategies_tested'] > 0:
                print(f"\n✅ Validated {result['strategies_tested']} strategies")
                print(f"   Passed: {len(result['passed'])}")
                print(f"   Failed: {len(result['failed'])}")
                sys.exit(0)
            else:
                print(f"\n❌ No strategies found")
                sys.exit(1)

        # Performance only
        elif args.performance:
            print(f"\n{'='*80}")
            print("ANALYZING PORTFOLIO PERFORMANCE")
            print(f"{'='*80}\n")
            result = engine.analyze_portfolio_performance(lookback_days=90)

            if result.get('success'):
                print(f"\n✅ Performance analysis completed")
                sys.exit(0)
            else:
                print(f"\n❌ Analysis failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)

        # Trade selection only
        elif args.trades:
            print(f"\n{'='*80}")
            print("SELECTING TOP TRADES")
            print(f"{'='*80}\n")
            top_trades = engine.get_top_trades(
                num_trades=args.num_trades,
                total_capital=args.capital
            )

            if len(top_trades) > 0:
                print(f"\n✅ Selected {len(top_trades)} top trades")

                if args.export:
                    # Export to CSV
                    output_path = Path(args.output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)

                    from datetime import datetime
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    csv_path = output_path / f"top_trades_{timestamp}.csv"

                    engine.ranker.export_trades_to_csv(top_trades, str(csv_path))

                sys.exit(0)
            else:
                print(f"\n❌ No trades selected")
                sys.exit(1)

        # Full workflow (default)
        else:
            print(f"\n{'='*80}")
            print("RUNNING COMPLETE BACKTEST WORKFLOW")
            print(f"{'='*80}")
            print(f"\nConfiguration:")
            print(f"  Number of Trades: {args.num_trades}")
            print(f"  Total Capital:    ${args.capital:,.0f}")
            print(f"  Output Directory: {args.output_dir}")
            print(f"{'='*80}\n")

            result = engine.run_complete_backtest(
                validate_strategies=True,
                select_trades=True,
                analyze_performance=True,
                num_trades=args.num_trades,
                total_capital=args.capital,
                export_csv=True,
                output_dir=args.output_dir
            )

            # Success if we got results
            if result:
                print(f"\n✅ Backtesting workflow completed successfully")

                # Print key results
                if 'validation' in result:
                    val = result['validation']
                    print(f"\nStrategies Validated: {val['strategies_tested']}")
                    print(f"  ✅ Passed: {len(val['passed'])}")
                    print(f"  ❌ Failed: {len(val['failed'])}")

                if 'performance' in result and result['performance'].get('success'):
                    perf = result['performance']['overall_metrics']
                    print(f"\nPortfolio Performance:")
                    print(f"  Sharpe Ratio:    {perf.get('sharpe_ratio', 0):>6.2f}")
                    print(f"  Total Return:    {perf.get('total_return', 0):>6.2%}")
                    print(f"  Max Drawdown:    {perf.get('max_drawdown', 0):>6.2%}")
                    print(f"  Win Rate:        {perf.get('win_rate', 0):>6.2%}")

                if 'top_trades' in result:
                    trades = result['top_trades']
                    print(f"\nTop Trades: {len(trades)}")

                if 'export_path' in result:
                    print(f"\n📄 Trades exported to: {result['export_path']}")

                sys.exit(0)
            else:
                print(f"\n❌ Backtesting failed")
                sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
