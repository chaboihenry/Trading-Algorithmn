"""
Streamlined Backtesting Engine

Main orchestrator that combines:
1. Walk-forward validation
2. Industry-standard metrics
3. Kelly Criterion trade ranking
4. Top 5 trade selection

Simple, focused, and effective.
"""

import numpy as np
import pandas as pd
import sqlite3
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from backtesting.metrics_calculator import MetricsCalculator
from backtesting.strategy_validator import StrategyValidator
from backtesting.ensemble_validator import EnsembleValidator
from backtesting.trade_ranker import TradeRanker
from backtesting.report_generator import ReportGenerator


class BacktestEngine:
    """Streamlined backtesting engine - main orchestrator"""

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db"):
        """
        Args:
            db_path: Path to database
        """
        self.db_path = db_path
        self.metrics_calc = MetricsCalculator()
        self.strategy_validator = StrategyValidator(db_path)
        self.ensemble_validator = EnsembleValidator(db_path)
        self.ranker = TradeRanker(db_path)
        self.report_generator = ReportGenerator(db_path)

    # ========== Strategy Validation ==========

    def validate_all_strategies(self, quick: bool = False) -> Dict[str, any]:
        """
        Validate all strategies in the database

        Args:
            quick: Use quick validation (single split) vs full walk-forward

        Returns:
            Validation results for all strategies
        """
        print("\n" + "="*80)
        print("VALIDATING ALL STRATEGIES")
        print("="*80)

        # Get list of strategies
        conn = sqlite3.connect(self.db_path)
        strategies = pd.read_sql_query(
            "SELECT DISTINCT strategy_name FROM trading_signals",
            conn
        )
        conn.close()

        strategy_names = strategies['strategy_name'].tolist()
        print(f"\nFound {len(strategy_names)} strategies:")
        for s in strategy_names:
            print(f"  - {s}")

        # Validate each strategy
        results = {}

        for strategy_name in strategy_names:
            print(f"\n{'='*80}")
            print(f"Validating: {strategy_name}")
            print(f"{'='*80}")

            if quick:
                result = self.validator.quick_validation(strategy_name)
            else:
                # For walk-forward, we need to implement strategy-specific logic
                # For now, use quick validation
                result = self.validator.quick_validation(strategy_name)

            results[strategy_name] = result

        # Summary
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)

        passed = []
        failed = []

        for strategy_name, result in results.items():
            if result.get('success') and result.get('passes_validation'):
                passed.append(strategy_name)
                status = "✅ PASS"
            else:
                failed.append(strategy_name)
                status = "❌ FAIL"

            print(f"\n{status} {strategy_name}")
            if 'metrics' in result:
                m = result['metrics']
                print(f"  Sharpe:  {m.get('sharpe_ratio', 0):>6.2f}")
                print(f"  Return:  {m.get('total_return', 0):>6.2%}")
                print(f"  Max DD:  {m.get('max_drawdown', 0):>6.2%}")
                print(f"  Win Rate: {m.get('win_rate', 0):>6.2%}")
            if 'validation_reason' in result:
                print(f"  Reason: {result['validation_reason']}")

        print(f"\n{'─'*80}")
        print(f"Passed: {len(passed)}/{len(strategy_names)}")
        print(f"Failed: {len(failed)}/{len(strategy_names)}")
        print(f"{'='*80}\n")

        return {
            'strategies_tested': len(strategy_names),
            'passed': passed,
            'failed': failed,
            'results': results
        }

    def validate_strategy(self, strategy_name: str, quick: bool = True) -> Dict[str, any]:
        """
        Validate a single strategy

        Args:
            strategy_name: Name of strategy to validate
            quick: Use quick validation vs full walk-forward

        Returns:
            Validation results
        """
        if quick:
            return self.strategy_validator.quick_validation(strategy_name)
        else:
            # Full walk-forward would require strategy-specific train/predict functions
            # For existing signals, use quick validation
            return self.strategy_validator.quick_validation(strategy_name)

    # ========== Trade Selection ==========

    def get_top_trades(self, num_trades: int = 5,
                      total_capital: float = 100_000,
                      date: Optional[str] = None,
                      min_signal_strength: float = 0.3) -> pd.DataFrame:
        """
        Get top N trades for execution

        Args:
            num_trades: Number of trades to select (default 5)
            total_capital: Total available capital
            date: Date to get signals for (default: latest)
            min_signal_strength: Minimum signal strength to consider

        Returns:
            Top N trades with position sizes and levels
        """
        return self.ranker.get_current_top_trades(
            num_trades=num_trades,
            total_capital=total_capital,
            date=date
        )

    # ========== Performance Analysis ==========

    def analyze_portfolio_performance(self, lookback_days: int = 90) -> Dict[str, any]:
        """
        Analyze overall portfolio performance

        Args:
            lookback_days: Days to look back for analysis

        Returns:
            Portfolio performance metrics
        """
        print("\n" + "="*80)
        print("PORTFOLIO PERFORMANCE ANALYSIS")
        print("="*80)

        conn = sqlite3.connect(self.db_path)

        # Get historical signals and outcomes
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

        query = """
        SELECT
            s.date,
            s.symbol,
            s.strategy_name,
            s.signal,
            s.signal_strength,
            p1.close as entry_price,
            p2.close as exit_price
        FROM trading_signals s
        JOIN price_data p1 ON s.symbol = p1.symbol AND s.date = p1.date
        JOIN price_data p2 ON s.symbol = p2.symbol
            AND DATE(p2.date) = DATE(s.date, '+1 day')
        WHERE s.date >= ?
          AND s.signal != 0
        ORDER BY s.date
        """

        df = pd.read_sql_query(query, conn, params=[cutoff_date])
        conn.close()

        if len(df) == 0:
            print("\n❌ No historical trades found")
            return {'success': False, 'error': 'No historical trades'}

        # Calculate returns
        df['return'] = (df['exit_price'] - df['entry_price']) / df['entry_price']

        # Adjust for signal direction
        df.loc[df['signal'] < 0, 'return'] = -df.loc[df['signal'] < 0, 'return']

        # Daily portfolio returns (equal weighted for simplicity)
        daily_returns = df.groupby('date')['return'].mean()

        # Calculate metrics
        metrics = self.metrics_calc.trading_metrics(daily_returns.values, periods_per_year=252)

        # Statistical significance
        t_stat, p_value = self.metrics_calc.statistical_significance(daily_returns.values)
        metrics['t_statistic'] = t_stat
        metrics['p_value'] = p_value

        # Per-strategy breakdown
        strategy_metrics = {}
        for strategy in df['strategy_name'].unique():
            strategy_df = df[df['strategy_name'] == strategy]
            strategy_returns = strategy_df['return'].values

            strategy_metrics[strategy] = {
                'num_trades': len(strategy_returns),
                'win_rate': np.mean(strategy_returns > 0),
                'avg_return': np.mean(strategy_returns),
                'sharpe': (np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)) if np.std(strategy_returns) > 0 else 0
            }

        # Print results
        print(f"\nPeriod: {lookback_days} days")
        print(f"Total Trades: {len(df):,}")
        print(f"Trading Days: {len(daily_returns)}")

        print(f"\n{'─'*80}")
        print("OVERALL PERFORMANCE")
        print(f"{'─'*80}")
        print(f"Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):>8.2f}")
        print(f"Sortino Ratio:     {metrics.get('sortino_ratio', 0):>8.2f}")
        print(f"Calmar Ratio:      {metrics.get('calmar_ratio', 0):>8.2f}")
        print(f"Max Drawdown:      {metrics.get('max_drawdown', 0):>8.2%}")
        print(f"Win Rate:          {metrics.get('win_rate', 0):>8.2%}")
        print(f"Profit Factor:     {metrics.get('profit_factor', 0):>8.2f}")
        print(f"Total Return:      {metrics.get('total_return', 0):>8.2%}")
        print(f"Annual Return:     {metrics.get('annual_return', 0):>8.2%}")
        print(f"Annual Volatility: {metrics.get('annual_volatility', 0):>8.2%}")
        print(f"\nStatistical Significance:")
        print(f"  T-Statistic:     {metrics.get('t_statistic', 0):>8.2f}")
        print(f"  P-Value:         {metrics.get('p_value', 1):>8.4f}")

        print(f"\n{'─'*80}")
        print("PER-STRATEGY BREAKDOWN")
        print(f"{'─'*80}")

        for strategy, strat_metrics in strategy_metrics.items():
            print(f"\n{strategy}:")
            print(f"  Trades:      {strat_metrics['num_trades']:>6}")
            print(f"  Win Rate:    {strat_metrics['win_rate']:>6.2%}")
            print(f"  Avg Return:  {strat_metrics['avg_return']:>6.2%}")
            print(f"  Sharpe:      {strat_metrics['sharpe']:>6.2f}")

        # Validation check
        passes, reason = self.metrics_calc.passes_thresholds(metrics)
        print(f"\n{'─'*80}")
        print(f"Validation: {'✅ PASS' if passes else '❌ FAIL'}")
        print(f"Reason: {reason}")
        print(f"{'='*80}\n")

        return {
            'success': True,
            'overall_metrics': metrics,
            'strategy_metrics': strategy_metrics,
            'num_trades': len(df),
            'num_days': len(daily_returns),
            'passes_validation': passes,
            'validation_reason': reason
        }

    # ========== Complete Backtest Workflow ==========

    def run_complete_backtest(self, validate_strategies: bool = True,
                             select_trades: bool = True,
                             analyze_performance: bool = True,
                             num_trades: int = 5,
                             total_capital: float = 100_000,
                             export_csv: bool = True,
                             output_dir: str = "backtesting/results") -> Dict[str, any]:
        """
        Run complete backtesting workflow

        1. Validate all strategies
        2. Analyze portfolio performance
        3. Select top N trades for execution
        4. Export trades to CSV

        Args:
            validate_strategies: Whether to validate strategies
            select_trades: Whether to select top trades
            analyze_performance: Whether to analyze performance
            num_trades: Number of trades to select
            total_capital: Total available capital
            export_csv: Whether to export trades to CSV
            output_dir: Directory to save results

        Returns:
            Complete backtest results
        """
        print("\n" + "="*80)
        print("COMPLETE BACKTESTING WORKFLOW")
        print("="*80)
        print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'num_trades': num_trades,
                'total_capital': total_capital
            }
        }

        # 1. Validate strategies
        if validate_strategies:
            print("\n" + "▶"*40)
            print("STEP 1: VALIDATING STRATEGIES")
            print("▶"*40)
            validation_results = self.validate_all_strategies(quick=True)
            results['validation'] = validation_results

        # 2. Analyze performance
        if analyze_performance:
            print("\n" + "▶"*40)
            print("STEP 2: ANALYZING PORTFOLIO PERFORMANCE")
            print("▶"*40)
            performance = self.analyze_portfolio_performance(lookback_days=90)
            results['performance'] = performance

        # 3. Select top trades
        if select_trades:
            print("\n" + "▶"*40)
            print("STEP 3: SELECTING TOP TRADES")
            print("▶"*40)
            top_trades = self.get_top_trades(
                num_trades=num_trades,
                total_capital=total_capital
            )
            results['top_trades'] = top_trades

            # 4. Export to CSV
            if export_csv and len(top_trades) > 0:
                print("\n" + "▶"*40)
                print("STEP 4: EXPORTING TRADES")
                print("▶"*40)

                # Create output directory
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)

                # Generate filename with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                csv_path = output_path / f"top_trades_{timestamp}.csv"

                # Export
                self.ranker.export_trades_to_csv(top_trades, str(csv_path))
                results['export_path'] = str(csv_path)

        # Summary
        print("\n" + "="*80)
        print("BACKTEST COMPLETE")
        print("="*80)

        if validate_strategies and 'validation' in results:
            val = results['validation']
            print(f"\nStrategies Validated: {val['strategies_tested']}")
            print(f"  Passed: {len(val['passed'])}")
            print(f"  Failed: {len(val['failed'])}")

        if analyze_performance and 'performance' in results:
            perf = results['performance']
            if perf.get('success'):
                metrics = perf['overall_metrics']
                print(f"\nPortfolio Metrics:")
                print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
                print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")

        if select_trades and 'top_trades' in results:
            trades = results['top_trades']
            print(f"\nTop Trades Selected: {len(trades)}")
            if len(trades) > 0:
                total_allocation = trades['position_size'].sum()
                print(f"  Capital Allocated: {total_allocation:.2%} (${total_allocation * total_capital:,.0f})")

        if export_csv and 'export_path' in results:
            print(f"\nTrades exported to: {results['export_path']}")

        print("\n" + "="*80 + "\n")

        return results

    # ========== Daily Report Generation ==========

    def generate_daily_report(self, num_trades: int = 5,
                             total_capital: float = 100_000,
                             output_dir: str = "backtesting/results") -> Dict[str, any]:
        """
        Generate comprehensive daily trading report

        Args:
            num_trades: Number of top trades to recommend
            total_capital: Total available capital
            output_dir: Directory to save reports

        Returns:
            Report data and file paths
        """
        return self.report_generator.generate_daily_report(
            num_trades=num_trades,
            total_capital=total_capital,
            output_dir=output_dir
        )

    # ========== Quick Methods ==========

    def quick_validation(self, strategy_name: str):
        """Quick validation of a single strategy"""
        return self.strategy_validator.quick_validation(strategy_name)

    def quick_portfolio_analysis(self):
        """Quick portfolio performance analysis"""
        return self.analyze_portfolio_performance(lookback_days=90)

    def quick_trade_selection(self, num_trades: int = 5, capital: float = 100_000):
        """Quick trade selection"""
        return self.get_top_trades(num_trades, capital)

    def validate_ensemble(self):
        """Quick ensemble validation"""
        return self.ensemble_validator.validate_ensemble(lookback_days=90)


if __name__ == "__main__":
    """Run complete backtest when executed directly"""
    engine = BacktestEngine()
    results = engine.run_complete_backtest()
