"""
Strategy Validator with Walk-Forward Analysis

Validates individual trading strategies using time series analysis:
- 1 year training windows
- 3 month testing windows
- Tests strategies across different market regimes
- Quick validation mode for development
- Statistical significance testing
"""

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Callable, Optional
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from backtesting.metrics_calculator import MetricsCalculator


class StrategyValidator:
    """Validates individual trading strategies with walk-forward analysis"""

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db",
                 train_months: int = 12,
                 test_months: int = 3):
        """
        Args:
            db_path: Path to database
            train_months: Training window size (default 12 months)
            test_months: Testing window size (default 3 months)
        """
        self.db_path = db_path
        self.train_months = train_months
        self.test_months = test_months
        self.metrics_calc = MetricsCalculator()

    def get_data_range(self) -> Tuple[datetime, datetime]:
        """Get available date range from database"""
        conn = sqlite3.connect(self.db_path)

        # Get min/max dates from ml_features table (using feature_date column)
        query = """
        SELECT MIN(feature_date) as min_date, MAX(feature_date) as max_date
        FROM ml_features
        WHERE feature_date IS NOT NULL
        """

        result = pd.read_sql_query(query, conn)
        conn.close()

        min_date = pd.to_datetime(result['min_date'].iloc[0])
        max_date = pd.to_datetime(result['max_date'].iloc[0])

        return min_date, max_date

    def generate_windows(self, start_date: datetime, end_date: datetime) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate walk-forward windows

        Args:
            start_date: Earliest available date
            end_date: Latest available date

        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        windows = []

        current_train_start = start_date

        while True:
            # Calculate window boundaries
            current_train_end = current_train_start + timedelta(days=self.train_months * 30)
            current_test_start = current_train_end + timedelta(days=1)
            current_test_end = current_test_start + timedelta(days=self.test_months * 30)

            # Stop if test window goes beyond available data
            if current_test_end > end_date:
                break

            windows.append((
                current_train_start,
                current_train_end,
                current_test_start,
                current_test_end
            ))

            # Move forward by test_months for next window
            current_train_start = current_test_start

        return windows

    def load_data(self, start_date: datetime, end_date: datetime,
                  symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Load data for given date range

        Args:
            start_date: Start date
            end_date: End date
            symbol: Optional symbol filter

        Returns:
            DataFrame with features and signals
        """
        conn = sqlite3.connect(self.db_path)

        # Build query
        query = """
        SELECT
            m.*,
            s.signal,
            s.signal_strength,
            s.strategy_name,
            p.close,
            p.volume
        FROM ml_features m
        LEFT JOIN trading_signals s ON m.symbol = s.symbol AND m.date = s.date
        LEFT JOIN price_data p ON m.symbol = p.symbol AND m.date = p.date
        WHERE m.date >= ? AND m.date <= ?
        """

        params = [start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')]

        if symbol:
            query += " AND m.symbol = ?"
            params.append(symbol)

        query += " ORDER BY m.date, m.symbol"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])

        return df

    def validate_strategy(self, strategy_train_fn: Callable,
                         strategy_predict_fn: Callable,
                         strategy_name: str,
                         symbols: Optional[List[str]] = None,
                         min_windows: int = 3) -> Dict[str, any]:
        """
        Validate strategy using walk-forward analysis

        Args:
            strategy_train_fn: Function(train_data) -> model
            strategy_predict_fn: Function(model, test_data) -> predictions
            strategy_name: Name of strategy
            symbols: Optional list of symbols to test
            min_windows: Minimum number of windows required

        Returns:
            Validation results with metrics for each window
        """
        print("\n" + "="*80)
        print(f"WALK-FORWARD VALIDATION: {strategy_name}")
        print("="*80)

        # Get data range
        min_date, max_date = self.get_data_range()
        print(f"\nData Range: {min_date.date()} to {max_date.date()}")

        # Generate windows
        windows = self.generate_windows(min_date, max_date)
        print(f"Generated {len(windows)} walk-forward windows")
        print(f"  Training: {self.train_months} months")
        print(f"  Testing: {self.test_months} months")

        if len(windows) < min_windows:
            print(f"\n❌ ERROR: Only {len(windows)} windows available, need at least {min_windows}")
            return {
                'success': False,
                'error': f'Insufficient data for validation (need {min_windows} windows)'
            }

        # Run validation on each window
        window_results = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows, 1):
            print(f"\n{'─'*80}")
            print(f"Window {i}/{len(windows)}")
            print(f"  Train: {train_start.date()} to {train_end.date()}")
            print(f"  Test:  {test_start.date()} to {test_end.date()}")
            print(f"{'─'*80}")

            try:
                # Load train and test data
                print("  Loading data...")
                train_data = self.load_data(train_start, train_end, symbol=None)
                test_data = self.load_data(test_start, test_end, symbol=None)

                print(f"    Train: {len(train_data):,} rows, {train_data['symbol'].nunique()} symbols")
                print(f"    Test:  {len(test_data):,} rows, {test_data['symbol'].nunique()} symbols")

                if len(train_data) == 0 or len(test_data) == 0:
                    print("  ⚠️  Skipping window (insufficient data)")
                    continue

                # Train strategy
                print("  Training strategy...")
                model = strategy_train_fn(train_data)

                # Generate predictions
                print("  Generating predictions...")
                predictions = strategy_predict_fn(model, test_data)

                # Calculate returns
                print("  Calculating returns...")
                returns = self._calculate_returns(test_data, predictions)

                if len(returns) == 0:
                    print("  ⚠️  Skipping window (no valid returns)")
                    continue

                # Calculate metrics
                print("  Computing metrics...")
                metrics = self.metrics_calc.trading_metrics(returns, periods_per_year=252)

                # Add statistical significance
                t_stat, p_value = self.metrics_calc.statistical_significance(returns)
                metrics['t_statistic'] = t_stat
                metrics['p_value'] = p_value

                # Store results
                window_results.append({
                    'window': i,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'metrics': metrics,
                    'num_trades': len(returns)
                })

                # Print window metrics
                print(f"\n  Window {i} Results:")
                print(f"    Trades:          {len(returns):>8}")
                print(f"    Sharpe Ratio:    {metrics.get('sharpe_ratio', 0):>8.2f}")
                print(f"    Total Return:    {metrics.get('total_return', 0):>8.2%}")
                print(f"    Max Drawdown:    {metrics.get('max_drawdown', 0):>8.2%}")
                print(f"    Win Rate:        {metrics.get('win_rate', 0):>8.2%}")

            except Exception as e:
                print(f"  ❌ Error in window {i}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if len(window_results) == 0:
            print("\n❌ ERROR: No valid windows completed")
            return {
                'success': False,
                'error': 'No valid windows completed'
            }

        # Aggregate results across all windows
        print(f"\n{'='*80}")
        print("AGGREGATED RESULTS")
        print(f"{'='*80}")

        aggregated = self._aggregate_results(window_results)

        # Print aggregated metrics
        print(f"\nWindows Completed: {len(window_results)}/{len(windows)}")
        print(f"Total Trades: {aggregated['total_trades']:,}")
        print(f"\nAverage Metrics Across Windows:")
        print(f"  Sharpe Ratio:      {aggregated['avg_sharpe']:>8.2f}")
        print(f"  Sortino Ratio:     {aggregated['avg_sortino']:>8.2f}")
        print(f"  Max Drawdown:      {aggregated['avg_max_dd']:>8.2%}")
        print(f"  Win Rate:          {aggregated['avg_win_rate']:>8.2%}")
        print(f"  Profit Factor:     {aggregated['avg_profit_factor']:>8.2f}")
        print(f"\nConsistency:")
        print(f"  Sharpe Std Dev:    {aggregated['sharpe_std']:>8.2f}")
        print(f"  Win Rate Std Dev:  {aggregated['win_rate_std']:>8.2%}")

        # Check if passes validation thresholds
        passes, reason = self.metrics_calc.passes_thresholds(aggregated)
        print(f"\nValidation Result: {'✅ PASS' if passes else '❌ FAIL'}")
        print(f"Reason: {reason}")

        print(f"{'='*80}\n")

        return {
            'success': True,
            'strategy_name': strategy_name,
            'num_windows': len(window_results),
            'window_results': window_results,
            'aggregated': aggregated,
            'passes_validation': passes,
            'validation_reason': reason
        }

    def _calculate_returns(self, test_data: pd.DataFrame,
                          predictions: pd.DataFrame) -> np.ndarray:
        """
        Calculate returns from predictions

        Args:
            test_data: Test data with actual prices
            predictions: Predictions with signals (can be same as test_data)

        Returns:
            Array of returns
        """
        # If predictions is the same as test_data (already has prices), use it directly
        if 'close' in predictions.columns:
            merged = predictions.copy()
        else:
            # Merge predictions with actual data
            merged = predictions.merge(
                test_data[['symbol', 'date', 'close']],
                on=['symbol', 'date'],
                how='left'
            )

        # Calculate returns for each signal
        returns_list = []

        # Convert BUY/SELL to numeric
        if 'signal' in merged.columns and merged['signal'].dtype == 'object':
            merged['signal_numeric'] = merged['signal'].map({'BUY': 1, 'SELL': -1, 'HOLD': 0})
        else:
            merged['signal_numeric'] = merged['signal']

        for symbol in merged['symbol'].unique():
            symbol_data = merged[merged['symbol'] == symbol].sort_values('date')

            for i in range(len(symbol_data) - 1):
                row = symbol_data.iloc[i]
                next_row = symbol_data.iloc[i + 1]

                signal_value = row.get('signal_numeric', 0)
                if pd.notna(signal_value) and signal_value != 0:
                    # Calculate return
                    entry_price = row['close']
                    exit_price = next_row['close']

                    if pd.notna(entry_price) and pd.notna(exit_price) and entry_price > 0:
                        ret = (exit_price - entry_price) / entry_price

                        # Adjust for signal direction
                        if signal_value < 0:  # Short signal (SELL)
                            ret = -ret

                        returns_list.append(ret)

        return np.array(returns_list)

    def _aggregate_results(self, window_results: List[Dict]) -> Dict[str, float]:
        """
        Aggregate metrics across all windows

        Args:
            window_results: List of window results

        Returns:
            Aggregated metrics
        """
        # Extract metrics from each window
        sharpe_ratios = [w['metrics']['sharpe_ratio'] for w in window_results]
        sortino_ratios = [w['metrics']['sortino_ratio'] for w in window_results]
        max_drawdowns = [w['metrics']['max_drawdown'] for w in window_results]
        win_rates = [w['metrics']['win_rate'] for w in window_results]
        profit_factors = [w['metrics']['profit_factor'] for w in window_results]
        total_returns = [w['metrics']['total_return'] for w in window_results]

        # Calculate aggregated metrics
        aggregated = {
            'avg_sharpe': np.mean(sharpe_ratios),
            'sharpe_std': np.std(sharpe_ratios),
            'avg_sortino': np.mean(sortino_ratios),
            'avg_max_dd': np.mean(max_drawdowns),
            'avg_win_rate': np.mean(win_rates),
            'win_rate_std': np.std(win_rates),
            'avg_profit_factor': np.mean(profit_factors),
            'avg_total_return': np.mean(total_returns),
            'total_trades': sum(w['num_trades'] for w in window_results),
            'sharpe_ratio': np.mean(sharpe_ratios),  # For threshold checking
            'max_drawdown': np.mean(max_drawdowns),  # For threshold checking
            'win_rate': np.mean(win_rates),  # For threshold checking
            'profit_factor': np.mean(profit_factors)  # For threshold checking
        }

        return aggregated

    def quick_validation(self, strategy_name: str,
                        lookback_days: int = 365) -> Dict[str, any]:
        """
        Quick validation using single train/test split (for development)

        Args:
            strategy_name: Name of strategy to validate
            lookback_days: Days of historical data to use

        Returns:
            Quick validation results
        """
        print("\n" + "="*80)
        print(f"QUICK VALIDATION: {strategy_name}")
        print("="*80)

        # Get data range
        min_date, max_date = self.get_data_range()

        # Use last lookback_days
        train_start = max_date - timedelta(days=lookback_days)
        train_end = max_date - timedelta(days=90)  # Last 90 days for testing
        test_start = train_end + timedelta(days=1)
        test_end = max_date

        print(f"\nTrain: {train_start.date()} to {train_end.date()}")
        print(f"Test:  {test_start.date()} to {test_end.date()}")

        # Load existing signals from database (using correct column names)
        conn = sqlite3.connect(self.db_path)

        query = """
        SELECT
            s.signal_id,
            s.strategy_name,
            s.symbol_ticker as symbol,
            s.signal_date as date,
            s.signal_type as signal,
            s.strength as signal_strength,
            s.entry_price,
            s.stop_loss,
            s.take_profit,
            p.close,
            p.volume
        FROM trading_signals s
        JOIN raw_price_data p ON s.symbol_ticker = p.symbol_ticker AND s.signal_date = p.price_date
        WHERE s.strategy_name = ?
          AND s.signal_date >= ?
          AND s.signal_date <= ?
        ORDER BY s.signal_date, s.symbol_ticker
        """

        test_data = pd.read_sql_query(
            query,
            conn,
            params=[strategy_name, test_start.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d')]
        )
        conn.close()

        test_data['date'] = pd.to_datetime(test_data['date'])

        print(f"\nLoaded {len(test_data):,} signals")

        if len(test_data) == 0:
            print("\n❌ No signals found for this strategy")
            return {'success': False, 'error': 'No signals found'}

        # Calculate returns
        returns = self._calculate_returns(test_data, test_data)

        print(f"Calculated {len(returns)} returns")

        if len(returns) == 0:
            print("\n❌ No valid returns calculated")
            return {'success': False, 'error': 'No valid returns'}

        # Calculate metrics
        metrics = self.metrics_calc.trading_metrics(returns, periods_per_year=252)
        t_stat, p_value = self.metrics_calc.statistical_significance(returns)
        metrics['t_statistic'] = t_stat
        metrics['p_value'] = p_value

        # NEW: Advanced risk metrics
        metrics['mar_ratio'] = self.metrics_calc.mar_ratio(returns)
        metrics['var_95'] = self.metrics_calc.value_at_risk(returns, confidence=0.95)
        metrics['cvar_95'] = self.metrics_calc.conditional_value_at_risk(returns, confidence=0.95)

        # NEW: Bootstrap confidence interval for Sharpe ratio
        sharpe_ci = self.metrics_calc.bootstrap_confidence_interval(
            returns,
            lambda x: (x.mean() * np.sqrt(252)) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0,
            n_bootstrap=1000
        )
        metrics['sharpe_95_ci_lower'] = sharpe_ci[0]
        metrics['sharpe_95_ci_upper'] = sharpe_ci[2]

        # NEW: Monte Carlo drawdown analysis
        dd_analysis = self.metrics_calc.monte_carlo_drawdown_distribution(returns)
        metrics['max_dd_p_value'] = dd_analysis['p_value']
        metrics['max_dd_is_significant'] = dd_analysis['p_value'] < 0.05

        # Interpretation
        metrics['risk_level'] = 'LOW' if abs(metrics.get('max_drawdown', 0)) < 0.10 else 'MEDIUM' if abs(metrics.get('max_drawdown', 0)) < 0.20 else 'HIGH'
        metrics['statistical_significance'] = 'SIGNIFICANT' if sharpe_ci[0] > 0 else 'NOT SIGNIFICANT'

        # Print results
        print(f"\n{'─'*80}")
        print("RESULTS")
        print(f"{'─'*80}")
        print(f"Trades:            {len(returns):>8}")
        print(f"Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):>8.2f} (CI: [{metrics.get('sharpe_95_ci_lower', 0):.2f}, {metrics.get('sharpe_95_ci_upper', 0):.2f}])")
        print(f"Sortino Ratio:     {metrics.get('sortino_ratio', 0):>8.2f}")
        print(f"Calmar Ratio:      {metrics.get('calmar_ratio', 0):>8.2f}")
        print(f"MAR Ratio:         {metrics.get('mar_ratio', 0):>8.2f}")
        print(f"Total Return:      {metrics.get('total_return', 0):>8.2%}")
        print(f"Max Drawdown:      {metrics.get('max_drawdown', 0):>8.2%} (p-value: {metrics.get('max_dd_p_value', 1):.3f})")
        print(f"Win Rate:          {metrics.get('win_rate', 0):>8.2%}")
        print(f"Profit Factor:     {metrics.get('profit_factor', 0):>8.2f}")
        print(f"VaR (95%):         {metrics.get('var_95', 0):>8.2%}")
        print(f"CVaR (95%):        {metrics.get('cvar_95', 0):>8.2%}")
        print(f"T-Statistic:       {metrics.get('t_statistic', 0):>8.2f}")
        print(f"P-Value:           {metrics.get('p_value', 1):>8.4f}")
        print(f"Risk Level:        {metrics.get('risk_level', 'UNKNOWN'):>8}")
        print(f"Significance:      {metrics.get('statistical_significance', 'UNKNOWN')}")

        # Check thresholds
        passes, reason = self.metrics_calc.passes_thresholds(metrics)
        print(f"\nValidation: {'✅ PASS' if passes else '❌ FAIL'}")
        print(f"Reason: {reason}")
        print(f"{'='*80}\n")

        return {
            'success': True,
            'strategy_name': strategy_name,
            'metrics': metrics,
            'num_trades': len(returns),
            'passes_validation': passes,
            'validation_reason': reason
        }
