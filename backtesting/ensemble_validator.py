"""
Ensemble Strategy Validator

Tests combined strategies and validates ensemble performance:
- Validates EnsembleStrategy from strategies/run_strategies.py
- Tests signal aggregation and weighting
- Compares ensemble vs individual strategy performance
- Analyzes correlation and diversification benefits
"""

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from backtesting.metrics_calculator import MetricsCalculator
from backtesting.strategy_validator import StrategyValidator


class EnsembleValidator:
    """Validates ensemble strategy performance and combination logic"""

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db"):
        """
        Args:
            db_path: Path to database
        """
        self.db_path = db_path
        self.metrics_calc = MetricsCalculator()
        self.strategy_validator = StrategyValidator(db_path)

    def validate_ensemble(self, lookback_days: int = 90) -> Dict[str, any]:
        """
        Validate ensemble strategy performance

        Args:
            lookback_days: Days of historical data to analyze

        Returns:
            Ensemble validation results with comparison to individual strategies
        """
        print("\n" + "="*80)
        print("ENSEMBLE STRATEGY VALIDATION")
        print("="*80)

        # Get individual strategy results
        print("\nStep 1: Validating Individual Strategies")
        print("─"*80)

        individual_results = {}
        strategy_names = ['PairsTradingStrategy', 'SentimentTradingStrategy', 'VolatilityTradingStrategy']

        for strategy in strategy_names:
            print(f"\n  Validating {strategy}...")
            result = self.strategy_validator.quick_validation(strategy, lookback_days)

            if result.get('success'):
                individual_results[strategy] = result
                metrics = result['metrics']
                print(f"    Sharpe: {metrics.get('sharpe_ratio', 0):.2f}, "
                      f"Return: {metrics.get('total_return', 0):.2%}, "
                      f"Trades: {result.get('num_trades', 0)}")
            else:
                print(f"    ❌ Failed: {result.get('error', 'Unknown error')}")

        # Get ensemble results
        print(f"\n{'─'*80}")
        print("Step 2: Validating Ensemble Strategy")
        print("─"*80)

        ensemble_result = self.strategy_validator.quick_validation('EnsembleStrategy', lookback_days)

        if not ensemble_result.get('success'):
            print(f"\n❌ Ensemble validation failed: {ensemble_result.get('error', 'Unknown error')}")
            return {
                'success': False,
                'error': ensemble_result.get('error'),
                'individual_results': individual_results
            }

        ensemble_metrics = ensemble_result['metrics']
        print(f"\nEnsemble Results:")
        print(f"  Sharpe:  {ensemble_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Return:  {ensemble_metrics.get('total_return', 0):.2%}")
        print(f"  Trades:  {ensemble_result.get('num_trades', 0)}")

        # Compare ensemble vs individual
        print(f"\n{'─'*80}")
        print("Step 3: Performance Comparison")
        print("─"*80)

        comparison = self._compare_performance(ensemble_metrics, individual_results)

        # Print comparison table
        print(f"\n{'Strategy':<30} {'Sharpe':>8} {'Return':>8} {'Win Rate':>10} {'Max DD':>10}")
        print("─"*80)

        for strategy, result in individual_results.items():
            if result.get('success'):
                m = result['metrics']
                strategy_short = strategy.replace('Strategy', '')
                print(f"{strategy_short:<30} {m.get('sharpe_ratio', 0):>8.2f} "
                      f"{m.get('total_return', 0):>8.2%} {m.get('win_rate', 0):>10.2%} "
                      f"{m.get('max_drawdown', 0):>10.2%}")

        print(f"{'Ensemble':<30} {ensemble_metrics.get('sharpe_ratio', 0):>8.2f} "
              f"{ensemble_metrics.get('total_return', 0):>8.2%} "
              f"{ensemble_metrics.get('win_rate', 0):>10.2%} "
              f"{ensemble_metrics.get('max_drawdown', 0):>10.2%}")

        # Ensemble benefits
        print(f"\n{'─'*80}")
        print("Ensemble Benefits:")
        print("─"*80)
        print(f"  Sharpe Improvement:   {comparison['sharpe_improvement']:>6.1f}%")
        print(f"  Return Improvement:   {comparison['return_improvement']:>6.1f}%")
        print(f"  Drawdown Reduction:   {comparison['drawdown_reduction']:>6.1f}%")
        print(f"  Diversification:      {comparison['diversification_score']:>6.2f}/10")

        # Overall assessment
        print(f"\n{'='*80}")
        assessment = self._assess_ensemble(comparison)
        print(f"Assessment: {assessment['verdict']}")
        print(f"Reason: {assessment['reason']}")
        print("="*80 + "\n")

        return {
            'success': True,
            'ensemble_metrics': ensemble_metrics,
            'individual_results': individual_results,
            'comparison': comparison,
            'assessment': assessment,
            'ensemble_beneficial': comparison['sharpe_improvement'] > 0
        }

    def _compare_performance(self, ensemble_metrics: Dict[str, float],
                            individual_results: Dict[str, Dict]) -> Dict[str, float]:
        """
        Compare ensemble vs individual strategy performance

        Args:
            ensemble_metrics: Ensemble performance metrics
            individual_results: Individual strategy results

        Returns:
            Comparison metrics
        """
        # Get average of individual strategies
        individual_sharpes = []
        individual_returns = []
        individual_drawdowns = []

        for result in individual_results.values():
            if result.get('success'):
                m = result['metrics']
                individual_sharpes.append(m.get('sharpe_ratio', 0))
                individual_returns.append(m.get('total_return', 0))
                individual_drawdowns.append(m.get('max_drawdown', 0))

        avg_sharpe = np.mean(individual_sharpes) if individual_sharpes else 0
        avg_return = np.mean(individual_returns) if individual_returns else 0
        avg_drawdown = np.mean(individual_drawdowns) if individual_drawdowns else 0

        # Calculate improvements
        sharpe_improvement = ((ensemble_metrics.get('sharpe_ratio', 0) - avg_sharpe) / abs(avg_sharpe) * 100
                             if avg_sharpe != 0 else 0)

        return_improvement = ((ensemble_metrics.get('total_return', 0) - avg_return) / abs(avg_return) * 100
                             if avg_return != 0 else 0)

        drawdown_reduction = ((avg_drawdown - ensemble_metrics.get('max_drawdown', 0)) / abs(avg_drawdown) * 100
                             if avg_drawdown != 0 else 0)

        # Diversification score (0-10)
        # Higher score = better diversification benefits
        sharpe_benefit = min(10, max(0, sharpe_improvement / 10))
        dd_benefit = min(10, max(0, drawdown_reduction / 10))
        diversification_score = (sharpe_benefit + dd_benefit) / 2

        return {
            'avg_individual_sharpe': avg_sharpe,
            'ensemble_sharpe': ensemble_metrics.get('sharpe_ratio', 0),
            'sharpe_improvement': sharpe_improvement,
            'avg_individual_return': avg_return,
            'ensemble_return': ensemble_metrics.get('total_return', 0),
            'return_improvement': return_improvement,
            'avg_individual_drawdown': avg_drawdown,
            'ensemble_drawdown': ensemble_metrics.get('max_drawdown', 0),
            'drawdown_reduction': drawdown_reduction,
            'diversification_score': diversification_score
        }

    def _assess_ensemble(self, comparison: Dict[str, float]) -> Dict[str, str]:
        """
        Assess whether ensemble provides meaningful benefits

        Args:
            comparison: Performance comparison metrics

        Returns:
            Assessment verdict and reason
        """
        sharpe_improvement = comparison['sharpe_improvement']
        drawdown_reduction = comparison['drawdown_reduction']
        diversification = comparison['diversification_score']

        # Criteria for beneficial ensemble
        if sharpe_improvement > 10 and drawdown_reduction > 10:
            verdict = "✅ EXCELLENT - Ensemble strongly recommended"
            reason = f"Sharpe improved {sharpe_improvement:.1f}%, DD reduced {drawdown_reduction:.1f}%"

        elif sharpe_improvement > 5 or drawdown_reduction > 5:
            verdict = "✅ GOOD - Ensemble beneficial"
            reason = f"Moderate improvements in risk-adjusted returns"

        elif sharpe_improvement > 0 and drawdown_reduction > 0:
            verdict = "⚠️  MARGINAL - Ensemble slightly better"
            reason = f"Small improvements, monitor ongoing performance"

        elif diversification >= 5:
            verdict = "⚠️  MIXED - Some diversification benefits"
            reason = f"Diversification score {diversification:.1f}/10, but limited performance gain"

        else:
            verdict = "❌ POOR - Individual strategies may be better"
            reason = f"Sharpe {sharpe_improvement:.1f}%, DD {drawdown_reduction:.1f}% - no clear benefit"

        return {
            'verdict': verdict,
            'reason': reason
        }

    def analyze_signal_correlation(self, lookback_days: int = 90) -> Dict[str, any]:
        """
        Analyze correlation between individual strategy signals

        Args:
            lookback_days: Days to analyze

        Returns:
            Signal correlation analysis
        """
        print("\n" + "="*80)
        print("SIGNAL CORRELATION ANALYSIS")
        print("="*80)

        conn = sqlite3.connect(self.db_path)

        # Get recent date
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

        # Load signals from each strategy
        query = """
        SELECT
            symbol_ticker as symbol,
            signal_date as date,
            strategy_name,
            signal_type as signal,
            strength
        FROM trading_signals
        WHERE signal_date >= ?
          AND strategy_name IN ('PairsTradingStrategy', 'SentimentTradingStrategy', 'VolatilityTradingStrategy')
        ORDER BY signal_date, symbol_ticker
        """

        df = pd.read_sql_query(query, conn, params=[cutoff_date])
        conn.close()

        if len(df) == 0:
            print("\n❌ No signals found for correlation analysis")
            return {'success': False, 'error': 'No signals found'}

        # Pivot to get strategy signals side by side
        df['signal_numeric'] = df['signal'].map({'BUY': 1, 'SELL': -1, 'HOLD': 0})

        pivot = df.pivot_table(
            values='signal_numeric',
            index=['date', 'symbol'],
            columns='strategy_name',
            aggfunc='first'
        ).reset_index()

        # Calculate correlations
        strategies = ['PairsTradingStrategy', 'SentimentTradingStrategy', 'VolatilityTradingStrategy']
        correlation_matrix = pivot[strategies].corr()

        print(f"\nSignal Correlation Matrix:")
        print(f"{'─'*80}")
        print(correlation_matrix.to_string())

        # Agreement rates (how often strategies agree on direction)
        print(f"\n{'─'*80}")
        print("Signal Agreement Rates:")
        print(f"{'─'*80}")

        for i, strat1 in enumerate(strategies):
            for strat2 in strategies[i+1:]:
                if strat1 in pivot.columns and strat2 in pivot.columns:
                    valid = pivot[[strat1, strat2]].dropna()
                    if len(valid) > 0:
                        agreement = np.mean(np.sign(valid[strat1]) == np.sign(valid[strat2]))
                        print(f"  {strat1.replace('Strategy', '')} ↔ {strat2.replace('Strategy', '')}: "
                              f"{agreement:.2%}")

        print("="*80 + "\n")

        return {
            'success': True,
            'correlation_matrix': correlation_matrix.to_dict(),
            'num_signals': len(df)
        }


if __name__ == "__main__":
    """Run ensemble validation when executed directly"""
    validator = EnsembleValidator()
    validator.validate_ensemble(lookback_days=90)
    validator.analyze_signal_correlation(lookback_days=90)
