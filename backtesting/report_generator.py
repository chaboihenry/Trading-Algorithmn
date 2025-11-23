#!/Users/henry/miniconda3/envs/trading/bin/python
"""
Daily Trade Recommendation Report Generator

Generates comprehensive daily reports with:
- Strategy validation results
- Ensemble performance analysis
- Top 5 trade recommendations
- Signal quality metrics
- Risk warnings and compliance checks
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backtesting.strategy_validator import StrategyValidator
from backtesting.ensemble_validator import EnsembleValidator
from backtesting.trade_ranker import TradeRanker
from backtesting.metrics_calculator import MetricsCalculator


class ReportGenerator:
    """Generates daily trade recommendation reports"""

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db"):
        """
        Args:
            db_path: Path to database
        """
        self.db_path = db_path
        self.strategy_validator = StrategyValidator(db_path)
        self.ensemble_validator = EnsembleValidator(db_path)
        self.trade_ranker = TradeRanker(db_path)
        self.metrics_calc = MetricsCalculator()

    def generate_daily_report(self, num_trades: int = 5,
                             total_capital: float = 100_000,
                             output_dir: str = "backtesting/results",
                             lookback_days: int = 90) -> Dict[str, any]:
        """
        Generate comprehensive daily trading report

        Args:
            num_trades: Number of top trades to recommend
            total_capital: Total available capital
            output_dir: Directory to save reports
            lookback_days: Days of historical data for validation

        Returns:
            Report data and file paths
        """
        print("\n" + "="*80)
        print("DAILY TRADING REPORT GENERATOR")
        print("="*80)
        print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Capital: ${total_capital:,.0f}")
        print(f"Recommendations: Top {num_trades} trades")
        print("="*80)

        report_data = {
            'timestamp': datetime.now().isoformat(),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'config': {
                'num_trades': num_trades,
                'total_capital': total_capital,
                'lookback_days': lookback_days
            }
        }

        # Section 1: Strategy Validation
        print("\n" + "▶"*40)
        print("SECTION 1: STRATEGY VALIDATION")
        print("▶"*40)

        strategy_results = self._validate_all_strategies(lookback_days)
        report_data['strategy_validation'] = strategy_results

        # Section 2: Ensemble Analysis
        print("\n" + "▶"*40)
        print("SECTION 2: ENSEMBLE PERFORMANCE")
        print("▶"*40)

        ensemble_results = self.ensemble_validator.validate_ensemble(lookback_days)
        report_data['ensemble_analysis'] = ensemble_results

        # Section 3: Trade Recommendations
        print("\n" + "▶"*40)
        print("SECTION 3: TRADE RECOMMENDATIONS")
        print("▶"*40)

        top_trades = self.trade_ranker.get_current_top_trades(
            num_trades=num_trades,
            total_capital=total_capital
        )
        report_data['top_trades'] = top_trades

        # Section 4: Risk Assessment
        print("\n" + "▶"*40)
        print("SECTION 4: RISK ASSESSMENT")
        print("▶"*40)

        risk_assessment = self._assess_risks(
            strategy_results,
            ensemble_results,
            top_trades,
            total_capital
        )
        report_data['risk_assessment'] = risk_assessment

        # Section 5: Signal Quality Metrics
        print("\n" + "▶"*40)
        print("SECTION 5: SIGNAL QUALITY METRICS")
        print("▶"*40)

        signal_quality = self._analyze_signal_quality(lookback_days)
        report_data['signal_quality'] = signal_quality

        # Generate report files
        print("\n" + "▶"*40)
        print("SECTION 6: GENERATING REPORT FILES")
        print("▶"*40)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save detailed report (markdown)
        md_path = output_path / f"daily_report_{timestamp}.md"
        self._save_markdown_report(report_data, md_path)

        # Save trade recommendations (CSV)
        if len(top_trades) > 0:
            csv_path = output_path / f"trade_recommendations_{timestamp}.csv"
            self.trade_ranker.export_trades_to_csv(top_trades, str(csv_path))
            report_data['csv_path'] = str(csv_path)

        # Save summary (JSON)
        import json
        json_path = output_path / f"report_summary_{timestamp}.json"
        with open(json_path, 'w') as f:
            # Convert DataFrames to dicts for JSON serialization
            json_data = report_data.copy()
            if 'top_trades' in json_data and isinstance(json_data['top_trades'], pd.DataFrame):
                json_data['top_trades'] = json_data['top_trades'].to_dict('records')
            json.dump(json_data, f, indent=2, default=str)

        report_data['md_path'] = str(md_path)
        report_data['json_path'] = str(json_path)

        # Print summary
        print(f"\n{'='*80}")
        print("REPORT GENERATION COMPLETE")
        print(f"{'='*80}")
        print(f"\nFiles Generated:")
        print(f"  📄 Markdown Report: {md_path}")
        if 'csv_path' in report_data:
            print(f"  📊 Trade CSV:       {report_data['csv_path']}")
        print(f"  📋 JSON Summary:    {json_path}")
        print(f"\n{'='*80}\n")

        return report_data

    def _validate_all_strategies(self, lookback_days: int) -> Dict[str, any]:
        """Validate all strategies"""
        strategies = ['PairsTradingStrategy', 'SentimentTradingStrategy', 'EnsembleStrategy']
        results = {}

        for strategy in strategies:
            result = self.strategy_validator.quick_validation(strategy, lookback_days)
            results[strategy] = result

            if result.get('success'):
                status = "✅ PASS" if result.get('passes_validation') else "❌ FAIL"
                metrics = result['metrics']
                print(f"\n{status} {strategy}")
                print(f"  Sharpe: {metrics.get('sharpe_ratio', 0):.2f}, "
                      f"Return: {metrics.get('total_return', 0):.2%}, "
                      f"Trades: {result.get('num_trades', 0)}")

        return results

    def _assess_risks(self, strategy_results: Dict, ensemble_results: Dict,
                     top_trades: pd.DataFrame, total_capital: float) -> Dict[str, any]:
        """Assess trading risks"""
        risks = {
            'warnings': [],
            'alerts': [],
            'risk_score': 0,  # 0-10 scale
            'recommendation': ''
        }

        # Check strategy validation
        failing_strategies = []
        for strategy, result in strategy_results.items():
            if result.get('success') and not result.get('passes_validation'):
                failing_strategies.append(strategy)
                risks['warnings'].append(f"{strategy} failed validation thresholds")

        if len(failing_strategies) > 1:
            risks['alerts'].append("⚠️  Multiple strategies failing validation")
            risks['risk_score'] += 3

        # Check ensemble performance
        if ensemble_results.get('success'):
            if not ensemble_results.get('ensemble_beneficial'):
                risks['warnings'].append("Ensemble not outperforming individual strategies")
                risks['risk_score'] += 2

        # Check trade concentration
        if len(top_trades) > 0:
            max_position = top_trades['position_size'].max()
            if max_position > 0.15:  # More than 15%
                risks['alerts'].append(f"⚠️  High concentration: {max_position:.1%} in single position")
                risks['risk_score'] += 2

            total_allocation = top_trades['position_size'].sum()
            if total_allocation > 0.60:  # More than 60% allocated
                risks['warnings'].append(f"High total allocation: {total_allocation:.1%}")
                risks['risk_score'] += 1

        # Risk recommendation
        if risks['risk_score'] <= 3:
            risks['recommendation'] = "✅ LOW RISK - Safe to execute recommended trades"
        elif risks['risk_score'] <= 6:
            risks['recommendation'] = "⚠️  MEDIUM RISK - Execute with caution, monitor closely"
        else:
            risks['recommendation'] = "❌ HIGH RISK - Consider reducing position sizes or skipping"

        # Print risk assessment
        print(f"\nRisk Score: {risks['risk_score']}/10")
        print(f"Recommendation: {risks['recommendation']}")

        if risks['alerts']:
            print(f"\n⚠️  ALERTS:")
            for alert in risks['alerts']:
                print(f"  {alert}")

        if risks['warnings']:
            print(f"\n⚠️  Warnings:")
            for warning in risks['warnings']:
                print(f"  • {warning}")

        return risks

    def _analyze_signal_quality(self, lookback_days: int) -> Dict[str, any]:
        """Analyze recent signal quality"""
        import sqlite3
        from datetime import timedelta

        conn = sqlite3.connect(self.db_path)
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

        # Get signal statistics
        query = """
        SELECT
            strategy_name,
            COUNT(*) as num_signals,
            AVG(strength) as avg_strength,
            COUNT(DISTINCT symbol_ticker) as num_symbols,
            COUNT(CASE WHEN signal_type = 'BUY' THEN 1 END) as buy_signals,
            COUNT(CASE WHEN signal_type = 'SELL' THEN 1 END) as sell_signals
        FROM trading_signals
        WHERE signal_date >= ?
        GROUP BY strategy_name
        """

        df = pd.read_sql_query(query, conn, params=[cutoff_date])
        conn.close()

        if len(df) == 0:
            print("\n❌ No signals found for quality analysis")
            return {'success': False, 'error': 'No signals found'}

        # Print signal quality table
        print(f"\n{'Strategy':<30} {'Signals':>8} {'Avg Strength':>13} {'Symbols':>8} {'Buy/Sell':>12}")
        print("─"*80)

        for _, row in df.iterrows():
            strategy = row['strategy_name'].replace('Strategy', '')
            buy_sell_ratio = f"{row['buy_signals']}/{row['sell_signals']}"
            print(f"{strategy:<30} {row['num_signals']:>8.0f} {row['avg_strength']:>13.2f} "
                  f"{row['num_symbols']:>8.0f} {buy_sell_ratio:>12}")

        return {
            'success': True,
            'statistics': df.to_dict('records'),
            'total_signals': df['num_signals'].sum()
        }

    def _save_markdown_report(self, report_data: Dict, output_path: Path):
        """Save detailed markdown report"""
        md_content = f"""# Daily Trading Report
**Date:** {report_data['date']}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

"""

        # Risk assessment summary
        risk = report_data.get('risk_assessment', {})
        md_content += f"**Risk Level:** {risk.get('risk_score', 0)}/10  \n"
        md_content += f"**Recommendation:** {risk.get('recommendation', 'Unknown')}  \n\n"

        # Strategy validation summary
        md_content += "### Strategy Validation\n\n"
        for strategy, result in report_data.get('strategy_validation', {}).items():
            if result.get('success'):
                status = "✅ PASS" if result.get('passes_validation') else "❌ FAIL"
                metrics = result['metrics']
                md_content += f"- **{strategy}** {status}\n"
                md_content += f"  - Sharpe: {metrics.get('sharpe_ratio', 0):.2f}\n"
                md_content += f"  - Return: {metrics.get('total_return', 0):.2%}\n"
                md_content += f"  - Trades: {result.get('num_trades', 0)}\n\n"

        # Trade recommendations
        md_content += "---\n\n## Trade Recommendations\n\n"
        top_trades = report_data.get('top_trades')
        if isinstance(top_trades, pd.DataFrame) and len(top_trades) > 0:
            for i, row in top_trades.iterrows():
                direction = "LONG" if row['signal'] == 'BUY' else "SHORT"
                if direction == "LONG":
                    action = "BUY"
                    exit_action = "SELL"
                else:
                    action = "SELL"
                    exit_action = "BUY BACK"

                md_content += f"### #{i+1} {row['symbol']} ({direction})\n"
                md_content += f"- **{action} at:** ${row['close']:.2f}\n"
                md_content += f"- **{exit_action} at:** ${row['take_profit_price']:.2f}\n"
                md_content += f"- **Stop Loss:** ${row['stop_loss_price']:.2f}\n"
                md_content += f"- **Signal Strength:** {row['signal_strength']:.2f}\n\n"
        else:
            md_content += "❌ No trade recommendations generated.\n\n"

        # Risk warnings
        md_content += "---\n\n## Risk Assessment\n\n"
        if risk.get('alerts'):
            md_content += "### ⚠️  Alerts\n"
            for alert in risk['alerts']:
                md_content += f"- {alert}\n"
            md_content += "\n"

        if risk.get('warnings'):
            md_content += "### Warnings\n"
            for warning in risk['warnings']:
                md_content += f"- {warning}\n"
            md_content += "\n"

        # Signal quality
        md_content += "---\n\n## Signal Quality Metrics\n\n"
        signal_quality = report_data.get('signal_quality', {})
        if signal_quality.get('success'):
            md_content += "| Strategy | Signals | Avg Strength | Symbols | Buy/Sell |\n"
            md_content += "|----------|---------|--------------|---------|----------|\n"
            for stat in signal_quality.get('statistics', []):
                strategy = stat['strategy_name'].replace('Strategy', '')
                buy_sell = f"{stat['buy_signals']}/{stat['sell_signals']}"
                md_content += f"| {strategy} | {stat['num_signals']:.0f} | {stat['avg_strength']:.2f} | {stat['num_symbols']:.0f} | {buy_sell} |\n"

        md_content += "\n---\n\n"
        md_content += "*Report generated by Automated Trading System*  \n"
        md_content += f"*Database: {self.db_path}*\n"

        # Write to file
        with open(output_path, 'w') as f:
            f.write(md_content)

        print(f"\n✅ Markdown report saved to: {output_path}")


if __name__ == "__main__":
    """Generate daily report when executed directly"""
    generator = ReportGenerator()
    generator.generate_daily_report(
        num_trades=5,
        total_capital=100_000,
        output_dir="backtesting/results"
    )
