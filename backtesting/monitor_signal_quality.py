#!/Users/henry/miniconda3/envs/trading/bin/python
"""
Continuous Signal Quality Monitoring

Monitors trading signal quality in real-time:
- Tracks strategy performance metrics
- Detects degradation in signal quality
- Alerts when strategies need retraining
- Stores monitoring data in database
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backtesting.strategy_validator import StrategyValidator
from backtesting.metrics_calculator import MetricsCalculator


class SignalQualityMonitor:
    """Monitors signal quality and alerts on degradation"""

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db"):
        """
        Args:
            db_path: Path to database
        """
        self.db_path = db_path
        self.validator = StrategyValidator(db_path)
        self.metrics_calc = MetricsCalculator()
        self._ensure_monitoring_table()

    def _ensure_monitoring_table(self):
        """Create signal quality monitoring table if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS signal_quality_monitoring (
            monitoring_id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT NOT NULL,
            check_date DATE NOT NULL,
            lookback_days INTEGER NOT NULL,
            num_signals INTEGER,
            num_trades INTEGER,
            sharpe_ratio REAL,
            total_return REAL,
            max_drawdown REAL,
            win_rate REAL,
            profit_factor REAL,
            t_statistic REAL,
            p_value REAL,
            passes_validation BOOLEAN,
            quality_score REAL,
            alert_level TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_signal_monitoring_strategy_date
        ON signal_quality_monitoring(strategy_name, check_date)
        """)

        conn.commit()
        conn.close()

    def monitor_strategy(self, strategy_name: str, lookback_days: int = 30) -> Dict[str, any]:
        """
        Monitor single strategy signal quality

        Args:
            strategy_name: Name of strategy to monitor
            lookback_days: Days to analyze

        Returns:
            Monitoring results with alert level
        """
        print(f"\n{'─'*80}")
        print(f"Monitoring: {strategy_name}")
        print(f"{'─'*80}")

        # Validate strategy
        result = self.validator.quick_validation(strategy_name, lookback_days)

        if not result.get('success'):
            print(f"❌ Validation failed: {result.get('error')}")
            return {
                'success': False,
                'error': result.get('error'),
                'alert_level': 'CRITICAL'
            }

        # Extract metrics
        metrics = result['metrics']
        num_trades = result.get('num_trades', 0)

        # Calculate quality score (0-100)
        quality_score = self._calculate_quality_score(metrics, num_trades)

        # Determine alert level
        alert_level = self._determine_alert_level(
            metrics,
            num_trades,
            quality_score,
            result.get('passes_validation', False)
        )

        # Print results
        print(f"\nMetrics:")
        print(f"  Trades:       {num_trades:>6}")
        print(f"  Sharpe:       {metrics.get('sharpe_ratio', 0):>6.2f}")
        print(f"  Return:       {metrics.get('total_return', 0):>6.2%}")
        print(f"  Win Rate:     {metrics.get('win_rate', 0):>6.2%}")
        print(f"  Quality:      {quality_score:>6.1f}/100")
        print(f"\nAlert Level: {alert_level}")

        # Store in database
        self._store_monitoring_result(
            strategy_name=strategy_name,
            lookback_days=lookback_days,
            num_signals=num_trades,
            num_trades=num_trades,
            metrics=metrics,
            passes_validation=result.get('passes_validation', False),
            quality_score=quality_score,
            alert_level=alert_level
        )

        return {
            'success': True,
            'strategy_name': strategy_name,
            'metrics': metrics,
            'num_trades': num_trades,
            'quality_score': quality_score,
            'alert_level': alert_level,
            'passes_validation': result.get('passes_validation', False)
        }

    def monitor_all_strategies(self, lookback_days: int = 30) -> Dict[str, any]:
        """
        Monitor all strategies

        Args:
            lookback_days: Days to analyze

        Returns:
            Monitoring results for all strategies
        """
        print("\n" + "="*80)
        print("SIGNAL QUALITY MONITORING - ALL STRATEGIES")
        print("="*80)
        print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Lookback: {lookback_days} days")
        print("="*80)

        strategies = [
            'PairsTradingStrategy',
            'SentimentTradingStrategy',
            'EnsembleStrategy'
        ]

        results = {}
        alerts = []

        for strategy in strategies:
            result = self.monitor_strategy(strategy, lookback_days)
            results[strategy] = result

            # Collect alerts
            if result.get('alert_level') in ['WARNING', 'CRITICAL']:
                alerts.append({
                    'strategy': strategy,
                    'level': result.get('alert_level'),
                    'quality_score': result.get('quality_score', 0)
                })

        # Print summary
        print(f"\n{'='*80}")
        print("MONITORING SUMMARY")
        print(f"{'='*80}")

        print(f"\n{'Strategy':<30} {'Quality':>8} {'Alert':>12} {'Status':>10}")
        print("─"*80)

        for strategy, result in results.items():
            if result.get('success'):
                strategy_short = strategy.replace('Strategy', '')
                quality = result.get('quality_score', 0)
                alert = result.get('alert_level', 'UNKNOWN')
                status = "✅ PASS" if result.get('passes_validation') else "❌ FAIL"
                print(f"{strategy_short:<30} {quality:>8.1f} {alert:>12} {status:>10}")

        # Print alerts
        if alerts:
            print(f"\n{'─'*80}")
            print("⚠️  ALERTS REQUIRING ATTENTION")
            print("─"*80)
            for alert in alerts:
                print(f"  {alert['level']}: {alert['strategy']} (Quality: {alert['quality_score']:.1f}/100)")

                if alert['level'] == 'CRITICAL':
                    print(f"    → ACTION: Strategy needs immediate review and possible retraining")
                elif alert['level'] == 'WARNING':
                    print(f"    → ACTION: Monitor closely, consider retraining if persists")

        print(f"\n{'='*80}\n")

        return {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'alerts': alerts,
            'num_alerts': len(alerts)
        }

    def _calculate_quality_score(self, metrics: Dict[str, float], num_trades: int) -> float:
        """
        Calculate overall quality score (0-100)

        Components:
        - Sharpe ratio (40%)
        - Win rate (20%)
        - Profit factor (20%)
        - Trade volume (10%)
        - Statistical significance (10%)
        """
        score = 0

        # Sharpe ratio component (40 points max)
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe >= 2.0:
            sharpe_score = 40
        elif sharpe >= 1.0:
            sharpe_score = 20 + (sharpe - 1.0) * 20
        elif sharpe >= 0:
            sharpe_score = sharpe * 20
        else:
            sharpe_score = max(0, 20 + sharpe * 20)  # Penalty for negative Sharpe
        score += sharpe_score

        # Win rate component (20 points max)
        win_rate = metrics.get('win_rate', 0)
        if win_rate >= 0.60:
            win_rate_score = 20
        elif win_rate >= 0.50:
            win_rate_score = (win_rate - 0.50) / 0.10 * 20
        else:
            win_rate_score = max(0, win_rate / 0.50 * 10)
        score += win_rate_score

        # Profit factor component (20 points max)
        pf = metrics.get('profit_factor', 0)
        if pf >= 2.0:
            pf_score = 20
        elif pf >= 1.0:
            pf_score = (pf - 1.0) * 20
        else:
            pf_score = 0
        score += pf_score

        # Trade volume component (10 points max)
        if num_trades >= 20:
            volume_score = 10
        elif num_trades >= 10:
            volume_score = 5 + (num_trades - 10) / 10 * 5
        elif num_trades >= 5:
            volume_score = (num_trades / 10) * 5
        else:
            volume_score = 0
        score += volume_score

        # Statistical significance component (10 points max)
        t_stat = abs(metrics.get('t_statistic', 0))
        p_value = metrics.get('p_value', 1.0)

        if t_stat >= 2.0 and p_value <= 0.05:
            sig_score = 10
        elif t_stat >= 1.5 and p_value <= 0.10:
            sig_score = 5
        else:
            sig_score = 0
        score += sig_score

        return min(100, max(0, score))

    def _determine_alert_level(self, metrics: Dict[str, float], num_trades: int,
                               quality_score: float, passes_validation: bool) -> str:
        """Determine alert level based on metrics and quality score"""
        # CRITICAL: Strategy is broken or dangerous
        if quality_score < 30:
            return "CRITICAL"
        if num_trades > 5 and metrics.get('sharpe_ratio', 0) < -1.0:
            return "CRITICAL"
        if metrics.get('max_drawdown', 0) < -0.30:  # -30% drawdown
            return "CRITICAL"

        # WARNING: Strategy is degrading
        if quality_score < 50:
            return "WARNING"
        if not passes_validation:
            return "WARNING"
        if num_trades > 0 and metrics.get('sharpe_ratio', 0) < 0.5:
            return "WARNING"

        # ATTENTION: Monitor closely
        if quality_score < 70:
            return "ATTENTION"
        if num_trades < 5:
            return "ATTENTION"

        # GOOD: Strategy performing well
        return "GOOD"

    def _store_monitoring_result(self, strategy_name: str, lookback_days: int,
                                 num_signals: int, num_trades: int,
                                 metrics: Dict[str, float], passes_validation: bool,
                                 quality_score: float, alert_level: str):
        """Store monitoring result in database"""
        conn = sqlite3.connect(self.db_path)

        conn.execute("""
        INSERT INTO signal_quality_monitoring (
            strategy_name, check_date, lookback_days,
            num_signals, num_trades,
            sharpe_ratio, total_return, max_drawdown, win_rate, profit_factor,
            t_statistic, p_value,
            passes_validation, quality_score, alert_level
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            strategy_name,
            datetime.now().strftime('%Y-%m-%d'),
            lookback_days,
            num_signals,
            num_trades,
            metrics.get('sharpe_ratio'),
            metrics.get('total_return'),
            metrics.get('max_drawdown'),
            metrics.get('win_rate'),
            metrics.get('profit_factor'),
            metrics.get('t_statistic'),
            metrics.get('p_value'),
            passes_validation,
            quality_score,
            alert_level
        ))

        conn.commit()
        conn.close()

    def get_quality_trend(self, strategy_name: str, days: int = 30) -> pd.DataFrame:
        """
        Get quality score trend over time

        Args:
            strategy_name: Strategy to analyze
            days: Days of history to retrieve

        Returns:
            DataFrame with quality trend
        """
        conn = sqlite3.connect(self.db_path)

        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        query = """
        SELECT
            check_date,
            quality_score,
            alert_level,
            sharpe_ratio,
            win_rate,
            num_trades
        FROM signal_quality_monitoring
        WHERE strategy_name = ?
          AND check_date >= ?
        ORDER BY check_date
        """

        df = pd.read_sql_query(query, conn, params=[strategy_name, cutoff_date])
        conn.close()

        return df


if __name__ == "__main__":
    """Run signal quality monitoring when executed directly"""
    monitor = SignalQualityMonitor()
    monitor.monitor_all_strategies(lookback_days=30)
