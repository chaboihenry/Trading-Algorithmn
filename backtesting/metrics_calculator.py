"""
Industry-Standard Validation Metrics Calculator

Provides comprehensive metrics for:
- Classification Models: Precision, Recall, F1, Accuracy, MCC
- Regression Models: RMSE, MAE, R², Directional Accuracy
- Trading Performance: Sharpe, Sortino, Calmar, Information Ratio, Max Drawdown, Win Rate, Profit Factor
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy import stats


class MetricsCalculator:
    """Calculate industry-standard validation metrics"""

    def __init__(self, risk_free_rate: float = 0.04):
        """
        Args:
            risk_free_rate: Annual risk-free rate (default 4%)
        """
        self.risk_free_rate = risk_free_rate

    # ========== Classification Metrics ==========

    def classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate classification metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)

        Returns:
            Dictionary with precision, recall, f1, accuracy, mcc
        """
        from sklearn.metrics import (precision_score, recall_score, f1_score,
                                     accuracy_score, matthews_corrcoef)

        metrics = {
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
            'mcc': matthews_corrcoef(y_true, y_pred)
        }

        return metrics

    # ========== Regression Metrics ==========

    def regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary with rmse, mae, r2, directional_accuracy
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        # Standard regression metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Directional accuracy (did we predict direction correctly?)
        y_true_diff = np.diff(y_true)
        y_pred_diff = np.diff(y_pred)
        directional_accuracy = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff))

        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_accuracy
        }

        return metrics

    # ========== Trading Performance Metrics ==========

    def sharpe_ratio(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        """
        Calculate annualized Sharpe ratio

        Args:
            returns: Array of returns
            periods_per_year: 252 for daily, 52 for weekly

        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        sharpe = np.mean(excess_returns) / np.std(returns) * np.sqrt(periods_per_year)
        return sharpe

    def sortino_ratio(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        """
        Calculate annualized Sortino ratio (uses downside deviation)

        Args:
            returns: Array of returns
            periods_per_year: 252 for daily, 52 for weekly

        Returns:
            Annualized Sortino ratio
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0

        downside_std = np.std(downside_returns)
        sortino = np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year)
        return sortino

    def calmar_ratio(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown)

        Args:
            returns: Array of returns
            periods_per_year: 252 for daily, 52 for weekly

        Returns:
            Calmar ratio
        """
        if len(returns) == 0:
            return 0.0

        annual_return = np.mean(returns) * periods_per_year
        max_dd = self.max_drawdown(returns)

        if max_dd == 0:
            return 0.0

        calmar = annual_return / abs(max_dd)
        return calmar

    def information_ratio(self, returns: np.ndarray, benchmark_returns: np.ndarray,
                         periods_per_year: int = 252) -> float:
        """
        Calculate Information Ratio (excess return / tracking error)

        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            periods_per_year: 252 for daily

        Returns:
            Annualized Information Ratio
        """
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0

        excess_returns = returns - benchmark_returns
        tracking_error = np.std(excess_returns)

        if tracking_error == 0:
            return 0.0

        ir = np.mean(excess_returns) / tracking_error * np.sqrt(periods_per_year)
        return ir

    def max_drawdown(self, returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown

        Args:
            returns: Array of returns

        Returns:
            Maximum drawdown (negative value)
        """
        if len(returns) == 0:
            return 0.0

        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = np.min(drawdown)
        return max_dd

    def win_rate(self, returns: np.ndarray) -> float:
        """
        Calculate win rate (percentage of positive returns)

        Args:
            returns: Array of returns

        Returns:
            Win rate (0-1)
        """
        if len(returns) == 0:
            return 0.0

        win_rate = np.sum(returns > 0) / len(returns)
        return win_rate

    def profit_factor(self, returns: np.ndarray) -> float:
        """
        Calculate profit factor (gross profit / gross loss)

        Args:
            returns: Array of returns

        Returns:
            Profit factor
        """
        if len(returns) == 0:
            return 0.0

        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        pf = gross_profit / gross_loss
        return pf

    def trading_metrics(self, returns: np.ndarray,
                       benchmark_returns: Optional[np.ndarray] = None,
                       periods_per_year: int = 252) -> Dict[str, float]:
        """
        Calculate comprehensive trading performance metrics

        Args:
            returns: Array of returns
            benchmark_returns: Optional benchmark returns for IR
            periods_per_year: 252 for daily, 52 for weekly

        Returns:
            Dictionary with all trading metrics
        """
        metrics = {
            'sharpe_ratio': self.sharpe_ratio(returns, periods_per_year),
            'sortino_ratio': self.sortino_ratio(returns, periods_per_year),
            'calmar_ratio': self.calmar_ratio(returns, periods_per_year),
            'max_drawdown': self.max_drawdown(returns),
            'win_rate': self.win_rate(returns),
            'profit_factor': self.profit_factor(returns),
            'total_return': np.sum(returns),
            'annual_return': np.mean(returns) * periods_per_year,
            'annual_volatility': np.std(returns) * np.sqrt(periods_per_year)
        }

        # Add Information Ratio if benchmark provided
        if benchmark_returns is not None:
            metrics['information_ratio'] = self.information_ratio(
                returns, benchmark_returns, periods_per_year
            )

        return metrics

    # ========== Statistical Significance ==========

    def statistical_significance(self, returns: np.ndarray) -> Tuple[float, float]:
        """
        Calculate statistical significance of returns

        Args:
            returns: Array of returns

        Returns:
            Tuple of (t_statistic, p_value)
        """
        if len(returns) < 2:
            return 0.0, 1.0

        # One-sample t-test: are returns significantly different from zero?
        t_stat, p_value = stats.ttest_1samp(returns, 0)

        return t_stat, p_value

    # ========== Validation Thresholds ==========

    def passes_thresholds(self, metrics: Dict[str, float],
                         min_sharpe: float = 1.0,
                         max_drawdown: float = -0.15,
                         min_win_rate: float = 0.55,
                         min_profit_factor: float = 1.5,
                         min_t_stat: float = 2.0,
                         max_p_value: float = 0.05) -> Tuple[bool, str]:
        """
        Check if strategy passes minimum validation thresholds

        Args:
            metrics: Dictionary of calculated metrics
            min_sharpe: Minimum Sharpe ratio (default 1.0)
            max_drawdown: Maximum acceptable drawdown (default -15%)
            min_win_rate: Minimum win rate (default 55%)
            min_profit_factor: Minimum profit factor (default 1.5)
            min_t_stat: Minimum t-statistic (default 2.0)
            max_p_value: Maximum p-value (default 0.05)

        Returns:
            Tuple of (passes, failure_reason)
        """
        # Check Sharpe ratio
        if metrics.get('sharpe_ratio', 0) < min_sharpe:
            return False, f"Sharpe ratio {metrics.get('sharpe_ratio', 0):.2f} < {min_sharpe}"

        # Check max drawdown
        if metrics.get('max_drawdown', 0) < max_drawdown:
            return False, f"Max drawdown {metrics.get('max_drawdown', 0):.2%} < {max_drawdown:.2%}"

        # Check win rate
        if metrics.get('win_rate', 0) < min_win_rate:
            return False, f"Win rate {metrics.get('win_rate', 0):.2%} < {min_win_rate:.2%}"

        # Check profit factor
        if metrics.get('profit_factor', 0) < min_profit_factor:
            return False, f"Profit factor {metrics.get('profit_factor', 0):.2f} < {min_profit_factor}"

        # Check statistical significance
        if 't_statistic' in metrics and 'p_value' in metrics:
            if metrics['t_statistic'] < min_t_stat:
                return False, f"T-statistic {metrics['t_statistic']:.2f} < {min_t_stat}"
            if metrics['p_value'] > max_p_value:
                return False, f"P-value {metrics['p_value']:.4f} > {max_p_value}"

        return True, "All thresholds passed"

    # ========== Comprehensive Report ==========

    def generate_report(self, returns: np.ndarray,
                       y_true: Optional[np.ndarray] = None,
                       y_pred: Optional[np.ndarray] = None,
                       benchmark_returns: Optional[np.ndarray] = None,
                       periods_per_year: int = 252) -> Dict[str, any]:
        """
        Generate comprehensive metrics report

        Args:
            returns: Trading returns
            y_true: True labels/values for classification/regression metrics
            y_pred: Predicted labels/values
            benchmark_returns: Optional benchmark returns
            periods_per_year: 252 for daily

        Returns:
            Complete metrics report
        """
        report = {}

        # Trading metrics (always calculated)
        report['trading'] = self.trading_metrics(returns, benchmark_returns, periods_per_year)

        # Statistical significance
        t_stat, p_value = self.statistical_significance(returns)
        report['trading']['t_statistic'] = t_stat
        report['trading']['p_value'] = p_value

        # Classification metrics (if labels provided)
        if y_true is not None and y_pred is not None:
            if len(np.unique(y_true)) <= 10:  # Classification
                report['classification'] = self.classification_metrics(y_true, y_pred)
            else:  # Regression
                report['regression'] = self.regression_metrics(y_true, y_pred)

        # Validation check
        passes, reason = self.passes_thresholds(report['trading'])
        report['validation'] = {
            'passes': passes,
            'reason': reason
        }

        return report

    def print_report(self, report: Dict[str, any]):
        """Print formatted metrics report"""
        print("\n" + "="*80)
        print("VALIDATION METRICS REPORT")
        print("="*80)

        # Trading metrics
        if 'trading' in report:
            print("\nTRADING PERFORMANCE:")
            print("-" * 80)
            t = report['trading']
            print(f"  Sharpe Ratio:        {t.get('sharpe_ratio', 0):>8.2f}")
            print(f"  Sortino Ratio:       {t.get('sortino_ratio', 0):>8.2f}")
            print(f"  Calmar Ratio:        {t.get('calmar_ratio', 0):>8.2f}")
            if 'information_ratio' in t:
                print(f"  Information Ratio:   {t.get('information_ratio', 0):>8.2f}")
            print(f"  Max Drawdown:        {t.get('max_drawdown', 0):>8.2%}")
            print(f"  Win Rate:            {t.get('win_rate', 0):>8.2%}")
            print(f"  Profit Factor:       {t.get('profit_factor', 0):>8.2f}")
            print(f"  Total Return:        {t.get('total_return', 0):>8.2%}")
            print(f"  Annual Return:       {t.get('annual_return', 0):>8.2%}")
            print(f"  Annual Volatility:   {t.get('annual_volatility', 0):>8.2%}")
            print(f"\n  Statistical Significance:")
            print(f"    T-Statistic:       {t.get('t_statistic', 0):>8.2f}")
            print(f"    P-Value:           {t.get('p_value', 1):>8.4f}")

        # Classification metrics
        if 'classification' in report:
            print("\nCLASSIFICATION METRICS:")
            print("-" * 80)
            c = report['classification']
            print(f"  Precision:           {c.get('precision', 0):>8.2%}")
            print(f"  Recall:              {c.get('recall', 0):>8.2%}")
            print(f"  F1 Score:            {c.get('f1', 0):>8.2%}")
            print(f"  Accuracy:            {c.get('accuracy', 0):>8.2%}")
            print(f"  MCC:                 {c.get('mcc', 0):>8.2f}")

        # Regression metrics
        if 'regression' in report:
            print("\nREGRESSION METRICS:")
            print("-" * 80)
            r = report['regression']
            print(f"  RMSE:                {r.get('rmse', 0):>8.4f}")
            print(f"  MAE:                 {r.get('mae', 0):>8.4f}")
            print(f"  R²:                  {r.get('r2', 0):>8.2%}")
            print(f"  Directional Acc:     {r.get('directional_accuracy', 0):>8.2%}")

        # Validation result
        if 'validation' in report:
            print("\nVALIDATION RESULT:")
            print("-" * 80)
            v = report['validation']
            status = "✅ PASS" if v.get('passes', False) else "❌ FAIL"
            print(f"  Status: {status}")
            print(f"  Reason: {v.get('reason', 'Unknown')}")

        print("="*80 + "\n")
