"""
Performance Metrics Calculator
==============================
Comprehensive metrics for evaluating strategy performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Calculate comprehensive performance metrics for backtested strategies
    """

    def __init__(self, trades_df: pd.DataFrame, portfolio_values: List[Tuple[str, float]],
                 initial_capital: float, risk_free_rate: float = 0.04):
        """
        Initialize performance calculator

        Args:
            trades_df: DataFrame with trade results
            portfolio_values: List of (date, value) tuples
            initial_capital: Starting capital
            risk_free_rate: Annual risk-free rate (default 4%)
        """
        self.trades_df = trades_df
        self.portfolio_values = portfolio_values
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate

    def calculate_all_metrics(self) -> Dict:
        """Calculate all performance metrics"""
        metrics = {}

        # Basic metrics
        metrics.update(self._calculate_basic_metrics())

        # Return metrics
        metrics.update(self._calculate_return_metrics())

        # Risk metrics
        metrics.update(self._calculate_risk_metrics())

        # Risk-adjusted metrics
        metrics.update(self._calculate_risk_adjusted_metrics())

        # Trade statistics
        metrics.update(self._calculate_trade_statistics())

        # Win/Loss analysis
        metrics.update(self._calculate_win_loss_metrics())

        # Drawdown analysis
        metrics.update(self._calculate_drawdown_metrics())

        # Time-based metrics
        metrics.update(self._calculate_time_metrics())

        return metrics

    def _calculate_basic_metrics(self) -> Dict:
        """Calculate basic performance metrics"""
        if not self.portfolio_values:
            return {
                'total_trades': len(self.trades_df),
                'initial_capital': self.initial_capital,
                'final_capital': self.initial_capital,
                'total_return': 0.0,
                'total_return_pct': 0.0
            }

        final_value = self.portfolio_values[-1][1]
        total_return = final_value - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100

        return {
            'total_trades': len(self.trades_df),
            'initial_capital': self.initial_capital,
            'final_capital': final_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct
        }

    def _calculate_return_metrics(self) -> Dict:
        """Calculate return-based metrics"""
        if self.trades_df.empty:
            return {
                'avg_return_per_trade': 0.0,
                'avg_return_pct_per_trade': 0.0,
                'best_trade': 0.0,
                'worst_trade': 0.0,
                'median_return': 0.0
            }

        return {
            'avg_return_per_trade': self.trades_df['pnl'].mean(),
            'avg_return_pct_per_trade': self.trades_df['pnl_pct'].mean() * 100,
            'best_trade': self.trades_df['pnl'].max(),
            'worst_trade': self.trades_df['pnl'].min(),
            'median_return': self.trades_df['pnl'].median(),
            'std_return': self.trades_df['pnl'].std()
        }

    def _calculate_risk_metrics(self) -> Dict:
        """Calculate risk metrics"""
        if len(self.portfolio_values) < 2:
            return {
                'volatility': 0.0,
                'downside_volatility': 0.0,
                'value_at_risk_95': 0.0,
                'conditional_var_95': 0.0
            }

        # Calculate daily returns
        dates, values = zip(*self.portfolio_values)
        returns = pd.Series(values).pct_change().dropna()

        # Annualized volatility (assuming ~252 trading days)
        volatility = returns.std() * np.sqrt(252) * 100

        # Downside volatility (only negative returns)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) * 100 if len(downside_returns) > 0 else 0

        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5) * 100

        # Conditional VaR (expected shortfall)
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100 if len(returns) > 0 else 0

        return {
            'volatility': volatility,
            'downside_volatility': downside_volatility,
            'value_at_risk_95': var_95,
            'conditional_var_95': cvar_95
        }

    def _calculate_risk_adjusted_metrics(self) -> Dict:
        """Calculate risk-adjusted performance metrics"""
        if len(self.portfolio_values) < 2:
            return {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'information_ratio': 0.0
            }

        # Calculate daily returns
        dates, values = zip(*self.portfolio_values)
        returns = pd.Series(values).pct_change().dropna()

        if len(returns) == 0 or returns.std() == 0:
            return {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'information_ratio': 0.0
            }

        # Annualized return
        days = len(returns)
        annualized_return = (1 + returns.mean()) ** 252 - 1

        # Sharpe Ratio
        excess_return = annualized_return - self.risk_free_rate
        sharpe_ratio = excess_return / (returns.std() * np.sqrt(252))

        # Sortino Ratio (using downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else returns.std() * np.sqrt(252)
        sortino_ratio = excess_return / downside_std if downside_std > 0 else 0

        # Calmar Ratio (return / max drawdown)
        drawdown_metrics = self._calculate_drawdown_metrics()
        max_dd = abs(drawdown_metrics.get('max_drawdown_pct', 1))
        calmar_ratio = (annualized_return * 100) / max_dd if max_dd > 0 else 0

        # Information Ratio (assuming benchmark return is risk-free rate)
        tracking_error = returns.std() * np.sqrt(252)
        information_ratio = excess_return / tracking_error if tracking_error > 0 else 0

        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio
        }

    def _calculate_trade_statistics(self) -> Dict:
        """Calculate trade-level statistics"""
        if self.trades_df.empty:
            return {
                'avg_holding_days': 0.0,
                'trades_per_symbol': 0.0,
                'avg_position_size': 0.0,
                'total_commission_cost': 0.0
            }

        # Calculate holding days
        self.trades_df['holding_days'] = (
            pd.to_datetime(self.trades_df['exit_date']) -
            pd.to_datetime(self.trades_df['entry_date'])
        ).dt.days

        # Count trades per symbol
        trades_per_symbol = self.trades_df.groupby('symbol').size().mean()

        # Average position size (in dollars)
        self.trades_df['position_value'] = self.trades_df['entry_price'] * self.trades_df['position_size']
        avg_position_size = self.trades_df['position_value'].mean()

        return {
            'avg_holding_days': self.trades_df['holding_days'].mean(),
            'median_holding_days': self.trades_df['holding_days'].median(),
            'trades_per_symbol': trades_per_symbol,
            'avg_position_size': avg_position_size,
            'unique_symbols_traded': self.trades_df['symbol'].nunique()
        }

    def _calculate_win_loss_metrics(self) -> Dict:
        """Calculate win/loss statistics"""
        if self.trades_df.empty:
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'win_loss_ratio': 0.0,
                'consecutive_wins': 0,
                'consecutive_losses': 0
            }

        # Separate winners and losers
        winners = self.trades_df[self.trades_df['pnl'] > 0]
        losers = self.trades_df[self.trades_df['pnl'] < 0]

        # Win rate
        win_rate = (len(winners) / len(self.trades_df)) * 100

        # Profit factor
        total_wins = winners['pnl'].sum() if len(winners) > 0 else 0
        total_losses = abs(losers['pnl'].sum()) if len(losers) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Average win/loss
        avg_win = winners['pnl'].mean() if len(winners) > 0 else 0
        avg_loss = losers['pnl'].mean() if len(losers) > 0 else 0

        # Largest win/loss
        largest_win = winners['pnl'].max() if len(winners) > 0 else 0
        largest_loss = losers['pnl'].min() if len(losers) > 0 else 0

        # Win/loss ratio
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        # Consecutive wins/losses
        consecutive_wins, consecutive_losses = self._calculate_consecutive_streaks()

        # Exit reason analysis
        exit_reasons = self.trades_df['exit_reason'].value_counts().to_dict()

        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'win_loss_ratio': win_loss_ratio,
            'consecutive_wins': consecutive_wins,
            'consecutive_losses': consecutive_losses,
            'total_winning_trades': len(winners),
            'total_losing_trades': len(losers),
            'exit_reasons': exit_reasons
        }

    def _calculate_consecutive_streaks(self) -> Tuple[int, int]:
        """Calculate longest consecutive winning and losing streaks"""
        if self.trades_df.empty:
            return 0, 0

        # Sort by exit date
        df = self.trades_df.sort_values('exit_date')

        # Create win/loss indicator
        df['is_win'] = df['pnl'] > 0

        # Calculate streaks
        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0

        for is_win in df['is_win']:
            if is_win:
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            else:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)

        return max_win_streak, max_loss_streak

    def _calculate_drawdown_metrics(self) -> Dict:
        """Calculate drawdown statistics"""
        if len(self.portfolio_values) < 2:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'max_drawdown_duration_days': 0,
                'avg_drawdown': 0.0,
                'recovery_time_days': 0
            }

        # Convert to DataFrame
        df = pd.DataFrame(self.portfolio_values, columns=['date', 'value'])
        df['date'] = pd.to_datetime(df['date'])

        # Calculate running maximum
        df['running_max'] = df['value'].cummax()

        # Calculate drawdown
        df['drawdown'] = df['value'] - df['running_max']
        df['drawdown_pct'] = (df['drawdown'] / df['running_max']) * 100

        # Maximum drawdown
        max_drawdown = df['drawdown'].min()
        max_drawdown_pct = df['drawdown_pct'].min()

        # Find maximum drawdown duration
        df['is_drawdown'] = df['drawdown'] < 0
        drawdown_periods = []
        current_period = 0

        for is_dd in df['is_drawdown']:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0

        max_dd_duration = max(drawdown_periods) if drawdown_periods else 0

        # Average drawdown (when in drawdown)
        avg_drawdown = df[df['drawdown'] < 0]['drawdown_pct'].mean() if len(df[df['drawdown'] < 0]) > 0 else 0

        # Recovery time (from max drawdown to new high)
        max_dd_idx = df['drawdown'].idxmin()
        recovery_df = df[df.index > max_dd_idx]
        recovery_idx = recovery_df[recovery_df['drawdown'] == 0].first_valid_index()

        if recovery_idx is not None:
            recovery_time = recovery_idx - max_dd_idx
        else:
            recovery_time = len(df) - max_dd_idx  # Still in drawdown

        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'max_drawdown_duration_days': max_dd_duration,
            'avg_drawdown': avg_drawdown,
            'recovery_time_days': recovery_time
        }

    def _calculate_time_metrics(self) -> Dict:
        """Calculate time-based metrics"""
        if self.trades_df.empty or len(self.portfolio_values) < 2:
            return {
                'backtest_days': 0,
                'annualized_return': 0.0,
                'monthly_return': 0.0
            }

        # Get date range
        start_date = pd.to_datetime(self.portfolio_values[0][0])
        end_date = pd.to_datetime(self.portfolio_values[-1][0])
        days = (end_date - start_date).days

        # Calculate annualized return
        total_return_pct = ((self.portfolio_values[-1][1] / self.initial_capital) - 1)
        years = days / 365.25
        annualized_return = ((1 + total_return_pct) ** (1 / years) - 1) * 100 if years > 0 else 0

        # Monthly return (approximate)
        monthly_return = annualized_return / 12

        return {
            'backtest_days': days,
            'backtest_years': years,
            'annualized_return': annualized_return,
            'monthly_return': monthly_return,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d')
        }

    def print_summary(self) -> None:
        """Print formatted summary of all metrics"""
        metrics = self.calculate_all_metrics()

        print("\n" + "="*60)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("="*60)

        print("\n--- Basic Metrics ---")
        print(f"Total Trades:          {metrics['total_trades']}")
        print(f"Initial Capital:       ${metrics['initial_capital']:,.2f}")
        print(f"Final Capital:         ${metrics['final_capital']:,.2f}")
        print(f"Total Return:          ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:.2f}%)")

        print("\n--- Return Metrics ---")
        print(f"Avg Return/Trade:      ${metrics['avg_return_per_trade']:,.2f} ({metrics['avg_return_pct_per_trade']:.2f}%)")
        print(f"Best Trade:            ${metrics['best_trade']:,.2f}")
        print(f"Worst Trade:           ${metrics['worst_trade']:,.2f}")
        print(f"Median Return:         ${metrics['median_return']:,.2f}")

        print("\n--- Win/Loss Metrics ---")
        print(f"Win Rate:              {metrics['win_rate']:.2f}%")
        print(f"Profit Factor:         {metrics['profit_factor']:.2f}")
        print(f"Avg Win:               ${metrics['avg_win']:,.2f}")
        print(f"Avg Loss:              ${metrics['avg_loss']:,.2f}")
        print(f"Win/Loss Ratio:        {metrics['win_loss_ratio']:.2f}")
        print(f"Winning Trades:        {metrics['total_winning_trades']}")
        print(f"Losing Trades:         {metrics['total_losing_trades']}")

        print("\n--- Risk Metrics ---")
        print(f"Volatility (Annual):   {metrics['volatility']:.2f}%")
        print(f"Downside Volatility:   {metrics['downside_volatility']:.2f}%")
        print(f"Max Drawdown:          {metrics['max_drawdown_pct']:.2f}%")
        print(f"VaR (95%):             {metrics['value_at_risk_95']:.2f}%")

        print("\n--- Risk-Adjusted Metrics ---")
        print(f"Sharpe Ratio:          {metrics['sharpe_ratio']:.3f}")
        print(f"Sortino Ratio:         {metrics['sortino_ratio']:.3f}")
        print(f"Calmar Ratio:          {metrics['calmar_ratio']:.3f}")

        print("\n--- Trade Statistics ---")
        print(f"Avg Holding Days:      {metrics['avg_holding_days']:.1f}")
        print(f"Median Holding Days:   {metrics['median_holding_days']:.1f}")
        print(f"Unique Symbols:        {metrics['unique_symbols_traded']}")
        print(f"Avg Position Size:     ${metrics['avg_position_size']:,.2f}")

        print("\n--- Time Metrics ---")
        print(f"Backtest Period:       {metrics['backtest_days']} days ({metrics['backtest_years']:.2f} years)")
        print(f"Annualized Return:     {metrics['annualized_return']:.2f}%")
        print(f"Monthly Return (est):  {metrics['monthly_return']:.2f}%")

        print("\n" + "="*60)
