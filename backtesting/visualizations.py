"""
Backtesting Visualizations
==========================
Create comprehensive visualizations for backtest results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')


class BacktestVisualizer:
    """
    Create visualizations for backtesting results
    """

    def __init__(self, trades_df: pd.DataFrame,
                 portfolio_values: List[Tuple[str, float]],
                 initial_capital: float):
        """
        Initialize visualizer

        Args:
            trades_df: DataFrame with trade results
            portfolio_values: List of (date, value) tuples
            initial_capital: Starting capital
        """
        self.trades_df = trades_df
        self.portfolio_values = portfolio_values
        self.initial_capital = initial_capital

    def create_full_report(self, save_path: str = 'backtest_report.png') -> None:
        """
        Create comprehensive visualization report

        Args:
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        # 1. Portfolio Value Over Time
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_portfolio_value(ax1)

        # 2. Drawdown Chart
        ax2 = fig.add_subplot(gs[1, :])
        self._plot_drawdown(ax2)

        # 3. Returns Distribution
        ax3 = fig.add_subplot(gs[2, 0])
        self._plot_returns_distribution(ax3)

        # 4. Win Rate by Strategy
        ax4 = fig.add_subplot(gs[2, 1])
        self._plot_strategy_performance(ax4)

        # 5. Exit Reasons
        ax5 = fig.add_subplot(gs[2, 2])
        self._plot_exit_reasons(ax5)

        # 6. Cumulative Returns
        ax6 = fig.add_subplot(gs[3, 0])
        self._plot_cumulative_returns(ax6)

        # 7. Monthly Returns Heatmap
        ax7 = fig.add_subplot(gs[3, 1])
        self._plot_monthly_returns(ax7)

        # 8. Trade Duration Analysis
        ax8 = fig.add_subplot(gs[3, 2])
        self._plot_trade_duration(ax8)

        plt.suptitle('Backtest Performance Report', fontsize=20, fontweight='bold')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comprehensive report to {save_path}")
        plt.close()

    def _plot_portfolio_value(self, ax) -> None:
        """Plot portfolio value over time"""
        if not self.portfolio_values:
            return

        dates, values = zip(*self.portfolio_values)
        dates = pd.to_datetime(dates)

        ax.plot(dates, values, linewidth=2, color='#2E86AB', label='Portfolio Value')
        ax.axhline(y=self.initial_capital, color='gray', linestyle='--',
                   linewidth=1, alpha=0.7, label='Initial Capital')

        # Fill area
        ax.fill_between(dates, self.initial_capital, values,
                        where=np.array(values) >= self.initial_capital,
                        alpha=0.3, color='green', label='Profit')
        ax.fill_between(dates, self.initial_capital, values,
                        where=np.array(values) < self.initial_capital,
                        alpha=0.3, color='red', label='Loss')

        ax.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    def _plot_drawdown(self, ax) -> None:
        """Plot drawdown over time"""
        if not self.portfolio_values:
            return

        dates, values = zip(*self.portfolio_values)
        dates = pd.to_datetime(dates)
        values = np.array(values)

        # Calculate drawdown
        running_max = np.maximum.accumulate(values)
        drawdown = (values - running_max) / running_max * 100

        ax.fill_between(dates, 0, drawdown, color='red', alpha=0.5)
        ax.plot(dates, drawdown, color='darkred', linewidth=1.5)

        ax.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add max drawdown annotation
        max_dd_idx = np.argmin(drawdown)
        ax.annotate(f'Max DD: {drawdown[max_dd_idx]:.2f}%',
                   xy=(dates[max_dd_idx], drawdown[max_dd_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   arrowprops=dict(arrowstyle='->', color='black'))

    def _plot_returns_distribution(self, ax) -> None:
        """Plot distribution of trade returns"""
        if self.trades_df.empty:
            return

        returns = self.trades_df['pnl_pct'] * 100

        # Histogram
        ax.hist(returns, bins=50, alpha=0.7, color='#A23B72', edgecolor='black')

        # Add vertical lines for mean and median
        mean_return = returns.mean()
        median_return = returns.median()

        ax.axvline(mean_return, color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {mean_return:.2f}%')
        ax.axvline(median_return, color='green', linestyle='--',
                  linewidth=2, label=f'Median: {median_return:.2f}%')
        ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

        ax.set_title('Distribution of Trade Returns', fontsize=12, fontweight='bold')
        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Frequency')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    def _plot_strategy_performance(self, ax) -> None:
        """Plot win rate by strategy"""
        if self.trades_df.empty or 'strategy' not in self.trades_df.columns:
            return

        # Calculate win rate by strategy
        strategy_stats = self.trades_df.groupby('strategy').apply(
            lambda x: pd.Series({
                'win_rate': (x['pnl'] > 0).sum() / len(x) * 100,
                'total_trades': len(x)
            })
        )

        # Sort by win rate
        strategy_stats = strategy_stats.sort_values('win_rate', ascending=True)

        # Create horizontal bar chart
        colors = ['#EF476F' if wr < 50 else '#06D6A0' for wr in strategy_stats['win_rate']]
        bars = ax.barh(strategy_stats.index, strategy_stats['win_rate'], color=colors, alpha=0.7)

        # Add trade count labels
        for i, (idx, row) in enumerate(strategy_stats.iterrows()):
            ax.text(row['win_rate'] + 1, i, f"n={int(row['total_trades'])}",
                   va='center', fontsize=9)

        ax.axvline(50, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_title('Win Rate by Strategy', fontsize=12, fontweight='bold')
        ax.set_xlabel('Win Rate (%)')
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3, axis='x')

    def _plot_exit_reasons(self, ax) -> None:
        """Plot pie chart of exit reasons"""
        if self.trades_df.empty or 'exit_reason' not in self.trades_df.columns:
            return

        exit_counts = self.trades_df['exit_reason'].value_counts()

        colors = ['#06D6A0', '#EF476F', '#FFD166', '#118AB2', '#073B4C']
        explode = [0.05] * len(exit_counts)

        ax.pie(exit_counts.values, labels=exit_counts.index, autopct='%1.1f%%',
              colors=colors, explode=explode, shadow=True, startangle=90)

        ax.set_title('Exit Reasons Distribution', fontsize=12, fontweight='bold')

    def _plot_cumulative_returns(self, ax) -> None:
        """Plot cumulative returns"""
        if self.trades_df.empty:
            return

        # Sort by exit date
        df = self.trades_df.sort_values('exit_date')
        df['cumulative_return'] = df['pnl'].cumsum()

        dates = pd.to_datetime(df['exit_date'])

        ax.plot(dates, df['cumulative_return'], linewidth=2, color='#118AB2')
        ax.fill_between(dates, 0, df['cumulative_return'],
                       where=df['cumulative_return'] >= 0,
                       alpha=0.3, color='green')
        ax.fill_between(dates, 0, df['cumulative_return'],
                       where=df['cumulative_return'] < 0,
                       alpha=0.3, color='red')

        ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative P&L ($)')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    def _plot_monthly_returns(self, ax) -> None:
        """Plot monthly returns heatmap"""
        if self.trades_df.empty:
            return

        # Group by month
        df = self.trades_df.copy()
        df['exit_date'] = pd.to_datetime(df['exit_date'])
        df['year_month'] = df['exit_date'].dt.to_period('M')

        monthly_returns = df.groupby('year_month')['pnl'].sum()

        if len(monthly_returns) == 0:
            return

        # Simple bar chart (heatmap requires more data restructuring)
        months = [str(m) for m in monthly_returns.index]
        values = monthly_returns.values

        colors = ['green' if v > 0 else 'red' for v in values]
        bars = ax.bar(range(len(months)), values, color=colors, alpha=0.7)

        ax.set_title('Monthly Returns', fontsize=12, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Returns ($)')
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45, ha='right', fontsize=8)
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.grid(True, alpha=0.3, axis='y')

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    def _plot_trade_duration(self, ax) -> None:
        """Plot trade duration analysis"""
        if self.trades_df.empty:
            return

        # Calculate holding days
        df = self.trades_df.copy()
        df['holding_days'] = (
            pd.to_datetime(df['exit_date']) -
            pd.to_datetime(df['entry_date'])
        ).dt.days

        # Create bins
        df['duration_bin'] = pd.cut(
            df['holding_days'],
            bins=[0, 3, 7, 14, 21, float('inf')],
            labels=['0-3d', '3-7d', '7-14d', '14-21d', '21d+']
        )

        # Count by bin
        duration_counts = df['duration_bin'].value_counts().sort_index()

        ax.bar(range(len(duration_counts)), duration_counts.values,
              color='#FFD166', alpha=0.7, edgecolor='black')

        ax.set_title('Trade Duration Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Holding Period')
        ax.set_ylabel('Number of Trades')
        ax.set_xticks(range(len(duration_counts)))
        ax.set_xticklabels(duration_counts.index, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

    def plot_individual_charts(self, output_dir: str = './') -> None:
        """
        Create individual chart files

        Args:
            output_dir: Directory to save charts
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Portfolio value
        fig, ax = plt.subplots(figsize=(12, 6))
        self._plot_portfolio_value(ax)
        plt.savefig(f'{output_dir}/portfolio_value.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Drawdown
        fig, ax = plt.subplots(figsize=(12, 6))
        self._plot_drawdown(ax)
        plt.savefig(f'{output_dir}/drawdown.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Returns distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        self._plot_returns_distribution(ax)
        plt.savefig(f'{output_dir}/returns_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Strategy performance
        fig, ax = plt.subplots(figsize=(10, 6))
        self._plot_strategy_performance(ax)
        plt.savefig(f'{output_dir}/strategy_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Cumulative returns
        fig, ax = plt.subplots(figsize=(12, 6))
        self._plot_cumulative_returns(ax)
        plt.savefig(f'{output_dir}/cumulative_returns.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved individual charts to {output_dir}")
