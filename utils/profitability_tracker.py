"""
Profitability Tracker for RiskLabAI Trading Bot

This module tracks key profitability metrics over time:
- Win rate (percentage of profitable trades)
- Sharpe ratio (risk-adjusted returns)
- Maximum drawdown (worst peak-to-trough decline)
- Total return
- Trade statistics

Logs are saved to logs/profitability_logs/ for analysis.
"""

import logging
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ProfitabilityTracker:
    """
    Tracks trading performance metrics for profitability validation.

    This class records every trade and calculates key metrics:
    - Win rate: % of profitable trades
    - Sharpe ratio: (mean return - risk_free) / std(returns)
    - Max drawdown: largest peak-to-trough decline
    - Total return: cumulative P&L

    Attributes:
        log_dir (Path): Directory for profitability logs
        trades (List[Dict]): Record of all trades
        daily_returns (List[float]): Daily return percentages
        portfolio_values (List[float]): Portfolio value over time
    """

    def __init__(self, log_dir: str = "logs/profitability_logs"):
        """
        Initialize profitability tracker.

        Args:
            log_dir: Directory to save profitability logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Trade tracking
        self.trades: List[Dict] = []
        self.daily_returns: List[float] = []
        self.portfolio_values: List[float] = []
        self.timestamps: List[datetime] = []

        # Performance metrics
        self.initial_capital = None
        self.current_capital = None

        # File paths
        self.trades_file = self.log_dir / "trades.csv"
        self.metrics_file = self.log_dir / "daily_metrics.csv"
        self.summary_file = self.log_dir / "performance_summary.json"

        # Initialize CSV files
        self._init_csv_files()

        logger.info(f"ProfitabilityTracker initialized: {log_dir}")

    def _init_csv_files(self):
        """Initialize CSV files with headers if they don't exist."""
        # Trades CSV
        if not self.trades_file.exists():
            with open(self.trades_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'side', 'quantity', 'entry_price',
                    'exit_price', 'pnl', 'pnl_pct', 'holding_period', 'win'
                ])

        # Daily metrics CSV
        if not self.metrics_file.exists():
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'date', 'portfolio_value', 'daily_return', 'cumulative_return',
                    'win_rate', 'sharpe_ratio', 'max_drawdown', 'num_trades'
                ])

    def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        entry_price: float,
        exit_price: float,
        entry_time: datetime,
        exit_time: datetime
    ):
        """
        Record a completed trade.

        Args:
            symbol: Stock ticker
            side: 'long' or 'short'
            quantity: Number of shares
            entry_price: Entry price per share
            exit_price: Exit price per share
            entry_time: Trade entry timestamp
            exit_time: Trade exit timestamp
        """
        # Calculate P&L
        if side.lower() == 'long':
            pnl = (exit_price - entry_price) * quantity
        else:  # short
            pnl = (entry_price - exit_price) * quantity

        pnl_pct = (pnl / (entry_price * quantity)) * 100
        holding_period = (exit_time - entry_time).total_seconds() / 3600  # hours
        is_win = pnl > 0

        trade = {
            'timestamp': exit_time.isoformat(),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'holding_period': holding_period,
            'win': is_win
        }

        self.trades.append(trade)

        # Write to CSV
        with open(self.trades_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                trade['timestamp'], trade['symbol'], trade['side'],
                trade['quantity'], trade['entry_price'], trade['exit_price'],
                f"{trade['pnl']:.2f}", f"{trade['pnl_pct']:.2f}",
                f"{trade['holding_period']:.2f}", trade['win']
            ])

        logger.info(f"Trade recorded: {symbol} {side} {quantity}@${exit_price:.2f} "
                   f"P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")

    def record_daily_snapshot(
        self,
        portfolio_value: float,
        timestamp: Optional[datetime] = None
    ):
        """
        Record daily portfolio snapshot.

        Args:
            portfolio_value: Current total portfolio value
            timestamp: Snapshot timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Initialize capital tracking
        if self.initial_capital is None:
            self.initial_capital = portfolio_value
            self.current_capital = portfolio_value

        # Calculate daily return
        if len(self.portfolio_values) > 0:
            prev_value = self.portfolio_values[-1]
            daily_return = ((portfolio_value - prev_value) / prev_value) * 100
        else:
            daily_return = 0.0

        self.portfolio_values.append(portfolio_value)
        self.daily_returns.append(daily_return)
        self.timestamps.append(timestamp)
        self.current_capital = portfolio_value

        # Calculate metrics
        metrics = self.calculate_metrics()

        # Write to CSV
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp.date().isoformat(),
                f"{portfolio_value:.2f}",
                f"{daily_return:.4f}",
                f"{metrics['cumulative_return']:.2f}",
                f"{metrics['win_rate']:.2f}",
                f"{metrics['sharpe_ratio']:.2f}",
                f"{metrics['max_drawdown']:.2f}",
                len(self.trades)
            ])

        logger.debug(f"Daily snapshot: ${portfolio_value:,.2f} "
                    f"(return: {daily_return:+.2f}%)")

    def calculate_metrics(self) -> Dict:
        """
        Calculate all profitability metrics.

        Returns:
            Dict with:
            - win_rate: Percentage of winning trades
            - sharpe_ratio: Risk-adjusted return
            - max_drawdown: Maximum peak-to-trough decline
            - total_return: Total P&L
            - cumulative_return: % return on initial capital
            - num_trades: Total number of trades
            - avg_win: Average winning trade %
            - avg_loss: Average losing trade %
        """
        metrics = {
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_return': 0.0,
            'cumulative_return': 0.0,
            'num_trades': len(self.trades),
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'num_wins': 0,
            'num_losses': 0
        }

        # Win rate
        if self.trades:
            wins = [t for t in self.trades if t['win']]
            losses = [t for t in self.trades if not t['win']]

            metrics['num_wins'] = len(wins)
            metrics['num_losses'] = len(losses)
            metrics['win_rate'] = (len(wins) / len(self.trades)) * 100

            if wins:
                metrics['avg_win'] = np.mean([t['pnl_pct'] for t in wins])
            if losses:
                metrics['avg_loss'] = np.mean([t['pnl_pct'] for t in losses])

        # Sharpe ratio (annualized, assuming 252 trading days)
        if len(self.daily_returns) > 1:
            returns_array = np.array(self.daily_returns)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)

            if std_return > 0:
                # Annualized Sharpe (assuming daily returns)
                metrics['sharpe_ratio'] = (mean_return / std_return) * np.sqrt(252)

        # Maximum drawdown
        if len(self.portfolio_values) > 1:
            portfolio_array = np.array(self.portfolio_values)
            peak = np.maximum.accumulate(portfolio_array)
            drawdown = (portfolio_array - peak) / peak * 100
            metrics['max_drawdown'] = np.min(drawdown)

        # Total return
        if self.initial_capital and self.current_capital:
            metrics['total_return'] = self.current_capital - self.initial_capital
            metrics['cumulative_return'] = (
                (self.current_capital - self.initial_capital) /
                self.initial_capital * 100
            )

        return metrics

    def save_summary(self):
        """Save performance summary to JSON file."""
        metrics = self.calculate_metrics()

        summary = {
            'generated_at': datetime.now().isoformat(),
            'period': {
                'start': self.timestamps[0].isoformat() if self.timestamps else None,
                'end': self.timestamps[-1].isoformat() if self.timestamps else None,
                'days': len(self.portfolio_values)
            },
            'capital': {
                'initial': self.initial_capital,
                'current': self.current_capital,
                'total_return': metrics['total_return'],
                'cumulative_return_pct': metrics['cumulative_return']
            },
            'metrics': {
                'win_rate': metrics['win_rate'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'num_trades': metrics['num_trades'],
                'num_wins': metrics['num_wins'],
                'num_losses': metrics['num_losses'],
                'avg_win_pct': metrics['avg_win'],
                'avg_loss_pct': metrics['avg_loss']
            },
            'profitability_criteria': {
                'meets_win_rate': metrics['win_rate'] >= 52.0,
                'meets_sharpe': metrics['sharpe_ratio'] >= 1.0,
                'meets_drawdown': metrics['max_drawdown'] >= -10.0,
                'meets_return': metrics['cumulative_return'] > 0,
                'ready_for_live': self._check_live_readiness(metrics)
            }
        }

        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Performance summary saved: {self.summary_file}")

        return summary

    def _check_live_readiness(self, metrics: Dict) -> bool:
        """
        Check if bot meets criteria for live trading.

        Criteria (conservative for $1000 capital):
        - Win rate >= 52%
        - Sharpe ratio >= 1.0
        - Max drawdown >= -10%
        - Cumulative return > 0%
        - At least 30 days of data
        - At least 20 trades

        Returns:
            bool: True if ready for live trading
        """
        criteria = [
            metrics['win_rate'] >= 52.0,
            metrics['sharpe_ratio'] >= 1.0,
            metrics['max_drawdown'] >= -10.0,
            metrics['cumulative_return'] > 0,
            len(self.portfolio_values) >= 30,
            metrics['num_trades'] >= 20
        ]

        return all(criteria)

    def print_summary(self):
        """Print formatted performance summary."""
        metrics = self.calculate_metrics()

        print("=" * 80)
        print("PROFITABILITY SUMMARY")
        print("=" * 80)
        print(f"\nCapital:")
        print(f"  Initial: ${self.initial_capital:,.2f}")
        print(f"  Current: ${self.current_capital:,.2f}")
        print(f"  Return:  ${metrics['total_return']:+,.2f} ({metrics['cumulative_return']:+.2f}%)")

        print(f"\nTrade Statistics:")
        print(f"  Total trades: {metrics['num_trades']}")
        print(f"  Wins: {metrics['num_wins']} ({metrics['win_rate']:.1f}%)")
        print(f"  Losses: {metrics['num_losses']}")
        print(f"  Avg win: {metrics['avg_win']:+.2f}%")
        print(f"  Avg loss: {metrics['avg_loss']:+.2f}%")

        print(f"\nRisk Metrics:")
        print(f"  Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max drawdown: {metrics['max_drawdown']:.2f}%")

        print(f"\nLive Trading Readiness:")
        ready = self._check_live_readiness(metrics)
        status = "✓ READY" if ready else "✗ NOT READY"
        print(f"  {status}")

        if not ready:
            print(f"\n  Requirements:")
            print(f"    Win rate >= 52%: {'✓' if metrics['win_rate'] >= 52 else '✗'} ({metrics['win_rate']:.1f}%)")
            print(f"    Sharpe >= 1.0: {'✓' if metrics['sharpe_ratio'] >= 1.0 else '✗'} ({metrics['sharpe_ratio']:.2f})")
            print(f"    Drawdown >= -10%: {'✓' if metrics['max_drawdown'] >= -10 else '✗'} ({metrics['max_drawdown']:.2f}%)")
            print(f"    Positive return: {'✓' if metrics['cumulative_return'] > 0 else '✗'} ({metrics['cumulative_return']:+.2f}%)")
            print(f"    >= 30 days: {'✓' if len(self.portfolio_values) >= 30 else '✗'} ({len(self.portfolio_values)} days)")
            print(f"    >= 20 trades: {'✓' if metrics['num_trades'] >= 20 else '✗'} ({metrics['num_trades']} trades)")

        print("=" * 80)
