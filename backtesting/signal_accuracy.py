"""
Signal Accuracy Analyzer
========================
Analyze the accuracy and reliability of trading signals
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class SignalAccuracyAnalyzer:
    """
    Analyze trading signal accuracy and prediction quality
    """

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db"):
        """
        Initialize signal accuracy analyzer

        Args:
            db_path: Path to database
        """
        self.db_path = db_path

    def _conn(self) -> sqlite3.Connection:
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def analyze_signal_accuracy(self, trades_df: pd.DataFrame) -> Dict:
        """
        Analyze signal accuracy across multiple dimensions

        Args:
            trades_df: DataFrame with trade results from backtest

        Returns:
            Dictionary with accuracy metrics
        """
        if trades_df.empty:
            logger.warning("No trades to analyze")
            return {}

        metrics = {}

        # Overall accuracy
        metrics.update(self._calculate_overall_accuracy(trades_df))

        # Signal strength analysis
        metrics.update(self._analyze_signal_strength(trades_df))

        # Strategy-specific accuracy
        metrics.update(self._analyze_strategy_accuracy(trades_df))

        # Signal type accuracy (BUY vs SELL)
        metrics.update(self._analyze_signal_type_accuracy(trades_df))

        # Exit reason analysis
        metrics.update(self._analyze_exit_reasons(trades_df))

        # Prediction horizon analysis
        metrics.update(self._analyze_prediction_horizon(trades_df))

        # Symbol-level accuracy
        metrics.update(self._analyze_symbol_accuracy(trades_df))

        return metrics

    def _calculate_overall_accuracy(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate overall signal accuracy"""
        profitable_trades = len(trades_df[trades_df['pnl'] > 0])
        total_trades = len(trades_df)

        accuracy = (profitable_trades / total_trades * 100) if total_trades > 0 else 0

        return {
            'overall_accuracy': accuracy,
            'total_signals_tested': total_trades,
            'profitable_signals': profitable_trades,
            'unprofitable_signals': total_trades - profitable_trades
        }

    def _analyze_signal_strength(self, trades_df: pd.DataFrame) -> Dict:
        """
        Analyze if signal strength correlates with profitability
        """
        if 'strength' not in trades_df.columns:
            return {}

        # Create strength bins
        trades_df['strength_bin'] = pd.cut(
            trades_df['strength'],
            bins=[0, 0.5, 0.7, 0.85, 1.0],
            labels=['weak', 'moderate', 'strong', 'very_strong']
        )

        # Calculate accuracy by strength
        strength_analysis = trades_df.groupby('strength_bin').agg({
            'pnl': ['count', 'mean', 'sum'],
            'pnl_pct': 'mean'
        }).round(4)

        # Calculate win rate by strength
        strength_win_rate = trades_df.groupby('strength_bin').apply(
            lambda x: (x['pnl'] > 0).sum() / len(x) * 100
        ).to_dict()

        # Correlation between strength and profitability
        correlation = trades_df['strength'].corr(trades_df['pnl'])

        return {
            'strength_analysis': strength_analysis.to_dict(),
            'strength_win_rate': strength_win_rate,
            'strength_pnl_correlation': correlation
        }

    def _analyze_strategy_accuracy(self, trades_df: pd.DataFrame) -> Dict:
        """Analyze accuracy by strategy"""
        if 'strategy' not in trades_df.columns:
            return {}

        strategy_metrics = {}

        for strategy in trades_df['strategy'].unique():
            strategy_trades = trades_df[trades_df['strategy'] == strategy]

            profitable = len(strategy_trades[strategy_trades['pnl'] > 0])
            total = len(strategy_trades)
            accuracy = (profitable / total * 100) if total > 0 else 0

            avg_pnl = strategy_trades['pnl'].mean()
            total_pnl = strategy_trades['pnl'].sum()
            avg_pnl_pct = strategy_trades['pnl_pct'].mean() * 100

            strategy_metrics[strategy] = {
                'accuracy': accuracy,
                'total_trades': total,
                'profitable_trades': profitable,
                'avg_pnl': avg_pnl,
                'total_pnl': total_pnl,
                'avg_pnl_pct': avg_pnl_pct,
                'best_trade': strategy_trades['pnl'].max(),
                'worst_trade': strategy_trades['pnl'].min()
            }

        # Rank strategies by accuracy
        strategy_ranking = sorted(
            strategy_metrics.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )

        return {
            'strategy_metrics': strategy_metrics,
            'strategy_ranking': [s[0] for s in strategy_ranking]
        }

    def _analyze_signal_type_accuracy(self, trades_df: pd.DataFrame) -> Dict:
        """Analyze accuracy by signal type (BUY vs SELL)"""
        signal_type_metrics = {}

        for signal_type in trades_df['signal_type'].unique():
            type_trades = trades_df[trades_df['signal_type'] == signal_type]

            profitable = len(type_trades[type_trades['pnl'] > 0])
            total = len(type_trades)
            accuracy = (profitable / total * 100) if total > 0 else 0

            signal_type_metrics[signal_type] = {
                'accuracy': accuracy,
                'total_signals': total,
                'profitable_signals': profitable,
                'avg_pnl': type_trades['pnl'].mean(),
                'avg_pnl_pct': type_trades['pnl_pct'].mean() * 100
            }

        return {'signal_type_metrics': signal_type_metrics}

    def _analyze_exit_reasons(self, trades_df: pd.DataFrame) -> Dict:
        """Analyze outcomes by exit reason"""
        if 'exit_reason' not in trades_df.columns:
            return {}

        exit_metrics = {}

        for reason in trades_df['exit_reason'].unique():
            reason_trades = trades_df[trades_df['exit_reason'] == reason]

            profitable = len(reason_trades[reason_trades['pnl'] > 0])
            total = len(reason_trades)

            exit_metrics[reason] = {
                'count': total,
                'percentage': (total / len(trades_df) * 100),
                'win_rate': (profitable / total * 100) if total > 0 else 0,
                'avg_pnl': reason_trades['pnl'].mean(),
                'avg_pnl_pct': reason_trades['pnl_pct'].mean() * 100
            }

        return {'exit_reason_metrics': exit_metrics}

    def _analyze_prediction_horizon(self, trades_df: pd.DataFrame) -> Dict:
        """Analyze accuracy by holding period"""
        if 'holding_days' not in trades_df.columns:
            # Calculate holding days
            trades_df['holding_days'] = (
                pd.to_datetime(trades_df['exit_date']) -
                pd.to_datetime(trades_df['entry_date'])
            ).dt.days

        # Create holding period bins
        trades_df['holding_period'] = pd.cut(
            trades_df['holding_days'],
            bins=[0, 3, 7, 14, 30, float('inf')],
            labels=['0-3d', '3-7d', '7-14d', '14-30d', '30d+']
        )

        horizon_metrics = {}

        for period in trades_df['holding_period'].unique():
            if pd.isna(period):
                continue

            period_trades = trades_df[trades_df['holding_period'] == period]

            profitable = len(period_trades[period_trades['pnl'] > 0])
            total = len(period_trades)

            horizon_metrics[str(period)] = {
                'count': total,
                'accuracy': (profitable / total * 100) if total > 0 else 0,
                'avg_pnl': period_trades['pnl'].mean(),
                'avg_pnl_pct': period_trades['pnl_pct'].mean() * 100,
                'avg_holding_days': period_trades['holding_days'].mean()
            }

        return {'prediction_horizon_metrics': horizon_metrics}

    def _analyze_symbol_accuracy(self, trades_df: pd.DataFrame) -> Dict:
        """Analyze accuracy by symbol"""
        symbol_metrics = trades_df.groupby('symbol').agg({
            'pnl': ['count', lambda x: (x > 0).sum(), 'mean', 'sum'],
            'pnl_pct': 'mean'
        }).round(4)

        symbol_metrics.columns = ['total_trades', 'profitable_trades', 'avg_pnl', 'total_pnl', 'avg_pnl_pct']
        symbol_metrics['accuracy'] = (symbol_metrics['profitable_trades'] / symbol_metrics['total_trades'] * 100).round(2)

        # Sort by accuracy
        symbol_metrics = symbol_metrics.sort_values('accuracy', ascending=False)

        # Get top and bottom performers
        top_symbols = symbol_metrics.head(10).to_dict('index')
        bottom_symbols = symbol_metrics.tail(10).to_dict('index')

        return {
            'symbol_metrics': symbol_metrics.to_dict('index'),
            'top_performing_symbols': top_symbols,
            'bottom_performing_symbols': bottom_symbols
        }

    def analyze_signal_reliability(self, strategy_name: Optional[str] = None,
                                   lookback_days: int = 30) -> Dict:
        """
        Analyze signal reliability by comparing signals to actual price movements

        Args:
            strategy_name: Strategy to analyze (None = all)
            lookback_days: Days to look back

        Returns:
            Dictionary with reliability metrics
        """
        conn = self._conn()

        # Get signals
        query = """
            SELECT
                ts.strategy_name,
                ts.symbol_ticker,
                ts.signal_date,
                ts.signal_type,
                ts.strength,
                ts.entry_price,
                -- Get actual price changes
                rpd1.close as price_1d,
                rpd5.close as price_5d,
                rpd10.close as price_10d
            FROM trading_signals ts
            LEFT JOIN raw_price_data rpd1
                ON ts.symbol_ticker = rpd1.symbol_ticker
                AND rpd1.price_date = date(ts.signal_date, '+1 day')
            LEFT JOIN raw_price_data rpd5
                ON ts.symbol_ticker = rpd5.symbol_ticker
                AND rpd5.price_date = date(ts.signal_date, '+5 days')
            LEFT JOIN raw_price_data rpd10
                ON ts.symbol_ticker = rpd10.symbol_ticker
                AND rpd10.price_date = date(ts.signal_date, '+10 days')
            WHERE ts.signal_date >= date('now', '-{} days')
        """.format(lookback_days)

        if strategy_name:
            query += f" AND ts.strategy_name = '{strategy_name}'"

        df = pd.read_sql(query, conn)
        conn.close()

        if df.empty:
            return {}

        # Calculate actual returns
        df['return_1d'] = (df['price_1d'] - df['entry_price']) / df['entry_price']
        df['return_5d'] = (df['price_5d'] - df['entry_price']) / df['entry_price']
        df['return_10d'] = (df['price_10d'] - df['entry_price']) / df['entry_price']

        # Check if signal direction was correct
        for horizon in ['1d', '5d', '10d']:
            df[f'correct_{horizon}'] = (
                ((df['signal_type'] == 'BUY') & (df[f'return_{horizon}'] > 0)) |
                ((df['signal_type'] == 'SELL') & (df[f'return_{horizon}'] < 0))
            )

        # Calculate directional accuracy
        reliability_metrics = {
            'directional_accuracy_1d': (df['correct_1d'].sum() / len(df) * 100) if len(df) > 0 else 0,
            'directional_accuracy_5d': (df['correct_5d'].sum() / len(df) * 100) if len(df) > 0 else 0,
            'directional_accuracy_10d': (df['correct_10d'].sum() / len(df) * 100) if len(df) > 0 else 0,
            'avg_return_1d': df['return_1d'].mean() * 100,
            'avg_return_5d': df['return_5d'].mean() * 100,
            'avg_return_10d': df['return_10d'].mean() * 100,
            'total_signals_analyzed': len(df)
        }

        return reliability_metrics

    def print_accuracy_report(self, trades_df: pd.DataFrame) -> None:
        """Print formatted accuracy report"""
        metrics = self.analyze_signal_accuracy(trades_df)

        print("\n" + "="*60)
        print("SIGNAL ACCURACY ANALYSIS")
        print("="*60)

        print("\n--- Overall Accuracy ---")
        print(f"Overall Accuracy:      {metrics['overall_accuracy']:.2f}%")
        print(f"Total Signals:         {metrics['total_signals_tested']}")
        print(f"Profitable Signals:    {metrics['profitable_signals']}")
        print(f"Unprofitable Signals:  {metrics['unprofitable_signals']}")

        if 'strength_pnl_correlation' in metrics:
            print("\n--- Signal Strength Analysis ---")
            print(f"Strength-PnL Correlation: {metrics['strength_pnl_correlation']:.3f}")
            print("\nWin Rate by Strength:")
            for strength, win_rate in metrics['strength_win_rate'].items():
                print(f"  {strength:12s}: {win_rate:.2f}%")

        if 'strategy_metrics' in metrics:
            print("\n--- Strategy Performance Ranking ---")
            for i, strategy_name in enumerate(metrics['strategy_ranking'], 1):
                strategy_data = metrics['strategy_metrics'][strategy_name]
                print(f"{i}. {strategy_name}")
                print(f"   Accuracy: {strategy_data['accuracy']:.2f}% | "
                      f"Trades: {strategy_data['total_trades']} | "
                      f"Avg PnL: ${strategy_data['avg_pnl']:.2f}")

        if 'signal_type_metrics' in metrics:
            print("\n--- Signal Type Accuracy ---")
            for signal_type, data in metrics['signal_type_metrics'].items():
                print(f"{signal_type}:")
                print(f"  Accuracy: {data['accuracy']:.2f}%")
                print(f"  Signals:  {data['total_signals']}")
                print(f"  Avg PnL:  ${data['avg_pnl']:.2f} ({data['avg_pnl_pct']:.2f}%)")

        if 'exit_reason_metrics' in metrics:
            print("\n--- Exit Reason Analysis ---")
            for reason, data in metrics['exit_reason_metrics'].items():
                print(f"{reason}:")
                print(f"  Count:    {data['count']} ({data['percentage']:.1f}%)")
                print(f"  Win Rate: {data['win_rate']:.2f}%")
                print(f"  Avg PnL:  ${data['avg_pnl']:.2f}")

        if 'prediction_horizon_metrics' in metrics:
            print("\n--- Prediction Horizon Analysis ---")
            for period, data in metrics['prediction_horizon_metrics'].items():
                print(f"{period}:")
                print(f"  Count:    {data['count']}")
                print(f"  Accuracy: {data['accuracy']:.2f}%")
                print(f"  Avg PnL:  ${data['avg_pnl']:.2f} ({data['avg_pnl_pct']:.2f}%)")

        if 'top_performing_symbols' in metrics:
            print("\n--- Top 5 Performing Symbols ---")
            for i, (symbol, data) in enumerate(list(metrics['top_performing_symbols'].items())[:5], 1):
                print(f"{i}. {symbol}: {data['accuracy']:.1f}% accuracy "
                      f"({data['total_trades']} trades, ${data['total_pnl']:.2f} total)")

        print("\n" + "="*60)
