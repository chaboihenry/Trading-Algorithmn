"""
Strategy Performance Tracker
=============================
Tracks recent performance of each strategy to help with:
1. Monitoring which strategies are hot/cold
2. Debugging performance issues
3. Informing ensemble weights (if needed)

This provides visibility into strategy performance without modifying the ensemble.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


class StrategyPerformanceTracker:
    """Track and analyze recent strategy performance"""

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db"):
        self.db_path = db_path

    def get_strategy_performance(self, lookback_days: int = 30) -> pd.DataFrame:
        """
        Calculate performance metrics for each strategy over recent period

        Args:
            lookback_days: Number of days to look back

        Returns:
            DataFrame with performance metrics per strategy
        """
        conn = sqlite3.connect(self.db_path)

        # Get signals and their outcomes
        query = f"""
            WITH signal_exits AS (
                SELECT
                    ts.strategy_name,
                    ts.symbol_ticker,
                    ts.signal_date,
                    ts.signal_type,
                    ts.entry_price,
                    ts.strength,
                    -- Get price 5 days later for return calculation
                    (
                        SELECT rpd2.close
                        FROM raw_price_data rpd2
                        WHERE rpd2.symbol_ticker = ts.symbol_ticker
                        AND rpd2.price_date >= date(ts.signal_date, '+5 days')
                        ORDER BY rpd2.price_date ASC
                        LIMIT 1
                    ) as exit_price
                FROM trading_signals ts
                WHERE ts.signal_date >= date('now', '-{lookback_days} days')
            ),
            signal_returns AS (
                SELECT
                    strategy_name,
                    signal_date,
                    signal_type,
                    entry_price,
                    strength,
                    exit_price,
                    CASE
                        WHEN signal_type = 'BUY' THEN
                            (exit_price - entry_price) / entry_price
                        WHEN signal_type = 'SELL' THEN
                            (entry_price - exit_price) / entry_price
                        ELSE 0
                    END as return_pct
                FROM signal_exits
                WHERE exit_price IS NOT NULL
            )
            SELECT
                strategy_name,
                COUNT(*) as num_signals,
                AVG(return_pct) * 100 as avg_return_pct,
                STDEV(return_pct) * 100 as volatility_pct,
                SUM(CASE WHEN return_pct > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as win_rate,
                AVG(return_pct) / NULLIF(STDEV(return_pct), 0) * SQRT(252) as sharpe_ratio
            FROM signal_returns
            WHERE return_pct IS NOT NULL
            GROUP BY strategy_name
            ORDER BY sharpe_ratio DESC
        """

        df = pd.read_sql(query, conn)
        conn.close()

        return df

    def get_strategy_weights_from_performance(self, lookback_days: int = 30) -> dict:
        """
        Calculate dynamic weights based on recent Sharpe ratios

        Args:
            lookback_days: Number of days to look back

        Returns:
            Dict of {strategy_name: weight}
        """
        perf = self.get_strategy_performance(lookback_days)

        if perf.empty:
            # Default equal weights if no performance data
            return {
                'SentimentTradingStrategy': 0.33,
                'PairsTradingStrategy': 0.33,
                'VolatilityTradingStrategy': 0.34
            }

        # Use positive Sharpe ratios only (set negative to small positive)
        perf['positive_sharpe'] = perf['sharpe_ratio'].apply(lambda x: max(x, 0.01))

        # Normalize to weights
        total_sharpe = perf['positive_sharpe'].sum()
        weights = {}

        for _, row in perf.iterrows():
            weights[row['strategy_name']] = row['positive_sharpe'] / total_sharpe

        return weights

    def print_performance_report(self, lookback_days: int = 30):
        """Print a formatted performance report"""
        perf = self.get_strategy_performance(lookback_days)

        print("\n" + "="*80)
        print(f"STRATEGY PERFORMANCE REPORT (Last {lookback_days} Days)")
        print("="*80)

        if perf.empty:
            print("No validated signals found (need 5+ days after signal date)")
            return

        print(f"\n{'Strategy':<30} {'Signals':>8} {'Win Rate':>10} {'Avg Ret':>10} {'Sharpe':>8}")
        print("-"*80)

        for _, row in perf.iterrows():
            print(f"{row['strategy_name']:<30} {int(row['num_signals']):>8} "
                  f"{row['win_rate']:>9.1%} {row['avg_return_pct']:>9.2f}% "
                  f"{row['sharpe_ratio']:>8.2f}")

        print("="*80)

        # Show dynamic weights
        weights = self.get_strategy_weights_from_performance(lookback_days)
        print("\nDynamic Weights (based on Sharpe ratios):")
        for strategy, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {strategy:<35} {weight:>6.1%}")

        print()


def main():
    """Example usage"""
    tracker = StrategyPerformanceTracker()

    # Show 30-day performance
    tracker.print_performance_report(lookback_days=30)

    # Show 7-day performance (more recent)
    tracker.print_performance_report(lookback_days=7)


if __name__ == "__main__":
    main()
