#!/usr/bin/env python3
"""
Walk-Forward Validation

Simulates realistic model retraining schedule:
- Train on rolling 6-month window
- Test on next 1-month period
- Retrain monthly
- Measure cumulative performance over 1+ years

This is the GOLD STANDARD for testing if a strategy will work in production.

Usage:
    python scripts/research/walk_forward_validation.py --symbol AAPL --months 12
    python scripts/research/walk_forward_validation.py --tier tier_1 --months 12
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy
from config.tick_config import TICK_DB_PATH
import sqlite3

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Represents one train/test window in walk-forward validation."""
    train_start_date: datetime
    train_end_date: datetime
    test_start_date: datetime
    test_end_date: datetime
    train_bars: pd.DataFrame
    test_bars: pd.DataFrame


class WalkForwardValidator:
    """Walk-forward validation with periodic retraining."""

    def __init__(self, symbol: str, train_months: int = 6, test_months: int = 1):
        self.symbol = symbol
        self.train_months = train_months
        self.test_months = test_months
        self.windows = []
        self.results = []

    def load_data(self) -> bool:
        """Load full historical data."""
        try:
            conn = sqlite3.connect(str(TICK_DB_PATH))
            query = f"""
                SELECT timestamp, price, volume
                FROM ticks
                WHERE symbol = '{self.symbol}'
                ORDER BY timestamp
            """
            ticks_df = pd.read_sql_query(query, conn)
            conn.close()

            if len(ticks_df) == 0:
                logger.error(f"{self.symbol}: No tick data")
                return False

            # Generate imbalance bars
            strategy = RiskLabAIStrategy()
            ticks_df['timestamp'] = pd.to_datetime(ticks_df['timestamp'])
            self.bars_df = strategy.generate_imbalance_bars(ticks_df)

            logger.info(f"{self.symbol}: Loaded {len(self.bars_df)} bars from "
                       f"{self.bars_df.index[0]} to {self.bars_df.index[-1]}")

            return True

        except Exception as e:
            logger.error(f"{self.symbol}: Error loading data - {e}")
            return False

    def create_walk_forward_windows(self) -> List[WalkForwardWindow]:
        """Create rolling train/test windows."""

        windows = []

        # Start with enough data for first training window
        current_date = self.bars_df.index[0] + pd.DateOffset(months=self.train_months)
        end_date = self.bars_df.index[-1] - pd.DateOffset(months=self.test_months)

        window_num = 1
        while current_date < end_date:
            # Training window
            train_end = current_date
            train_start = train_end - pd.DateOffset(months=self.train_months)

            # Test window
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.test_months)

            # Get bars for this window
            train_bars = self.bars_df[(self.bars_df.index >= train_start) &
                                     (self.bars_df.index < train_end)]
            test_bars = self.bars_df[(self.bars_df.index >= test_start) &
                                    (self.bars_df.index < test_end)]

            # Only include if we have sufficient data
            if len(train_bars) >= 100 and len(test_bars) >= 10:
                window = WalkForwardWindow(
                    train_start_date=train_start,
                    train_end_date=train_end,
                    test_start_date=test_start,
                    test_end_date=test_end,
                    train_bars=train_bars,
                    test_bars=test_bars
                )
                windows.append(window)
                logger.info(f"Window {window_num}: Train {len(train_bars)} bars "
                           f"({train_start.date()} to {train_end.date()}), "
                           f"Test {len(test_bars)} bars "
                           f"({test_start.date()} to {test_end.date()})")
                window_num += 1

            # Move forward by test period (monthly retraining)
            current_date += pd.DateOffset(months=self.test_months)

        self.windows = windows
        logger.info(f"\nCreated {len(windows)} walk-forward windows")
        return windows

    def train_and_test_window(self, window: WalkForwardWindow, window_num: int) -> Dict:
        """Train model on training window, test on test window."""

        logger.info(f"\nWindow {window_num}/{len(self.windows)}: "
                   f"{window.test_start_date.date()} to {window.test_end_date.date()}")

        # Create fresh strategy
        strategy = RiskLabAIStrategy()

        # Prepare features
        train_features = strategy.generate_features(window.train_bars)
        test_features = strategy.generate_features(window.test_bars)

        # Prepare labels
        train_labels_primary = strategy.triple_barrier_label(window.train_bars)
        test_labels_primary = strategy.triple_barrier_label(window.test_bars)

        # Train primary model
        strategy.primary_model.fit(train_features, train_labels_primary)

        # Predict on test set
        test_pred_primary = strategy.primary_model.predict(test_features)
        test_pred_proba_primary = strategy.primary_model.predict_proba(test_features)

        # Calculate accuracy
        test_acc = (test_pred_primary == test_labels_primary).mean()

        # Train meta model
        train_pred_primary = strategy.primary_model.predict(train_features)
        train_trade_mask = train_pred_primary != 0

        test_acc_meta = 0.0
        if train_trade_mask.sum() > 10:
            train_labels_meta = (train_labels_primary[train_trade_mask] == train_pred_primary[train_trade_mask]).astype(int)
            strategy.meta_model.fit(train_features[train_trade_mask], train_labels_meta)

            # Test meta model
            test_trade_mask = test_pred_primary != 0
            if test_trade_mask.sum() > 0:
                test_labels_meta = (test_labels_primary[test_trade_mask] == test_pred_primary[test_trade_mask]).astype(int)
                test_pred_meta = strategy.meta_model.predict(test_features[test_trade_mask])
                test_acc_meta = (test_pred_meta == test_labels_meta).mean()

        # Simulate trading on test period
        returns = self.simulate_trading(window.test_bars, test_pred_primary, test_labels_primary)

        result = {
            'window_num': window_num,
            'test_start': window.test_start_date,
            'test_end': window.test_end_date,
            'n_test_bars': len(window.test_bars),
            'test_acc_primary': test_acc,
            'test_acc_meta': test_acc_meta,
            'total_return': returns['total_return'],
            'sharpe_ratio': returns['sharpe_ratio'],
            'win_rate': returns['win_rate'],
            'n_trades': returns['n_trades']
        }

        logger.info(f"  Primary Acc: {test_acc:.1%}")
        logger.info(f"  Meta Acc:    {test_acc_meta:.1%}")
        logger.info(f"  Trades:      {returns['n_trades']}")
        logger.info(f"  Win Rate:    {returns['win_rate']:.1%}")
        logger.info(f"  Return:      {returns['total_return']:.2%}")
        logger.info(f"  Sharpe:      {returns['sharpe_ratio']:.2f}")

        return result

    def simulate_trading(self, bars: pd.DataFrame, predictions: np.ndarray,
                        actual_labels: np.ndarray) -> Dict:
        """Simulate realistic trading on test period."""

        trades = []

        for i in range(len(predictions)):
            if predictions[i] != 0:  # Model predicted a trade
                # Trade result based on actual label
                if predictions[i] == actual_labels[i]:
                    # Correct prediction - win
                    # Assume 4% profit target
                    pnl = 0.04
                else:
                    # Wrong prediction - loss
                    # Assume 2% stop loss
                    pnl = -0.02

                trades.append(pnl)

        if len(trades) == 0:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'n_trades': 0
            }

        # Calculate metrics
        total_return = np.sum(trades)
        win_rate = np.mean([1 if t > 0 else 0 for t in trades])

        # Sharpe ratio (annualized)
        if len(trades) > 1:
            sharpe = np.mean(trades) / (np.std(trades) + 1e-10) * np.sqrt(252)
        else:
            sharpe = 0.0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'n_trades': len(trades)
        }

    def run_walk_forward(self) -> pd.DataFrame:
        """Run complete walk-forward validation."""

        logger.info(f"\n{'='*80}")
        logger.info(f"WALK-FORWARD VALIDATION: {self.symbol}")
        logger.info(f"Training window: {self.train_months} months")
        logger.info(f"Test window: {self.test_months} month(s)")
        logger.info(f"{'='*80}\n")

        if not self.load_data():
            return None

        self.create_walk_forward_windows()

        if len(self.windows) == 0:
            logger.error("No valid windows created - insufficient data")
            return None

        # Test each window
        results = []
        for i, window in enumerate(self.windows, 1):
            result = self.train_and_test_window(window, i)
            results.append(result)

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Calculate cumulative metrics
        self.print_summary(results_df)

        return results_df

    def print_summary(self, results_df: pd.DataFrame):
        """Print summary statistics across all windows."""

        logger.info(f"\n{'='*80}")
        logger.info(f"WALK-FORWARD SUMMARY: {self.symbol}")
        logger.info(f"{'='*80}\n")

        # Overall metrics
        total_periods = len(results_df)
        avg_acc = results_df['test_acc_primary'].mean()
        std_acc = results_df['test_acc_primary'].std()

        cumulative_return = results_df['total_return'].sum()
        avg_monthly_return = results_df['total_return'].mean()
        annualized_return = avg_monthly_return * 12

        total_trades = results_df['n_trades'].sum()
        avg_win_rate = results_df['win_rate'].mean()
        avg_sharpe = results_df['sharpe_ratio'].mean()

        logger.info(f"Periods tested:        {total_periods}")
        logger.info(f"Average accuracy:      {avg_acc:.1%} ± {std_acc:.1%}")
        logger.info(f"\nTRADING PERFORMANCE:")
        logger.info(f"Total trades:          {total_trades}")
        logger.info(f"Average win rate:      {avg_win_rate:.1%}")
        logger.info(f"Cumulative return:     {cumulative_return:.2%}")
        logger.info(f"Annualized return:     {annualized_return:.2%}")
        logger.info(f"Average Sharpe:        {avg_sharpe:.2f}")

        # Stability check
        positive_periods = (results_df['total_return'] > 0).sum()
        consistency = positive_periods / total_periods

        logger.info(f"\nSTABILITY:")
        logger.info(f"Positive periods:      {positive_periods}/{total_periods} ({consistency:.1%})")

        if consistency < 0.5:
            logger.warning(f"⚠️  WARNING: Less than 50% of periods were profitable")
            logger.warning(f"   Strategy may not be robust")

        if avg_acc < 0.55:
            logger.warning(f"⚠️  WARNING: Average accuracy {avg_acc:.1%} is close to random (50%)")
            logger.warning(f"   Model may lack predictive power")

        # Print period-by-period results
        logger.info(f"\nPERIOD-BY-PERIOD RESULTS:")
        logger.info(f"{'Period':<15} {'Test Dates':<25} {'Acc':<8} {'Trades':<8} {'Win%':<8} {'Return':<10} {'Sharpe':<8}")
        logger.info("-" * 90)
        for _, row in results_df.iterrows():
            logger.info(f"{row['window_num']:<15} "
                       f"{row['test_start'].strftime('%Y-%m-%d')} - {row['test_end'].strftime('%Y-%m-%d'):<12} "
                       f"{row['test_acc_primary']:>6.1%}  "
                       f"{row['n_trades']:>6}  "
                       f"{row['win_rate']:>6.1%}  "
                       f"{row['total_return']:>8.2%}  "
                       f"{row['sharpe_ratio']:>6.2f}")


def main():
    parser = argparse.ArgumentParser(description='Walk-forward validation with retraining')
    parser.add_argument('--symbol', type=str, help='Symbol to test')
    parser.add_argument('--tier', type=str, help='Tier to test')
    parser.add_argument('--months', type=int, default=12, help='Number of months to test (default: 12)')
    parser.add_argument('--train-months', type=int, default=6, help='Training window in months (default: 6)')
    parser.add_argument('--test-months', type=int, default=1, help='Test window in months (default: 1)')

    args = parser.parse_args()

    # Get symbols
    symbols = []
    if args.symbol:
        symbols = [args.symbol]
    elif args.tier:
        from config.all_symbols import get_symbols_by_tier
        symbols = get_symbols_by_tier(args.tier)[:10]  # Limit to first 10 for testing
        logger.info(f"Testing first 10 symbols from {args.tier}")
    else:
        logger.error("Must specify --symbol or --tier")
        return 1

    # Run walk-forward validation
    all_results = {}

    for symbol in symbols:
        validator = WalkForwardValidator(
            symbol=symbol,
            train_months=args.train_months,
            test_months=args.test_months
        )

        results_df = validator.run_walk_forward()
        if results_df is not None:
            all_results[symbol] = results_df

    # Aggregate summary across all symbols
    if len(all_results) > 1:
        logger.info(f"\n{'='*80}")
        logger.info(f"AGGREGATE SUMMARY ACROSS ALL SYMBOLS")
        logger.info(f"{'='*80}\n")

        all_returns = []
        all_accuracies = []
        all_win_rates = []

        for symbol, df in all_results.items():
            all_returns.extend(df['total_return'].tolist())
            all_accuracies.extend(df['test_acc_primary'].tolist())
            all_win_rates.extend(df['win_rate'].tolist())

        avg_return = np.mean(all_returns)
        annualized = avg_return * 12
        avg_acc = np.mean(all_accuracies)
        avg_wr = np.mean(all_win_rates)

        logger.info(f"Symbols tested:        {len(all_results)}")
        logger.info(f"Total periods:         {len(all_returns)}")
        logger.info(f"Average accuracy:      {avg_acc:.1%}")
        logger.info(f"Average win rate:      {avg_wr:.1%}")
        logger.info(f"Avg monthly return:    {avg_return:.2%}")
        logger.info(f"Annualized return:     {annualized:.2%}")

        if annualized > 0.10:
            logger.info(f"\n✅ PROMISING: {annualized:.1%} annualized return suggests viable strategy")
        elif annualized > 0:
            logger.info(f"\n⚠️  MARGINAL: {annualized:.1%} annualized return is positive but low")
        else:
            logger.info(f"\n❌ POOR: Negative annualized return - strategy not working")


if __name__ == "__main__":
    main()
