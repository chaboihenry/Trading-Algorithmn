"""
Backtesting Script for All Strategies

This script backtests three configurations:
1. Sentiment-only strategy
2. Pairs-only strategy
3. Combined strategy (meta-learner ensemble)

It compares performance metrics across all strategies for the period
February 2020 to December 2023.

Usage:
    python run_backtest.py

Output:
    - Comparative performance metrics
    - Plots saved to backtests/results/
    - Summary table printed to console
"""

import logging
from datetime import datetime
from lumibot.backtesting import YahooDataBacktesting
from lumibot.traders import Trader
from lumibot.entities import TradingFee
import pandas as pd
import numpy as np
from pathlib import Path

# Import our strategies
from sentiment_strategy import SentimentStrategy
from pairs_strategy import PairsStrategy
from combined_strategy import CombinedStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Backtest configuration
START_DATE = datetime(2020, 2, 1)
END_DATE = datetime(2023, 12, 31)
INITIAL_CAPITAL = 100_000  # $100k starting capital


class BacktestRunner:
    """
    Manages backtesting of multiple strategies and compares results.
    """

    def __init__(self, start_date: datetime, end_date: datetime, initial_capital: float):
        """
        Initialize backtest runner.

        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_capital: Starting portfolio value
        """
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital

        # Results storage
        self.results = {}

        # Output directory
        self.output_dir = Path(__file__).parent / 'backtest_results'
        self.output_dir.mkdir(exist_ok=True)

    def run_backtest(self, strategy_class, strategy_name: str, parameters: dict = None):
        """
        Run backtest for a single strategy.

        Args:
            strategy_class: Strategy class to backtest
            strategy_name: Name for results tracking
            parameters: Optional parameters for strategy

        Returns:
            Backtest results object
        """
        logger.info("=" * 80)
        logger.info(f"BACKTESTING: {strategy_name}")
        logger.info("=" * 80)
        logger.info(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info("-" * 80)

        # Configure strategy
        strategy = strategy_class()

        if parameters:
            strategy.initialize(parameters)

        # Run backtest using Yahoo data
        try:
            results = strategy.backtest(
                YahooDataBacktesting,
                self.start_date,
                self.end_date,
                parameters=parameters or {},
                buy_trading_fees=[TradingFee(flat_fee=0)],  # Free trading
                sell_trading_fees=[TradingFee(flat_fee=0)]
            )

            self.results[strategy_name] = results

            logger.info(f"Backtest complete for {strategy_name}")
            logger.info("=" * 80)

            return results

        except Exception as e:
            logger.error(f"Error running backtest for {strategy_name}: {e}")
            logger.exception("Full traceback:")
            return None

    def calculate_metrics(self, results) -> dict:
        """
        Calculate comprehensive performance metrics.

        Args:
            results: Backtest results object

        Returns:
            Dictionary of metrics
        """
        if results is None:
            return {}

        try:
            # Extract portfolio values
            portfolio_values = results['portfolio_value']

            # Calculate returns
            returns = portfolio_values.pct_change().dropna()
            cumulative_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1

            # Calculate CAGR
            days = (self.end_date - self.start_date).days
            years = days / 365.25
            cagr = (1 + cumulative_return) ** (1 / years) - 1

            # Calculate Sharpe Ratio (assuming 0% risk-free rate)
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

            # Calculate max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # Get trades
            trades = results.get('trades', [])
            winning_trades = [t for t in trades if t.get('profit', 0) > 0]
            losing_trades = [t for t in trades if t.get('profit', 0) < 0]

            win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0

            metrics = {
                'Cumulative Return': cumulative_return,
                'CAGR': cagr,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown': max_drawdown,
                'Win Rate': win_rate,
                'Total Trades': len(trades),
                'Winning Trades': len(winning_trades),
                'Losing Trades': len(losing_trades),
                'Final Portfolio Value': portfolio_values.iloc[-1],
                'Total Return $': portfolio_values.iloc[-1] - self.initial_capital
            }

            return metrics

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}

    def print_comparison_table(self):
        """
        Print a formatted comparison table of all strategies.
        """
        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST RESULTS COMPARISON")
        logger.info("=" * 80)

        # Calculate metrics for all strategies
        all_metrics = {}
        for strategy_name, results in self.results.items():
            all_metrics[strategy_name] = self.calculate_metrics(results)

        # Create comparison DataFrame
        df = pd.DataFrame(all_metrics).T

        # Format percentages
        percentage_cols = ['Cumulative Return', 'CAGR', 'Max Drawdown', 'Win Rate']
        for col in percentage_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x*100:.2f}%")

        # Format currency
        currency_cols = ['Final Portfolio Value', 'Total Return $']
        for col in currency_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"${x:,.2f}")

        # Format ratios
        if 'Sharpe Ratio' in df.columns:
            df['Sharpe Ratio'] = df['Sharpe Ratio'].apply(lambda x: f"{x:.3f}")

        print("\n" + df.to_string())
        print("\n" + "=" * 80)

        # Save to CSV
        csv_path = self.output_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_path)
        logger.info(f"\nResults saved to: {csv_path}")

        # Determine winner
        cumulative_returns = {name: metrics.get('Cumulative Return', 0)
                             for name, metrics in all_metrics.items()}

        best_strategy = max(cumulative_returns, key=cumulative_returns.get)
        logger.info(f"\n🏆 BEST PERFORMING STRATEGY: {best_strategy}")
        logger.info(f"   Cumulative Return: {cumulative_returns[best_strategy]*100:.2f}%")
        logger.info("=" * 80 + "\n")

    def run_all_backtests(self):
        """
        Run backtests for all three strategy configurations.
        """
        logger.info("\n" + "#" * 80)
        logger.info("STARTING COMPREHENSIVE BACKTEST SUITE")
        logger.info("#" * 80)
        logger.info(f"Date Range: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")
        logger.info("#" * 80 + "\n")

        # 1. Sentiment-only strategy
        self.run_backtest(
            SentimentStrategy,
            "Sentiment Only",
            parameters={}
        )

        # 2. Pairs-only strategy
        self.run_backtest(
            PairsStrategy,
            "Pairs Only",
            parameters={'db_path': '/Volumes/Vault/85_assets_prediction.db'}
        )

        # 3. Combined strategy
        self.run_backtest(
            CombinedStrategy,
            "Combined (Meta-Learner)",
            parameters={
                'db_path': '/Volumes/Vault/85_assets_prediction.db',
                'retrain': True  # Train fresh meta-model
            }
        )

        # Print comparison
        self.print_comparison_table()


def main():
    """Main entry point for backtesting."""
    runner = BacktestRunner(
        start_date=START_DATE,
        end_date=END_DATE,
        initial_capital=INITIAL_CAPITAL
    )

    runner.run_all_backtests()


if __name__ == "__main__":
    main()
