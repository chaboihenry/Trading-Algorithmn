#!/usr/bin/env python3
"""
Multi-Symbol Portfolio Backtest

Comprehensive backtest that simulates real trading across all trained models.
Calculates realistic profit metrics including:
- Total return & Sharpe ratio
- Win rate & profit factor
- Maximum drawdown
- Per-symbol performance
- Trade-by-trade analysis

This simulates what would happen in your paper/live account.

Usage:
    # Test all tier_1 models
    python test_suite/backtest_multi_symbol.py --tier tier_1

    # Test specific symbols
    python test_suite/backtest_multi_symbol.py --symbols AAPL MSFT GOOGL

    # Custom parameters
    python test_suite/backtest_multi_symbol.py --tier tier_1 --capital 100000 --bars 1000
"""

import sys
import argparse
import logging
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy
from config.tick_config import (
    TICK_DB_PATH,
    INITIAL_IMBALANCE_THRESHOLD,
    OPTIMAL_PROFIT_TARGET,
    OPTIMAL_STOP_LOSS,
    OPTIMAL_MAX_HOLDING_BARS,
    OPTIMAL_FRACTIONAL_D,
    OPTIMAL_META_THRESHOLD,
    OPTIMAL_PROB_THRESHOLD
)
from data.tick_storage import TickStorage
from data.tick_to_bars import generate_bars_from_ticks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Portfolio:
    """Portfolio manager for backtest."""

    def __init__(self, starting_cash: float = 100000):
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.positions = {}  # symbol -> {'quantity': int, 'entry_price': float, 'entry_time': datetime}
        self.trades = []  # List of all trades executed
        self.equity_curve = []  # Portfolio value over time

    def get_position(self, symbol: str) -> dict:
        """Get current position for symbol."""
        return self.positions.get(symbol)

    def get_position_value(self, symbol: str, current_price: float) -> float:
        """Get current value of position."""
        pos = self.positions.get(symbol)
        if pos:
            return pos['quantity'] * current_price
        return 0.0

    def get_total_position_value(self, prices: Dict[str, float]) -> float:
        """Get total value of all positions."""
        return sum(
            pos['quantity'] * prices.get(symbol, pos['entry_price'])
            for symbol, pos in self.positions.items()
        )

    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Get total portfolio value (cash + positions)."""
        return self.cash + self.get_total_position_value(prices)

    def execute_trade(self, symbol: str, quantity: int, price: float,
                     timestamp: datetime, action: str, signal: int) -> bool:
        """Execute a trade and update portfolio."""

        trade_value = quantity * price
        commission = 0  # Assume zero commission for now

        if action == 'BUY':
            # Check if we have enough cash
            if trade_value > self.cash:
                logger.warning(f"{symbol}: Insufficient cash for BUY (need ${trade_value:.2f}, have ${self.cash:.2f})")
                return False

            # Execute buy
            self.cash -= trade_value + commission
            self.positions[symbol] = {
                'quantity': quantity,
                'entry_price': price,
                'entry_time': timestamp
            }

            trade_record = {
                'timestamp': timestamp,
                'symbol': symbol,
                'action': 'BUY',
                'quantity': quantity,
                'price': price,
                'value': trade_value,
                'commission': commission,
                'cash_after': self.cash,
                'signal': signal
            }
            self.trades.append(trade_record)
            return True

        elif action == 'SELL':
            # Check if we have the position
            pos = self.positions.get(symbol)
            if not pos or pos['quantity'] < quantity:
                logger.warning(f"{symbol}: Insufficient shares for SELL")
                return False

            # Execute sell
            self.cash += trade_value - commission

            # Calculate P&L
            cost_basis = pos['quantity'] * pos['entry_price']
            pnl = trade_value - cost_basis
            pnl_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0
            holding_time = (timestamp - pos['entry_time']).total_seconds() / 3600  # hours

            trade_record = {
                'timestamp': timestamp,
                'symbol': symbol,
                'action': 'SELL',
                'quantity': quantity,
                'price': price,
                'value': trade_value,
                'commission': commission,
                'cash_after': self.cash,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'holding_time_hours': holding_time,
                'entry_price': pos['entry_price'],
                'signal': signal
            }
            self.trades.append(trade_record)

            # Remove position
            del self.positions[symbol]
            return True

        return False

    def record_equity(self, timestamp: datetime, prices: Dict[str, float]):
        """Record current portfolio value."""
        portfolio_value = self.get_portfolio_value(prices)
        self.equity_curve.append({
            'timestamp': timestamp,
            'cash': self.cash,
            'positions_value': self.get_total_position_value(prices),
            'total_value': portfolio_value
        })


class MultiSymbolBacktest:
    """Backtest engine for multiple symbols."""

    def __init__(self, symbols: List[str], starting_cash: float = 100000,
                 bars_to_simulate: int = 500, kelly_fraction: float = 0.1):
        self.symbols = symbols
        self.starting_cash = starting_cash
        self.bars_to_simulate = bars_to_simulate
        self.kelly_fraction = kelly_fraction  # Conservative Kelly

        # Portfolio
        self.portfolio = Portfolio(starting_cash)

        # Load models for each symbol
        self.models = {}
        self.load_models()

        # Historical bars for each symbol
        self.symbol_bars = {}

    def load_models(self):
        """Load trained models for all symbols."""
        logger.info("=" * 80)
        logger.info("LOADING MODELS")
        logger.info("=" * 80)

        models_dir = Path("models")
        loaded = 0
        failed = []

        for symbol in self.symbols:
            model_path = models_dir / f"risklabai_{symbol}_models.pkl"

            if not model_path.exists():
                logger.warning(f"{symbol}: Model not found at {model_path}")
                failed.append(symbol)
                continue

            try:
                strategy = RiskLabAIStrategy(
                    profit_taking=OPTIMAL_PROFIT_TARGET,
                    stop_loss=OPTIMAL_STOP_LOSS,
                    max_holding=OPTIMAL_MAX_HOLDING_BARS,
                    d=OPTIMAL_FRACTIONAL_D,
                    n_cv_splits=5,
                    force_directional=True,
                    neutral_threshold=0.00001
                )
                strategy.load_models(str(model_path))
                self.models[symbol] = strategy
                loaded += 1
                logger.info(f"  ✓ {symbol}")
            except Exception as e:
                logger.error(f"  ✗ {symbol}: {e}")
                failed.append(symbol)

        logger.info(f"\nLoaded {loaded}/{len(self.symbols)} models")
        if failed:
            logger.warning(f"Failed to load: {', '.join(failed)}")
            # Remove failed symbols
            self.symbols = [s for s in self.symbols if s not in failed]
        logger.info("")

    def load_historical_data(self):
        """Load historical tick data and generate bars for all symbols."""
        logger.info("=" * 80)
        logger.info("LOADING HISTORICAL DATA")
        logger.info("=" * 80)

        storage = TickStorage(TICK_DB_PATH)

        for symbol in self.symbols:
            logger.info(f"\n{symbol}:")

            try:
                # Get date range
                date_range = storage.get_date_range(symbol)
                if not date_range:
                    logger.warning(f"  No tick data found")
                    continue

                earliest, latest = date_range
                logger.info(f"  Data range: {earliest} to {latest}")

                # Load ticks
                ticks = storage.load_ticks(symbol, limit=5000000)
                if not ticks:
                    logger.warning(f"  Failed to load ticks")
                    continue

                logger.info(f"  Loaded {len(ticks):,} ticks")

                # Generate bars
                bars_list = generate_bars_from_ticks(ticks, threshold=INITIAL_IMBALANCE_THRESHOLD)
                if not bars_list:
                    logger.warning(f"  Failed to generate bars")
                    continue

                logger.info(f"  Generated {len(bars_list)} bars")

                # Convert to DataFrame
                bars_df = pd.DataFrame(bars_list)
                bars_df['bar_end'] = pd.to_datetime(bars_df['bar_end'])
                if bars_df['bar_end'].dt.tz is not None:
                    bars_df['bar_end'] = bars_df['bar_end'].dt.tz_localize(None)
                bars_df.set_index('bar_end', inplace=True)

                # Store last N bars
                self.symbol_bars[symbol] = bars_df.iloc[-self.bars_to_simulate:].copy()
                logger.info(f"  ✓ Using {len(self.symbol_bars[symbol])} bars for backtest")

            except Exception as e:
                logger.error(f"  ✗ Error loading data: {e}")

        storage.close()
        logger.info(f"\nLoaded data for {len(self.symbol_bars)}/{len(self.symbols)} symbols")
        logger.info("")

    def run_backtest(self):
        """Run the backtest."""
        logger.info("=" * 80)
        logger.info("RUNNING BACKTEST")
        logger.info("=" * 80)
        logger.info(f"Symbols: {len(self.symbols)}")
        logger.info(f"Starting capital: ${self.starting_cash:,.2f}")
        logger.info(f"Kelly fraction: {self.kelly_fraction}")
        logger.info("")

        # Load data
        self.load_historical_data()

        if not self.symbol_bars:
            logger.error("No data loaded, cannot run backtest")
            return

        # Get common time range across all symbols
        earliest_start = max(bars.index[0] for bars in self.symbol_bars.values())
        latest_end = min(bars.index[-1] for bars in self.symbol_bars.values())

        logger.info(f"Common time range: {earliest_start} to {latest_end}")
        logger.info("")

        # Simulate trading bar by bar
        logger.info("Simulating trading...")
        iterations = 0
        check_interval = 5  # Check for signals every 5 bars

        # Initialize bar counters for each symbol
        bar_counts = {symbol: 50 for symbol in self.symbol_bars.keys()}  # Start at bar 50

        while True:
            # Get current timestamp (earliest timestamp across all symbols at current bar)
            current_times = {}
            for symbol, start_idx in bar_counts.items():
                if start_idx < len(self.symbol_bars[symbol]):
                    current_times[symbol] = self.symbol_bars[symbol].index[start_idx]

            if not current_times:
                break  # All symbols exhausted

            current_time = min(current_times.values())

            # Get current prices
            current_prices = {}
            for symbol, bars in self.symbol_bars.items():
                if symbol in current_times:
                    idx = bar_counts[symbol]
                    if idx < len(bars):
                        current_prices[symbol] = bars['close'].iloc[idx]

            # Record equity
            self.portfolio.record_equity(current_time, current_prices)

            # Check each symbol for trading signals
            for symbol in list(self.symbol_bars.keys()):
                if symbol not in self.models:
                    continue

                idx = bar_counts[symbol]
                if idx >= len(self.symbol_bars[symbol]):
                    continue

                # Get bars up to now
                bars_up_to_now = self.symbol_bars[symbol].iloc[:idx+1].copy()

                # Need minimum bars for features
                if len(bars_up_to_now) < 50:
                    continue

                # Get signal every N bars
                if idx % check_interval != 0:
                    continue

                try:
                    # Get prediction
                    signal, bet_size = self.models[symbol].predict(
                        bars_up_to_now,
                        prob_threshold=OPTIMAL_PROB_THRESHOLD,
                        meta_threshold=OPTIMAL_META_THRESHOLD
                    )

                    # Filter low confidence
                    if bet_size < 0.5:
                        signal = 0

                    # Execute based on signal
                    current_price = current_prices[symbol]
                    current_position = self.portfolio.get_position(symbol)
                    portfolio_value = self.portfolio.get_portfolio_value(current_prices)

                    if signal == 1:  # Long
                        if not current_position:
                            # Enter long
                            position_value = portfolio_value * self.kelly_fraction * bet_size
                            quantity = int(position_value / current_price)

                            if quantity > 0:
                                success = self.portfolio.execute_trade(
                                    symbol, quantity, current_price,
                                    current_time, 'BUY', signal
                                )
                                if success:
                                    logger.info(f"{current_time} | {symbol}: BUY {quantity} @ ${current_price:.2f} (bet_size={bet_size:.2f})")

                    elif signal == -1:  # Exit long (or short signal)
                        if current_position:
                            # Close position
                            quantity = current_position['quantity']
                            success = self.portfolio.execute_trade(
                                symbol, quantity, current_price,
                                current_time, 'SELL', signal
                            )
                            if success:
                                last_trade = self.portfolio.trades[-1]
                                logger.info(f"{current_time} | {symbol}: SELL {quantity} @ ${current_price:.2f} | P&L: ${last_trade['pnl']:.2f} ({last_trade['pnl_pct']:.1f}%)")

                except Exception as e:
                    logger.error(f"{symbol}: Prediction error: {e}")

            # Advance all symbols
            for symbol in bar_counts:
                bar_counts[symbol] += 1

            iterations += 1

            # Progress update
            if iterations % 20 == 0:
                pv = self.portfolio.get_portfolio_value(current_prices)
                ret = (pv - self.starting_cash) / self.starting_cash * 100
                logger.info(f"  [{iterations:4d} iterations] Portfolio: ${pv:,.2f} ({ret:+.2f}%)")

        logger.info(f"\n✓ Backtest complete ({iterations} iterations)")
        logger.info("")

        # Close any remaining positions
        final_prices = {symbol: bars['close'].iloc[-1] for symbol, bars in self.symbol_bars.items()}
        for symbol, pos in list(self.portfolio.positions.items()):
            if symbol in final_prices:
                self.portfolio.execute_trade(
                    symbol, pos['quantity'], final_prices[symbol],
                    self.symbol_bars[symbol].index[-1], 'SELL', 0
                )
                logger.info(f"Closing {symbol}: {pos['quantity']} @ ${final_prices[symbol]:.2f}")

    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if not self.portfolio.trades or not self.portfolio.equity_curve:
            return {}

        # Portfolio metrics
        equity_df = pd.DataFrame(self.portfolio.equity_curve)
        final_value = equity_df['total_value'].iloc[-1]
        total_return = (final_value - self.starting_cash) / self.starting_cash * 100

        # Calculate daily returns for Sharpe
        equity_df['returns'] = equity_df['total_value'].pct_change()
        daily_returns = equity_df['returns'].dropna()

        sharpe = 0
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()

        # Max drawdown
        equity_df['cummax'] = equity_df['total_value'].cummax()
        equity_df['drawdown'] = (equity_df['total_value'] - equity_df['cummax']) / equity_df['cummax'] * 100
        max_drawdown = equity_df['drawdown'].min()

        # Trade metrics
        trades_df = pd.DataFrame(self.portfolio.trades)
        completed_trades = trades_df[trades_df['action'] == 'SELL']

        num_trades = len(completed_trades)
        winning_trades = completed_trades[completed_trades['pnl'] > 0]
        losing_trades = completed_trades[completed_trades['pnl'] <= 0]

        win_rate = len(winning_trades) / num_trades * 100 if num_trades > 0 else 0

        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0

        profit_factor = (winning_trades['pnl'].sum() / abs(losing_trades['pnl'].sum())
                        if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else float('inf'))

        return {
            'starting_capital': self.starting_cash,
            'final_value': final_value,
            'total_return': total_return,
            'total_pnl': final_value - self.starting_cash,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_holding_hours': completed_trades['holding_time_hours'].mean() if num_trades > 0 else 0
        }

    def print_results(self):
        """Print backtest results."""
        metrics = self.calculate_metrics()

        if not metrics:
            logger.error("No metrics to display")
            return

        logger.info("=" * 80)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 80)
        logger.info("")

        # Portfolio performance
        logger.info("PORTFOLIO PERFORMANCE:")
        logger.info(f"  Starting Capital:    ${metrics['starting_capital']:>12,.2f}")
        logger.info(f"  Final Value:         ${metrics['final_value']:>12,.2f}")
        logger.info(f"  Total P&L:           ${metrics['total_pnl']:>12,.2f}")
        logger.info(f"  Total Return:        {metrics['total_return']:>12.2f}%")
        logger.info(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>12.2f}")
        logger.info(f"  Max Drawdown:        {metrics['max_drawdown']:>12.2f}%")
        logger.info("")

        # Trade statistics
        logger.info("TRADE STATISTICS:")
        logger.info(f"  Total Trades:        {metrics['num_trades']:>12d}")
        logger.info(f"  Win Rate:            {metrics['win_rate']:>12.1f}%")
        logger.info(f"  Average Win:         ${metrics['avg_win']:>12,.2f}")
        logger.info(f"  Average Loss:        ${metrics['avg_loss']:>12,.2f}")
        logger.info(f"  Profit Factor:       {metrics['profit_factor']:>12.2f}")
        logger.info(f"  Avg Hold Time:       {metrics['avg_holding_hours']:>12.1f} hours")
        logger.info("")

        # Per-symbol breakdown
        if self.portfolio.trades:
            trades_df = pd.DataFrame(self.portfolio.trades)
            completed = trades_df[trades_df['action'] == 'SELL']

            if len(completed) > 0:
                logger.info("PER-SYMBOL PERFORMANCE:")
                symbol_pnl = completed.groupby('symbol')['pnl'].agg(['sum', 'count', 'mean'])
                symbol_pnl = symbol_pnl.sort_values('sum', ascending=False)

                for symbol, row in symbol_pnl.iterrows():
                    logger.info(f"  {symbol:6s}: {row['count']:3.0f} trades | "
                              f"Total P&L: ${row['sum']:>10,.2f} | "
                              f"Avg: ${row['mean']:>8,.2f}")
                logger.info("")

        logger.info("=" * 80)

        # Verdict
        if metrics['total_return'] > 0 and metrics['sharpe_ratio'] > 1.0:
            logger.info("✅ STRONG POSITIVE RESULTS - Strategy shows promise!")
        elif metrics['total_return'] > 0:
            logger.info("⚠️  POSITIVE BUT WEAK - Strategy is profitable but needs improvement")
        else:
            logger.info("❌ NEGATIVE RESULTS - Strategy needs significant improvement")
        logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Multi-symbol portfolio backtest')
    parser.add_argument('--tier', type=str, help='Tier to backtest (tier_1, tier_2, etc.)')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to backtest')
    parser.add_argument('--capital', type=float, default=100000, help='Starting capital')
    parser.add_argument('--bars', type=int, default=500, help='Number of bars to simulate')
    parser.add_argument('--kelly', type=float, default=0.1, help='Kelly fraction (0.1 = 10% per trade)')

    args = parser.parse_args()

    # Determine symbols to test
    symbols = []
    if args.symbols:
        symbols = args.symbols
    elif args.tier:
        try:
            from config.all_symbols import get_symbols_by_tier
            symbols = get_symbols_by_tier(args.tier)
            logger.info(f"Testing {len(symbols)} symbols from {args.tier}")
        except ImportError:
            logger.error("Could not load all_symbols.py")
            return 1
    else:
        # Find all trained models
        models_dir = Path("models")
        model_files = list(models_dir.glob("risklabai_*_models.pkl"))
        symbols = [f.stem.replace('risklabai_', '').replace('_models', '') for f in model_files]
        logger.info(f"Testing all {len(symbols)} trained models")

    if not symbols:
        logger.error("No symbols specified. Use --tier or --symbols")
        return 1

    # Run backtest
    backtest = MultiSymbolBacktest(
        symbols=symbols,
        starting_cash=args.capital,
        bars_to_simulate=args.bars,
        kelly_fraction=args.kelly
    )

    backtest.run_backtest()
    backtest.print_results()

    return 0


if __name__ == "__main__":
    sys.exit(main())
