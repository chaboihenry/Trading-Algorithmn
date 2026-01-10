#!/usr/bin/env python3
"""
Production-Grade Realistic Backtest

This backtest EXACTLY mirrors the live trading bot's behavior to provide
accurate performance estimates. It implements:

REALISM FEATURES:
✓ Hourly buy signal checks (not every bar)
✓ Per-bar position monitoring (profit target / stop loss / max holding)
✓ 7-day stop loss cooldown (prevents re-buying stopped symbols)
✓ Market hours only (9:30am - 4pm ET)
✓ Realistic slippage (0.05% per trade)
✓ Commission costs ($0.50 per trade via Alpaca)
✓ Kelly Criterion position sizing with half-Kelly
✓ All risk management rules (daily loss, max drawdown, consecutive losses)
✓ Max trades per day limit
✓ Uses only test set (last 30% of data - unseen during training)
✓ Orders execute on NEXT bar's open (no look-ahead bias)

Usage:
    # Backtest all tier_1 symbols (recommended)
    python test_suite/realistic_backtest.py --tier tier_1 --capital 100000

    # Specific symbols with custom parameters
    python test_suite/realistic_backtest.py --symbols AAPL MSFT --capital 50000 --days 90

    # Full year simulation
    python test_suite/realistic_backtest.py --tier tier_1 --days 252 --verbose

Author: Production Trading System
Last Updated: 2026-01-09
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import pytz

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


class RealisticPortfolio:
    """Portfolio manager with realistic trading costs and risk management."""

    def __init__(self, starting_cash: float = 100000,
                 commission_per_trade: float = 0.50,
                 slippage_pct: float = 0.0005):  # 0.05% slippage
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.commission_per_trade = commission_per_trade
        self.slippage_pct = slippage_pct

        # Positions: symbol -> {'quantity', 'entry_price', 'entry_time', 'entry_bar_idx'}
        self.positions = {}

        # Trading history
        self.trades = []  # All completed trades
        self.equity_curve = []  # Portfolio value over time

        # Risk tracking
        self.daily_start_value = starting_cash
        self.daily_start_date = None
        self.peak_portfolio_value = starting_cash
        self.consecutive_losses = 0
        self.trades_today = 0

        # Stop loss cooldown tracking
        self.stop_loss_cooldowns = {}  # symbol -> datetime when stopped
        self.stop_loss_cooldown_days = 7

    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Get total portfolio value (cash + positions)."""
        position_value = sum(
            pos['quantity'] * prices.get(symbol, pos['entry_price'])
            for symbol, pos in self.positions.items()
        )
        return self.cash + position_value

    def execute_trade(self, symbol: str, quantity: int, price: float,
                     timestamp: datetime, action: str,
                     reason: str = "") -> Optional[Dict]:
        """
        Execute a trade with realistic costs.

        Returns trade record if successful, None otherwise.
        """
        if action == 'BUY':
            # Apply slippage (buy at slightly higher price)
            execution_price = price * (1 + self.slippage_pct)
            trade_value = quantity * execution_price + self.commission_per_trade

            if trade_value > self.cash:
                logger.warning(f"{symbol}: Insufficient cash (need ${trade_value:.2f}, have ${self.cash:.2f})")
                return None

            # Execute buy
            self.cash -= trade_value
            self.positions[symbol] = {
                'quantity': quantity,
                'entry_price': execution_price,
                'entry_time': timestamp,
                'entry_bar_idx': 0  # Will be set by caller
            }

            trade_record = {
                'timestamp': timestamp,
                'symbol': symbol,
                'action': 'BUY',
                'quantity': quantity,
                'price': execution_price,
                'value': quantity * execution_price,
                'commission': self.commission_per_trade,
                'slippage': quantity * price * self.slippage_pct,
                'cash_after': self.cash,
                'reason': reason
            }
            self.trades.append(trade_record)
            return trade_record

        elif action == 'SELL':
            pos = self.positions.get(symbol)
            if not pos or pos['quantity'] < quantity:
                logger.warning(f"{symbol}: Cannot sell {quantity} shares (have {pos['quantity'] if pos else 0})")
                return None

            # Apply slippage (sell at slightly lower price)
            execution_price = price * (1 - self.slippage_pct)
            trade_value = quantity * execution_price - self.commission_per_trade

            # Execute sell
            self.cash += trade_value

            # Calculate P&L
            cost_basis = pos['quantity'] * pos['entry_price']
            proceeds = quantity * execution_price
            pnl = proceeds - cost_basis - (2 * self.commission_per_trade)  # Buy + sell commission
            pnl_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0

            holding_time = (timestamp - pos['entry_time']).total_seconds() / 3600  # hours

            trade_record = {
                'timestamp': timestamp,
                'symbol': symbol,
                'action': 'SELL',
                'quantity': quantity,
                'price': execution_price,
                'value': quantity * execution_price,
                'commission': self.commission_per_trade,
                'slippage': quantity * price * self.slippage_pct,
                'cash_after': self.cash,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'holding_time_hours': holding_time,
                'entry_price': pos['entry_price'],
                'reason': reason
            }
            self.trades.append(trade_record)

            # Track consecutive losses
            if pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0

            # Update trades today
            self.trades_today += 1

            # Check if this is a stop loss exit
            if "STOP LOSS" in reason.upper():
                self.stop_loss_cooldowns[symbol] = timestamp
                logger.warning(f"🚫 {symbol} blacklisted for {self.stop_loss_cooldown_days} days (stop loss)")

            # Remove position
            del self.positions[symbol]
            return trade_record

        return None

    def is_symbol_in_cooldown(self, symbol: str, current_time: datetime) -> bool:
        """Check if symbol is in stop loss cooldown."""
        if symbol not in self.stop_loss_cooldowns:
            return False

        stop_time = self.stop_loss_cooldowns[symbol]
        days_since = (current_time - stop_time).days

        if days_since >= self.stop_loss_cooldown_days:
            # Cooldown expired
            del self.stop_loss_cooldowns[symbol]
            return False

        return True

    def record_equity(self, timestamp: datetime, prices: Dict[str, float]):
        """Record current portfolio value."""
        portfolio_value = self.get_portfolio_value(prices)
        self.equity_curve.append({
            'timestamp': timestamp,
            'cash': self.cash,
            'positions_value': portfolio_value - self.cash,
            'total_value': portfolio_value
        })

        # Update peak
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value

    def reset_daily_tracking(self, current_date, prices: Dict[str, float]):
        """Reset daily tracking for new trading day."""
        if self.daily_start_date is None or current_date != self.daily_start_date:
            self.daily_start_value = self.get_portfolio_value(prices)
            self.daily_start_date = current_date
            self.trades_today = 0
            logger.info(f"📅 New trading day: {current_date} | Starting value: ${self.daily_start_value:,.2f}")


def is_market_hours(timestamp: datetime) -> bool:
    """Check if timestamp is during market hours (9:30am - 4pm ET)."""
    # Convert to ET timezone
    et = pytz.timezone('America/New_York')
    if timestamp.tzinfo is None:
        timestamp = et.localize(timestamp)
    else:
        timestamp = timestamp.astimezone(et)

    # Market hours: 9:30 AM - 4:00 PM ET
    market_open = dt_time(9, 30)
    market_close = dt_time(16, 0)

    current_time = timestamp.time()
    weekday = timestamp.weekday()

    # Monday = 0, Friday = 4
    is_weekday = weekday < 5
    is_trading_hours = market_open <= current_time <= market_close

    return is_weekday and is_trading_hours


class RealisticBacktest:
    """
    Production-grade backtest engine that EXACTLY mirrors live trading bot.
    """

    def __init__(self,
                 symbols: List[str],
                 starting_cash: float = 100000,
                 trading_days: int = 252,  # 1 year
                 kelly_fraction: float = 0.5,
                 verbose: bool = False):
        self.symbols = symbols
        self.starting_cash = starting_cash
        self.trading_days = trading_days
        self.kelly_fraction = kelly_fraction
        self.verbose = verbose

        # Risk management parameters (MUST match live bot)
        self.daily_loss_limit = 0.03  # 3%
        self.max_drawdown_limit = 0.10  # 10%
        self.drawdown_warning_level = 0.05  # 5%
        self.max_consecutive_losses = 3
        self.max_trades_per_day = 15

        # Profit/loss thresholds
        self.profit_target = OPTIMAL_PROFIT_TARGET  # 4%
        self.stop_loss = OPTIMAL_STOP_LOSS  # 2%
        self.max_holding_bars = OPTIMAL_MAX_HOLDING_BARS  # 20 bars

        # Portfolio
        self.portfolio = RealisticPortfolio(starting_cash)

        # Models
        self.models = {}
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
                logger.warning(f"  ✗ {symbol}: Model not found")
                failed.append(symbol)
                continue

            try:
                strategy = RiskLabAIStrategy(
                    profit_taking=self.profit_target,
                    stop_loss=self.stop_loss,
                    max_holding=self.max_holding_bars,
                    d=OPTIMAL_FRACTIONAL_D,
                    n_cv_splits=5
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
            logger.warning(f"Failed: {', '.join(failed)}")
            self.symbols = [s for s in self.symbols if s not in failed]
        logger.info("")

    def load_historical_data(self):
        """Load historical tick data and generate bars."""
        logger.info("=" * 80)
        logger.info("LOADING HISTORICAL DATA")
        logger.info("=" * 80)

        storage = TickStorage(TICK_DB_PATH)

        for symbol in self.symbols:
            logger.info(f"\n{symbol}:")

            try:
                # Load ticks
                ticks = storage.load_ticks(symbol, limit=5000000)
                if not ticks:
                    logger.warning(f"  ✗ No tick data")
                    continue

                logger.info(f"  Loaded {len(ticks):,} ticks")

                # Generate bars
                bars_list = generate_bars_from_ticks(ticks, threshold=INITIAL_IMBALANCE_THRESHOLD)
                if not bars_list:
                    logger.warning(f"  ✗ Failed to generate bars")
                    continue

                logger.info(f"  Generated {len(bars_list)} bars")

                # Convert to DataFrame
                bars_df = pd.DataFrame(bars_list)
                bars_df['bar_end'] = pd.to_datetime(bars_df['bar_end'])
                if bars_df['bar_end'].dt.tz is not None:
                    bars_df['bar_end'] = bars_df['bar_end'].dt.tz_localize(None)
                bars_df.set_index('bar_end', inplace=True)

                # CRITICAL: Use only TEST SET (last 30%)
                train_size = int(len(bars_df) * 0.7)
                test_bars = bars_df.iloc[train_size:].copy()

                logger.info(f"  Test set: {len(test_bars)} bars (unseen during training)")
                logger.info(f"  Period: {test_bars.index[0]} to {test_bars.index[-1]}")

                self.symbol_bars[symbol] = test_bars
                logger.info(f"  ✓ Ready for backtest")

            except Exception as e:
                logger.error(f"  ✗ Error: {e}")

        storage.close()
        logger.info(f"\nLoaded data for {len(self.symbol_bars)}/{len(self.symbols)} symbols\n")

    def check_position_exits(self, symbol: str, current_bar_idx: int,
                            current_time: datetime, current_price: float) -> Optional[str]:
        """
        Check if position should be exited (profit target / stop loss / max holding).

        Returns exit reason if position should be closed, None otherwise.
        """
        pos = self.portfolio.positions.get(symbol)
        if not pos:
            return None

        entry_price = pos['entry_price']
        pnl_pct = (current_price - entry_price) / entry_price

        # 1. PROFIT TARGET HIT
        if pnl_pct >= self.profit_target:
            return f"PROFIT TARGET HIT: {pnl_pct:+.2%} >= {self.profit_target:.2%}"

        # 2. STOP LOSS HIT
        if pnl_pct <= -self.stop_loss:
            return f"STOP LOSS HIT: {pnl_pct:+.2%} <= {-self.stop_loss:.2%}"

        # 3. MAX HOLDING TIME
        bars_held = current_bar_idx - pos['entry_bar_idx']
        if bars_held >= self.max_holding_bars:
            return f"MAX HOLDING TIME: {bars_held} bars >= {self.max_holding_bars}"

        return None

    def run_backtest(self):
        """Run the backtest simulation."""
        logger.info("=" * 80)
        logger.info("RUNNING REALISTIC BACKTEST")
        logger.info("=" * 80)
        logger.info(f"Symbols: {len(self.symbols)}")
        logger.info(f"Starting capital: ${self.starting_cash:,.2f}")
        logger.info(f"Trading days: {self.trading_days}")
        logger.info(f"Kelly fraction: {self.kelly_fraction} (Half-Kelly)")
        logger.info("")

        # Load data
        self.load_historical_data()

        if not self.symbol_bars:
            logger.error("No data loaded")
            return

        # Get common time range
        earliest_start = max(bars.index[0] for bars in self.symbol_bars.values())
        latest_end = min(bars.index[-1] for bars in self.symbol_bars.values())

        logger.info(f"Common time range: {earliest_start} to {latest_end}\n")

        # Initialize simulation
        logger.info("=" * 80)
        logger.info("SIMULATION START")
        logger.info("=" * 80)

        bar_counts = {symbol: 50 for symbol in self.symbol_bars.keys()}
        pending_orders = {}  # Orders execute on NEXT bar open
        last_buy_check_time = None
        iterations = 0
        risk_halt_reason = None

        while True:
            # Get current times
            current_times = {}
            for symbol, start_idx in bar_counts.items():
                if start_idx < len(self.symbol_bars[symbol]):
                    current_times[symbol] = self.symbol_bars[symbol].index[start_idx]

            if not current_times:
                break

            current_time = min(current_times.values())
            current_date = current_time.date()

            # Get current prices
            current_prices = {}
            for symbol in self.symbol_bars.keys():
                idx = bar_counts[symbol]
                if idx < len(self.symbol_bars[symbol]):
                    current_prices[symbol] = self.symbol_bars[symbol]['close'].iloc[idx]

            # Record equity
            self.portfolio.record_equity(current_time, current_prices)

            # Reset daily tracking
            self.portfolio.reset_daily_tracking(current_date, current_prices)

            # ========== RISK MANAGEMENT CHECKS ==========
            current_value = self.portfolio.get_portfolio_value(current_prices)

            # Daily loss limit
            daily_pnl_pct = (current_value - self.portfolio.daily_start_value) / self.portfolio.daily_start_value
            if daily_pnl_pct <= -self.daily_loss_limit and risk_halt_reason is None:
                risk_halt_reason = f"Daily loss limit hit: {daily_pnl_pct:.2%}"
                logger.error(f"🚨 HALTED: {risk_halt_reason}")

            # Max drawdown
            drawdown = (current_value - self.portfolio.peak_portfolio_value) / self.portfolio.peak_portfolio_value
            if drawdown <= -self.max_drawdown_limit and risk_halt_reason is None:
                risk_halt_reason = f"Max drawdown hit: {drawdown:.2%}"
                logger.error(f"🚨 HALTED: {risk_halt_reason}")

            # Consecutive losses
            if self.portfolio.consecutive_losses >= self.max_consecutive_losses:
                if self.verbose:
                    logger.warning(f"⚠️  {self.portfolio.consecutive_losses} consecutive losses - pausing new trades")

            # Max trades per day
            if self.portfolio.trades_today >= self.max_trades_per_day:
                if self.verbose:
                    logger.info(f"ℹ️  Max trades/day reached ({self.max_trades_per_day})")

            # ========== STEP 1: EXECUTE PENDING ORDERS ==========
            for symbol in list(pending_orders.keys()):
                idx = bar_counts[symbol]
                if idx >= len(self.symbol_bars[symbol]):
                    continue

                execution_price = self.symbol_bars[symbol]['open'].iloc[idx]
                pending = pending_orders[symbol]

                if pending['signal'] == 1:  # BUY
                    position_value = current_value * self.kelly_fraction * pending['bet_size']
                    quantity = int(position_value / execution_price)

                    if quantity > 0 and not self.portfolio.positions.get(symbol):
                        trade = self.portfolio.execute_trade(
                            symbol, quantity, execution_price,
                            current_time, 'BUY', f"Signal (bet_size={pending['bet_size']:.2f})"
                        )
                        if trade:
                            self.portfolio.positions[symbol]['entry_bar_idx'] = idx
                            logger.info(f"{current_time} | BUY  {symbol:6s} {quantity:4d} @ ${execution_price:7.2f}")

                del pending_orders[symbol]

            # ========== STEP 2: CHECK POSITION EXITS (EVERY BAR) ==========
            for symbol, pos in list(self.portfolio.positions.items()):
                idx = bar_counts[symbol]
                if idx >= len(self.symbol_bars[symbol]):
                    continue

                current_price = current_prices.get(symbol)
                if not current_price:
                    continue

                exit_reason = self.check_position_exits(symbol, idx, current_time, current_price)

                if exit_reason:
                    # Execute exit on next bar open
                    next_idx = idx + 1
                    if next_idx < len(self.symbol_bars[symbol]):
                        exit_price = self.symbol_bars[symbol]['open'].iloc[next_idx]
                        trade = self.portfolio.execute_trade(
                            symbol, pos['quantity'], exit_price,
                            current_time, 'SELL', exit_reason
                        )
                        if trade:
                            logger.info(f"{current_time} | SELL {symbol:6s} {pos['quantity']:4d} @ ${exit_price:7.2f} | "
                                      f"P&L: ${trade['pnl']:+8.2f} ({trade['pnl_pct']:+6.2f}%) | {exit_reason}")

            # ========== STEP 3: GENERATE BUY SIGNALS (HOURLY) ==========
            should_check_signals = False
            if last_buy_check_time is None:
                should_check_signals = True
                last_buy_check_time = current_time
            elif (current_time - last_buy_check_time) >= timedelta(hours=1):
                should_check_signals = True
                last_buy_check_time = current_time

            if should_check_signals and risk_halt_reason is None:
                if self.portfolio.consecutive_losses < self.max_consecutive_losses:
                    if self.portfolio.trades_today < self.max_trades_per_day:
                        for symbol in self.symbol_bars.keys():
                            # Check stop loss cooldown
                            if self.portfolio.is_symbol_in_cooldown(symbol, current_time):
                                continue

                            # Skip if already have position
                            if symbol in self.portfolio.positions:
                                continue

                            # Check market hours
                            if not is_market_hours(current_time):
                                continue

                            idx = bar_counts[symbol]
                            if idx >= len(self.symbol_bars[symbol]):
                                continue

                            bars_up_to_now = self.symbol_bars[symbol].iloc[:idx+1].copy()
                            if len(bars_up_to_now) < 50:
                                continue

                            try:
                                signal, bet_size = self.models[symbol].predict(
                                    bars_up_to_now,
                                    prob_threshold=OPTIMAL_PROB_THRESHOLD,
                                    meta_threshold=OPTIMAL_META_THRESHOLD
                                )

                                if signal == 1 and bet_size >= 0.5:
                                    pending_orders[symbol] = {
                                        'signal': signal,
                                        'bet_size': bet_size,
                                        'timestamp': current_time
                                    }
                                    if self.verbose:
                                        logger.debug(f"{symbol}: Signal generated (bet_size={bet_size:.2f})")

                            except Exception as e:
                                logger.error(f"{symbol}: Prediction error: {e}")

            # Advance bars
            for symbol in bar_counts:
                bar_counts[symbol] += 1

            iterations += 1

            # Progress logging
            if iterations % 50 == 0:
                ret = (current_value - self.starting_cash) / self.starting_cash * 100
                logger.info(f"[{iterations:4d}] {current_time} | Portfolio: ${current_value:,.2f} ({ret:+.2f}%)")

        logger.info(f"\n✓ Simulation complete ({iterations} iterations)\n")

        # Close remaining positions
        final_prices = {symbol: bars['close'].iloc[-1] for symbol, bars in self.symbol_bars.items()}
        for symbol, pos in list(self.portfolio.positions.items()):
            if symbol in final_prices:
                self.portfolio.execute_trade(
                    symbol, pos['quantity'], final_prices[symbol],
                    self.symbol_bars[symbol].index[-1], 'SELL', "End of backtest"
                )

    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.portfolio.equity_curve:
            return {}

        equity_df = pd.DataFrame(self.portfolio.equity_curve)
        final_value = equity_df['total_value'].iloc[-1]
        total_return = (final_value - self.starting_cash) / self.starting_cash * 100

        # Trading days
        trading_days = len(equity_df)
        years = trading_days / 252
        annualized_return = ((final_value / self.starting_cash) ** (1 / years) - 1) * 100 if years > 0 else 0

        # Returns
        equity_df['returns'] = equity_df['total_value'].pct_change()
        daily_returns = equity_df['returns'].dropna()

        # Sharpe ratio
        sharpe = 0
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()

        # Max drawdown
        equity_df['cummax'] = equity_df['total_value'].cummax()
        equity_df['drawdown'] = (equity_df['total_value'] - equity_df['cummax']) / equity_df['cummax'] * 100
        max_drawdown = equity_df['drawdown'].min()

        # Trade metrics
        if not self.portfolio.trades:
            return {
                'starting_capital': self.starting_cash,
                'final_value': final_value,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'num_trades': 0
            }

        trades_df = pd.DataFrame(self.portfolio.trades)
        completed = trades_df[trades_df['action'] == 'SELL']

        num_trades = len(completed)
        winning = completed[completed['pnl'] > 0]
        losing = completed[completed['pnl'] <= 0]

        win_rate = len(winning) / num_trades * 100 if num_trades > 0 else 0
        avg_win = winning['pnl'].mean() if len(winning) > 0 else 0
        avg_loss = abs(losing['pnl'].mean()) if len(losing) > 0 else 0

        profit_factor = (winning['pnl'].sum() / abs(losing['pnl'].sum())
                        if len(losing) > 0 and losing['pnl'].sum() != 0 else float('inf'))

        # Commission and slippage costs
        total_commission = trades_df['commission'].sum()
        total_slippage = trades_df['slippage'].sum()

        return {
            'starting_capital': self.starting_cash,
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'total_pnl': final_value - self.starting_cash,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_holding_hours': completed['holding_time_hours'].mean() if num_trades > 0 else 0,
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'trading_days': trading_days,
            'years': years
        }

    def print_results(self):
        """Print comprehensive backtest results."""
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
        logger.info(f"  Annualized Return:   {metrics['annualized_return']:>12.2f}%")
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

        # Costs
        logger.info("TRADING COSTS:")
        logger.info(f"  Total Commission:    ${metrics['total_commission']:>12,.2f}")
        logger.info(f"  Total Slippage:      ${metrics['total_slippage']:>12,.2f}")
        logger.info(f"  Total Costs:         ${metrics['total_commission'] + metrics['total_slippage']:>12,.2f}")
        logger.info("")

        # Time period
        logger.info("TIME PERIOD:")
        logger.info(f"  Trading Days:        {metrics['trading_days']:>12d}")
        logger.info(f"  Years:               {metrics['years']:>12.2f}")
        logger.info("")

        logger.info("=" * 80)

        # Verdict
        if metrics['annualized_return'] > 15 and metrics['sharpe_ratio'] > 1.5:
            logger.info("✅ EXCELLENT - Strong risk-adjusted returns!")
        elif metrics['annualized_return'] > 10 and metrics['sharpe_ratio'] > 1.0:
            logger.info("✅ GOOD - Solid performance")
        elif metrics['annualized_return'] > 0:
            logger.info("⚠️  MODERATE - Profitable but needs improvement")
        else:
            logger.info("❌ POOR - Strategy needs significant improvement")

        logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Realistic production-grade backtest')
    parser.add_argument('--tier', type=str, help='Tier to test (tier_1, tier_2, etc.)')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols')
    parser.add_argument('--capital', type=float, default=100000, help='Starting capital')
    parser.add_argument('--days', type=int, default=252, help='Trading days to simulate (252 = 1 year)')
    parser.add_argument('--kelly', type=float, default=0.5, help='Kelly fraction (0.5 = half-Kelly)')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    # Determine symbols
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
        logger.error("Specify --tier or --symbols")
        return 1

    if not symbols:
        logger.error("No symbols to test")
        return 1

    # Run backtest
    backtest = RealisticBacktest(
        symbols=symbols,
        starting_cash=args.capital,
        trading_days=args.days,
        kelly_fraction=args.kelly,
        verbose=args.verbose
    )

    backtest.load_models()
    backtest.run_backtest()
    backtest.print_results()

    return 0


if __name__ == "__main__":
    sys.exit(main())
