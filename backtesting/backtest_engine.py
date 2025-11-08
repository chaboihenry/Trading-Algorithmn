"""
Backtesting Engine
==================

Tests trading strategies on historical data with realistic simulation.

Features:
- Transaction costs (0.1% per trade)
- Slippage simulation
- Position sizing rules
- Comprehensive risk metrics (Sharpe, max drawdown, win rate)
- Trade-by-trade analysis

Output: Detailed performance metrics and trade history
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestEngine:
    """Backtest trading strategies on historical data"""

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db",
                 initial_capital: float = 10000.0):
        """
        Initialize backtesting engine

        Args:
            db_path: Path to database
            initial_capital: Starting capital for backtest
        """
        self.db_path = db_path
        self.initial_capital = initial_capital

        # Trading costs
        self.transaction_cost_pct = 0.001  # 0.1% per trade
        self.slippage_pct = 0.0005  # 0.05% slippage

        # Position sizing
        self.max_position_size = 0.15  # Max 15% per position
        self.max_positions = 5  # Max 5 concurrent positions

        # Risk-free rate for Sharpe ratio (annual)
        self.risk_free_rate = 0.02  # 2%

        logger.info(f"Initialized BacktestEngine")
        logger.info(f"Initial Capital: ${initial_capital:,.2f}")
        logger.info(f"Transaction Cost: {self.transaction_cost_pct:.2%}")
        logger.info(f"Slippage: {self.slippage_pct:.2%}")

    def _get_db_connection(self) -> sqlite3.Connection:
        """Create database connection"""
        return sqlite3.connect(self.db_path)

    def _get_historical_signals(self, start_date: str, end_date: str,
                                signal_type: Optional[str] = None) -> pd.DataFrame:
        """
        Get historical trading signals

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            signal_type: Filter by signal type (PAIRS, SENTIMENT, VOLATILITY)

        Returns:
            DataFrame of trading signals
        """
        conn = self._get_db_connection()

        query = """
            SELECT
                signal_date,
                symbol_ticker_1,
                symbol_ticker_2,
                signal_type,
                signal_direction,
                confidence_score,
                z_score,
                sentiment_divergence,
                ensemble_score
            FROM trading_signals
            WHERE signal_date BETWEEN ? AND ?
        """

        params = [start_date, end_date]

        if signal_type:
            query += " AND signal_type = ?"
            params.append(signal_type)

        query += " ORDER BY signal_date ASC"

        df = pd.read_sql(query, conn, params=params)
        conn.close()

        return df

    def _get_historical_prices(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical price data for a ticker"""
        conn = self._get_db_connection()

        query = """
            SELECT price_date, open_price, close_price, high_price, low_price
            FROM price_data
            WHERE symbol_ticker = ?
              AND price_date BETWEEN ? AND ?
            ORDER BY price_date ASC
        """

        df = pd.read_sql(query, conn, params=(ticker, start_date, end_date))
        conn.close()

        return df

    def _calculate_entry_price(self, ticker: str, signal_date: str,
                               signal_direction: str) -> Optional[float]:
        """
        Calculate entry price with slippage

        Args:
            ticker: Stock ticker
            signal_date: Date of signal
            signal_direction: BUY/SELL/LONG_SPREAD/SHORT_SPREAD

        Returns:
            Entry price after slippage, or None if no data
        """
        # Get price data for signal date (or next available date)
        prices = self._get_historical_prices(
            ticker,
            signal_date,
            (pd.to_datetime(signal_date) + timedelta(days=5)).strftime('%Y-%m-%d')
        )

        if prices.empty:
            return None

        # Use open price of signal date (or next day if weekend)
        entry_price = prices.iloc[0]['open_price']

        # Apply slippage
        if 'BUY' in signal_direction or 'LONG' in signal_direction:
            # Pay more when buying
            entry_price *= (1 + self.slippage_pct)
        else:
            # Receive less when selling
            entry_price *= (1 - self.slippage_pct)

        return entry_price

    def _calculate_position_size(self, capital_available: float, price: float) -> int:
        """
        Calculate position size based on available capital

        Args:
            capital_available: Available capital
            price: Stock price

        Returns:
            Number of shares
        """
        max_investment = capital_available * self.max_position_size
        shares = int(max_investment / price)
        return max(shares, 0)

    def _simulate_trade(self, signal: pd.Series, capital: float,
                       open_positions: List[Dict]) -> Optional[Dict]:
        """
        Simulate a single trade

        Args:
            signal: Trading signal
            capital: Available capital
            open_positions: List of currently open positions

        Returns:
            Trade dict or None if trade not executed
        """
        # Check position limit
        if len(open_positions) >= self.max_positions:
            return None

        # Get entry price
        entry_price1 = self._calculate_entry_price(
            signal['symbol_ticker_1'],
            signal['signal_date'],
            signal['signal_direction']
        )

        if entry_price1 is None:
            return None

        # Calculate position size
        shares = self._calculate_position_size(capital, entry_price1)

        if shares < 1:
            return None

        # Calculate costs
        trade_value = shares * entry_price1
        transaction_cost = trade_value * self.transaction_cost_pct

        # For pairs trades, need both legs
        if signal['signal_type'] == 'PAIRS' and signal['symbol_ticker_2']:
            entry_price2 = self._calculate_entry_price(
                signal['symbol_ticker_2'],
                signal['signal_date'],
                signal['signal_direction']
            )

            if entry_price2 is None:
                return None

            # Approximate shares for second leg
            shares2 = shares  # Simplified - could use hedge ratio
            trade_value += shares2 * entry_price2
            transaction_cost += shares2 * entry_price2 * self.transaction_cost_pct

        total_cost = trade_value + transaction_cost

        if total_cost > capital:
            return None

        # Create trade record
        trade = {
            'entry_date': signal['signal_date'],
            'ticker1': signal['symbol_ticker_1'],
            'ticker2': signal['symbol_ticker_2'],
            'signal_type': signal['signal_type'],
            'direction': signal['signal_direction'],
            'entry_price1': entry_price1,
            'entry_price2': entry_price2 if signal['signal_type'] == 'PAIRS' else None,
            'shares': shares,
            'trade_value': trade_value,
            'transaction_cost': transaction_cost,
            'total_cost': total_cost,
            'confidence': signal['confidence_score'],
            'status': 'OPEN'
        }

        return trade

    def _close_position(self, position: Dict, exit_date: str,
                       exit_reason: str = 'Signal Exit') -> Dict:
        """
        Close an open position

        Args:
            position: Open position dict
            exit_date: Date to close position
            exit_reason: Reason for exit

        Returns:
            Updated position dict with P&L
        """
        # Get exit prices
        exit_price1 = self._calculate_entry_price(
            position['ticker1'],
            exit_date,
            'SELL'  # Opposite of entry
        )

        if exit_price1 is None:
            # Use last known entry price if exit price unavailable
            exit_price1 = position['entry_price1']

        # Calculate P&L for first leg
        if 'BUY' in position['direction'] or 'LONG' in position['direction']:
            pnl1 = (exit_price1 - position['entry_price1']) * position['shares']
        else:
            pnl1 = (position['entry_price1'] - exit_price1) * position['shares']

        pnl = pnl1

        # Handle pairs trade
        if position['signal_type'] == 'PAIRS' and position['ticker2']:
            exit_price2 = self._calculate_entry_price(
                position['ticker2'],
                exit_date,
                'BUY'
            )

            if exit_price2 is None:
                exit_price2 = position['entry_price2']

            if 'LONG_SPREAD' in position['direction']:
                pnl2 = -(exit_price2 - position['entry_price2']) * position['shares']
            else:
                pnl2 = (exit_price2 - position['entry_price2']) * position['shares']

            pnl += pnl2
            position['exit_price2'] = exit_price2

        # Calculate exit transaction costs
        exit_value = exit_price1 * position['shares']
        if position['signal_type'] == 'PAIRS':
            exit_value += exit_price2 * position['shares']

        exit_transaction_cost = exit_value * self.transaction_cost_pct

        # Net P&L
        net_pnl = pnl - exit_transaction_cost

        # Update position
        position['exit_date'] = exit_date
        position['exit_price1'] = exit_price1
        position['exit_transaction_cost'] = exit_transaction_cost
        position['gross_pnl'] = pnl
        position['net_pnl'] = net_pnl
        position['return_pct'] = (net_pnl / position['total_cost']) * 100
        position['exit_reason'] = exit_reason
        position['status'] = 'CLOSED'

        return position

    def backtest_strategy(self, start_date: str, end_date: str,
                         signal_type: Optional[str] = None) -> Dict:
        """
        Run backtest on historical data

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            signal_type: Filter by signal type

        Returns:
            Dictionary with performance metrics
        """
        logger.info("="*60)
        logger.info("BACKTEST STARTED")
        logger.info("="*60)
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Signal Type: {signal_type or 'ALL'}")
        logger.info(f"Initial Capital: ${self.initial_capital:,.2f}")

        # Get signals
        signals = self._get_historical_signals(start_date, end_date, signal_type)

        if signals.empty:
            logger.warning("No signals found for backtest period")
            return {}

        logger.info(f"Found {len(signals)} trading signals")

        # Initialize portfolio
        capital = self.initial_capital
        open_positions = []
        closed_positions = []
        equity_curve = [(start_date, capital)]

        # Process each signal
        for idx, signal in signals.iterrows():
            # Try to open new position
            trade = self._simulate_trade(signal, capital, open_positions)

            if trade:
                open_positions.append(trade)
                capital -= trade['total_cost']
                logger.info(f"Opened {trade['signal_type']} position: {trade['ticker1']} @ ${trade['entry_price1']:.2f}")

            # Check exit conditions for open positions (simplified)
            # In real backtest, would check daily for exits
            # For now, close after 20 days or at end of backtest
            positions_to_close = []
            for pos in open_positions:
                days_held = (pd.to_datetime(signal['signal_date']) - pd.to_datetime(pos['entry_date'])).days

                if days_held >= 20:  # Simplified exit rule
                    positions_to_close.append(pos)

            for pos in positions_to_close:
                closed_pos = self._close_position(pos, signal['signal_date'], 'Time Exit')
                closed_positions.append(closed_pos)
                open_positions.remove(pos)
                capital += closed_pos['trade_value'] + closed_pos['net_pnl']

            # Record equity
            equity_curve.append((signal['signal_date'], capital))

        # Close any remaining positions at end date
        for pos in open_positions:
            closed_pos = self._close_position(pos, end_date, 'Backtest End')
            closed_positions.append(closed_pos)
            capital += closed_pos['trade_value'] + closed_pos['net_pnl']

        # Calculate metrics
        metrics = self._calculate_performance_metrics(
            closed_positions, equity_curve, start_date, end_date
        )

        logger.info("="*60)
        logger.info("BACKTEST COMPLETED")
        logger.info("="*60)

        return metrics

    def _calculate_performance_metrics(self, trades: List[Dict],
                                      equity_curve: List[Tuple],
                                      start_date: str, end_date: str) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not trades:
            logger.warning("No trades to analyze")
            return {}

        trades_df = pd.DataFrame(trades)

        # Basic metrics
        total_trades = len(trades)
        winners = len(trades_df[trades_df['net_pnl'] > 0])
        losers = len(trades_df[trades_df['net_pnl'] < 0])
        win_rate = (winners / total_trades) * 100 if total_trades > 0 else 0

        total_pnl = trades_df['net_pnl'].sum()
        avg_pnl = trades_df['net_pnl'].mean()
        avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean() if winners > 0 else 0
        avg_loss = trades_df[trades_df['net_pnl'] < 0]['net_pnl'].mean() if losers > 0 else 0

        final_capital = self.initial_capital + total_pnl
        total_return = (total_pnl / self.initial_capital) * 100

        # Sharpe ratio
        returns = trades_df['return_pct'].values
        sharpe_ratio = self._calculate_sharpe_ratio(returns)

        # Max drawdown
        equity_df = pd.DataFrame(equity_curve, columns=['date', 'equity'])
        max_drawdown, max_drawdown_pct = self._calculate_max_drawdown(equity_df)

        # Profit factor
        gross_profit = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum()
        gross_loss = abs(trades_df[trades_df['net_pnl'] < 0]['net_pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        metrics = {
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'winners': winners,
            'losers': losers,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'profit_factor': profit_factor,
            'trades': trades
        }

        # Log results
        logger.info(f"Final Capital: ${final_capital:,.2f}")
        logger.info(f"Total Return: {total_return:.2f}%")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {max_drawdown_pct:.2f}%")
        logger.info(f"Profit Factor: {profit_factor:.2f}")

        return metrics

    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(returns) == 0:
            return 0.0

        mean_return = np.mean(returns) / 100  # Convert from percentage
        std_return = np.std(returns) / 100

        if std_return == 0:
            return 0.0

        # Annualize (assuming ~252 trading days)
        annual_return = mean_return * 252
        annual_std = std_return * np.sqrt(252)

        sharpe = (annual_return - self.risk_free_rate) / annual_std

        return sharpe

    def _calculate_max_drawdown(self, equity_df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate maximum drawdown"""
        equity = equity_df['equity'].values
        running_max = np.maximum.accumulate(equity)
        drawdown = equity - running_max
        max_drawdown = np.min(drawdown)
        max_drawdown_pct = (max_drawdown / running_max[np.argmin(drawdown)]) * 100

        return max_drawdown, max_drawdown_pct


if __name__ == "__main__":
    # Example usage
    engine = BacktestEngine(initial_capital=10000.0)

    # Backtest pairs trading strategy
    metrics = engine.backtest_strategy(
        start_date='2025-01-01',
        end_date='2025-10-31',
        signal_type='PAIRS'
    )

    if metrics:
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Period: {metrics['start_date']} to {metrics['end_date']}")
        print(f"Initial Capital: ${metrics['initial_capital']:,.2f}")
        print(f"Final Capital: ${metrics['final_capital']:,.2f}")
        print(f"Total Return: {metrics['total_return']:.2f}%")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print("="*60)