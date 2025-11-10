"""
Backtesting Engine
==================
Core engine for backtesting trading strategies with realistic execution modeling
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade"""
    symbol: str
    entry_date: str
    entry_price: float
    signal_type: str  # BUY or SELL
    position_size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # stop_loss, take_profit, time_exit, signal_exit
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    strategy: Optional[str] = None
    strength: Optional[float] = None
    metadata: Optional[str] = None


class BacktestEngine:
    """
    Comprehensive backtesting engine with realistic execution modeling
    """

    def __init__(self,
                 db_path: str = "/Volumes/Vault/85_assets_prediction.db",
                 initial_capital: float = 100000,
                 commission: float = 0.001,  # 0.1% per trade
                 slippage: float = 0.0005,   # 0.05% slippage
                 max_position_size: float = 0.1,  # Max 10% per position
                 max_holding_days: int = 20):
        """
        Initialize backtesting engine

        Args:
            db_path: Path to database
            initial_capital: Starting capital
            commission: Commission rate per trade
            slippage: Slippage rate per trade
            max_position_size: Maximum position size as fraction of capital
            max_holding_days: Maximum days to hold a position
        """
        self.db_path = db_path
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.max_holding_days = max_holding_days

        self.trades: List[Trade] = []
        self.portfolio_value: List[Tuple[str, float]] = []
        self.open_positions: Dict[str, Trade] = {}

    def _conn(self) -> sqlite3.Connection:
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def _get_signals(self, strategy_name: Optional[str] = None,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> pd.DataFrame:
        """Get trading signals from database"""
        conn = self._conn()

        query = """
            SELECT
                strategy_name,
                symbol_ticker,
                signal_date,
                signal_type,
                strength,
                entry_price,
                stop_loss,
                take_profit,
                metadata
            FROM trading_signals
            WHERE 1=1
        """

        params = []
        if strategy_name:
            query += " AND strategy_name = ?"
            params.append(strategy_name)
        if start_date:
            query += " AND signal_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND signal_date <= ?"
            params.append(end_date)

        query += " ORDER BY signal_date, symbol_ticker"

        df = pd.read_sql(query, conn, params=params)
        conn.close()

        return df

    def _get_price_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get price data for a symbol"""
        conn = self._conn()

        query = """
            SELECT
                price_date,
                open,
                high,
                low,
                close,
                volume
            FROM raw_price_data
            WHERE symbol_ticker = ?
                AND price_date >= ?
                AND price_date <= ?
            ORDER BY price_date
        """

        df = pd.read_sql(query, conn, params=(symbol, start_date, end_date))
        conn.close()

        return df

    def _calculate_position_size(self, signal_strength: float,
                                 current_capital: float,
                                 price: float) -> float:
        """
        Calculate position size based on signal strength and risk management

        Kelly Criterion inspired: size = strength * max_position_size
        """
        # Adjust position size by signal strength
        position_value = current_capital * self.max_position_size * signal_strength

        # Convert to number of shares (rounded down)
        shares = int(position_value / price)

        return shares

    def _apply_execution_costs(self, price: float, signal_type: str) -> float:
        """Apply slippage and get execution price"""
        if signal_type == 'BUY':
            # Buy at slightly higher price (slippage)
            execution_price = price * (1 + self.slippage)
        else:  # SELL
            # Sell at slightly lower price (slippage)
            execution_price = price * (1 - self.slippage)

        return execution_price

    def _check_exit_conditions(self, trade: Trade, current_date: str,
                               current_price: float) -> Tuple[bool, Optional[str]]:
        """
        Check if trade should be exited

        Returns: (should_exit, exit_reason)
        """
        # Check stop loss
        if trade.stop_loss:
            if trade.signal_type == 'BUY' and current_price <= trade.stop_loss:
                return True, 'stop_loss'
            elif trade.signal_type == 'SELL' and current_price >= trade.stop_loss:
                return True, 'stop_loss'

        # Check take profit
        if trade.take_profit:
            if trade.signal_type == 'BUY' and current_price >= trade.take_profit:
                return True, 'take_profit'
            elif trade.signal_type == 'SELL' and current_price <= trade.take_profit:
                return True, 'take_profit'

        # Check maximum holding period
        entry_dt = datetime.strptime(trade.entry_date, '%Y-%m-%d')
        current_dt = datetime.strptime(current_date, '%Y-%m-%d')
        days_held = (current_dt - entry_dt).days

        if days_held >= self.max_holding_days:
            return True, 'time_exit'

        return False, None

    def _close_position(self, trade: Trade, exit_date: str,
                        exit_price: float, exit_reason: str) -> Trade:
        """Close a position and calculate P&L"""
        # Apply execution costs
        execution_price = self._apply_execution_costs(exit_price,
                                                      'SELL' if trade.signal_type == 'BUY' else 'BUY')

        # Calculate P&L
        if trade.signal_type == 'BUY':
            pnl = (execution_price - trade.entry_price) * trade.position_size
        else:  # SELL (short)
            pnl = (trade.entry_price - execution_price) * trade.position_size

        # Subtract commissions (both entry and exit)
        commission_cost = (trade.entry_price + execution_price) * trade.position_size * self.commission
        pnl -= commission_cost

        # Calculate percentage return
        pnl_pct = pnl / (trade.entry_price * trade.position_size)

        # Update trade
        trade.exit_date = exit_date
        trade.exit_price = execution_price
        trade.exit_reason = exit_reason
        trade.pnl = pnl
        trade.pnl_pct = pnl_pct

        return trade

    def run_backtest(self, strategy_name: Optional[str] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> Dict:
        """
        Run backtest for a strategy

        Args:
            strategy_name: Strategy to backtest (None = all strategies)
            start_date: Start date for backtest
            end_date: End date for backtest

        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest for {strategy_name or 'all strategies'}")

        # Get signals
        signals = self._get_signals(strategy_name, start_date, end_date)

        if signals.empty:
            logger.warning("No signals found for backtest")
            return {}

        logger.info(f"Found {len(signals)} signals to backtest")

        # Initialize
        self.trades = []
        self.open_positions = {}
        current_capital = self.initial_capital

        # Get unique dates
        all_dates = sorted(signals['signal_date'].unique())

        # Process each date
        for current_date in all_dates:
            # Get signals for this date
            daily_signals = signals[signals['signal_date'] == current_date]

            # First, check exit conditions for open positions
            positions_to_close = []

            for symbol, trade in self.open_positions.items():
                # Get price data for this position
                price_data = self._get_price_data(symbol, current_date, current_date)

                if price_data.empty:
                    continue

                current_price = price_data.iloc[0]['close']

                # Check if should exit
                should_exit, exit_reason = self._check_exit_conditions(
                    trade, current_date, current_price
                )

                if should_exit:
                    # Close position
                    closed_trade = self._close_position(
                        trade, current_date, current_price, exit_reason
                    )
                    self.trades.append(closed_trade)
                    positions_to_close.append(symbol)

                    # Return capital
                    current_capital += closed_trade.pnl + (closed_trade.entry_price * closed_trade.position_size)

            # Remove closed positions
            for symbol in positions_to_close:
                del self.open_positions[symbol]

            # Process new signals
            for _, signal in daily_signals.iterrows():
                symbol = signal['symbol_ticker']

                # Skip if already have position in this symbol
                if symbol in self.open_positions:
                    continue

                # Get price data
                price_data = self._get_price_data(symbol, current_date, current_date)

                if price_data.empty:
                    continue

                entry_price = price_data.iloc[0]['close']

                # Apply execution costs
                execution_price = self._apply_execution_costs(entry_price, signal['signal_type'])

                # Calculate position size
                position_size = self._calculate_position_size(
                    signal['strength'], current_capital, execution_price
                )

                if position_size == 0:
                    continue

                # Calculate capital needed
                capital_needed = execution_price * position_size

                # Add commission
                commission_cost = capital_needed * self.commission
                total_cost = capital_needed + commission_cost

                # Check if we have enough capital
                if total_cost > current_capital:
                    continue

                # Create trade
                trade = Trade(
                    symbol=symbol,
                    entry_date=current_date,
                    entry_price=execution_price,
                    signal_type=signal['signal_type'],
                    position_size=position_size,
                    stop_loss=signal['stop_loss'],
                    take_profit=signal['take_profit'],
                    strategy=signal['strategy_name'],
                    strength=signal['strength'],
                    metadata=signal['metadata']
                )

                # Open position
                self.open_positions[symbol] = trade
                current_capital -= total_cost

            # Track portfolio value
            portfolio_value = current_capital
            for trade in self.open_positions.values():
                portfolio_value += trade.entry_price * trade.position_size

            self.portfolio_value.append((current_date, portfolio_value))

        # Close any remaining open positions at end date
        if end_date and self.open_positions:
            for symbol, trade in list(self.open_positions.items()):
                price_data = self._get_price_data(symbol, end_date, end_date)

                if not price_data.empty:
                    exit_price = price_data.iloc[0]['close']
                    closed_trade = self._close_position(
                        trade, end_date, exit_price, 'backtest_end'
                    )
                    self.trades.append(closed_trade)

        logger.info(f"Backtest complete: {len(self.trades)} trades executed")

        return {
            'trades': self.trades,
            'portfolio_value': self.portfolio_value,
            'initial_capital': self.initial_capital
        }

    def get_trades_df(self) -> pd.DataFrame:
        """Convert trades to DataFrame"""
        if not self.trades:
            return pd.DataFrame()

        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'symbol': trade.symbol,
                'strategy': trade.strategy,
                'entry_date': trade.entry_date,
                'exit_date': trade.exit_date,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'signal_type': trade.signal_type,
                'position_size': trade.position_size,
                'stop_loss': trade.stop_loss,
                'take_profit': trade.take_profit,
                'exit_reason': trade.exit_reason,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'strength': trade.strength,
                'metadata': trade.metadata
            })

        return pd.DataFrame(trades_data)
