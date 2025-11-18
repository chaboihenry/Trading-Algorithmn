"""
Trade Ranking System Using Kelly Criterion

Ranks trading signals by composite score:
- Risk-adjusted returns
- Signal confidence
- Kelly Criterion position sizing
- Liquidity considerations

Selects top 5 trades with optimal position sizing.
"""

import numpy as np
import pandas as pd
import sqlite3
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta


class TradeRanker:
    """Rank and select top trades using Kelly Criterion"""

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db",
                 kelly_fraction: float = 0.25,
                 max_position_size: float = 0.10,
                 min_liquidity: float = 1_000_000):
        """
        Args:
            db_path: Path to database
            kelly_fraction: Fractional Kelly multiplier (default 25%)
            max_position_size: Maximum position size as fraction of capital (default 10%)
            min_liquidity: Minimum daily volume in dollars (default $1M)
        """
        self.db_path = db_path
        self.kelly_fraction = kelly_fraction
        self.max_position_size = max_position_size
        self.min_liquidity = min_liquidity

    def calculate_kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly Criterion fraction

        Kelly% = (p * b - q) / b
        where:
          p = win rate
          q = 1 - p (loss rate)
          b = avg_win / avg_loss (win/loss ratio)

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning return
            avg_loss: Average losing return (absolute value)

        Returns:
            Optimal Kelly fraction (0-1)
        """
        if avg_loss == 0 or win_rate == 0 or win_rate == 1:
            return 0.0

        loss_rate = 1 - win_rate
        win_loss_ratio = avg_win / avg_loss

        kelly = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio

        # Kelly can be negative (don't take trade) or > 1 (leverage)
        # Clamp to [0, 1] for safety
        kelly = max(0.0, min(1.0, kelly))

        return kelly

    def get_strategy_performance(self, strategy_name: str,
                                 lookback_days: int = 90) -> Dict[str, float]:
        """
        Get historical performance metrics for a strategy

        Args:
            strategy_name: Name of strategy
            lookback_days: Days of historical data to analyze

        Returns:
            Dictionary with win_rate, avg_win, avg_loss, sharpe_ratio
        """
        conn = sqlite3.connect(self.db_path)

        # Get historical signals and outcomes
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

        query = """
        SELECT
            s.signal_type as signal,
            s.strength as signal_strength,
            p1.close as entry_price,
            p2.close as exit_price
        FROM trading_signals s
        JOIN raw_price_data p1 ON s.symbol_ticker = p1.symbol_ticker AND s.signal_date = p1.price_date
        JOIN raw_price_data p2 ON s.symbol_ticker = p2.symbol_ticker
            AND DATE(p2.price_date) = DATE(s.signal_date, '+1 day')
        WHERE s.strategy_name = ?
          AND s.signal_date >= ?
          AND s.signal_type != 'HOLD'
        """

        df = pd.read_sql_query(query, conn, params=[strategy_name, cutoff_date])
        conn.close()

        # Convert BUY/SELL to numeric
        if len(df) > 0:
            df['signal_numeric'] = df['signal'].map({'BUY': 1, 'SELL': -1, 'HOLD': 0})

        if len(df) == 0:
            # No historical data - use conservative defaults
            return {
                'win_rate': 0.50,
                'avg_win': 0.02,
                'avg_loss': 0.02,
                'sharpe_ratio': 0.0,
                'num_trades': 0
            }

        # Calculate returns
        df['return'] = ((df['exit_price'] - df['entry_price']) / df['entry_price'])

        # Adjust for signal direction (short positions)
        df.loc[df['signal_numeric'] < 0, 'return'] = -df.loc[df['signal_numeric'] < 0, 'return']

        # Calculate metrics
        returns = df['return'].dropna()

        if len(returns) == 0:
            return {
                'win_rate': 0.50,
                'avg_win': 0.02,
                'avg_loss': 0.02,
                'sharpe_ratio': 0.0,
                'num_trades': 0
            }

        wins = returns[returns > 0]
        losses = returns[returns < 0]

        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0.50
        avg_win = wins.mean() if len(wins) > 0 else 0.02
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.02

        # Calculate Sharpe ratio
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe,
            'num_trades': len(returns)
        }

    def score_signal(self, row: pd.Series, strategy_perf: Dict[str, float],
                     market_volatility: float = 0.15) -> float:
        """
        Calculate composite score for a trading signal

        Score = (Kelly% * Signal_Strength * Sharpe_Adjustment * Liquidity_Factor)

        Args:
            row: Signal row from database
            strategy_perf: Historical strategy performance metrics
            market_volatility: Current market volatility (VIX/100)

        Returns:
            Composite score (higher = better)
        """
        # Calculate Kelly fraction from historical performance
        kelly = self.calculate_kelly_fraction(
            strategy_perf['win_rate'],
            strategy_perf['avg_win'],
            strategy_perf['avg_loss']
        )

        # Apply fractional Kelly
        kelly_adjusted = kelly * self.kelly_fraction

        # Signal strength (0-1, provided by strategy)
        signal_strength = abs(row.get('signal_strength', 0.5))

        # Sharpe adjustment (prefer strategies with higher Sharpe)
        sharpe_adjustment = max(0.5, min(2.0, 1 + strategy_perf['sharpe_ratio'] / 2))

        # Liquidity factor (penalize illiquid stocks)
        daily_volume_dollars = row.get('volume', 0) * row.get('close', 0)
        if daily_volume_dollars >= self.min_liquidity:
            liquidity_factor = 1.0
        elif daily_volume_dollars >= self.min_liquidity * 0.5:
            liquidity_factor = 0.8
        else:
            liquidity_factor = 0.5

        # Volatility adjustment (reduce size in high volatility)
        vol_adjustment = 1.0 / (1.0 + market_volatility)

        # Composite score
        score = kelly_adjusted * signal_strength * sharpe_adjustment * liquidity_factor * vol_adjustment

        return score

    def calculate_position_size(self, score: float, capital: float,
                                current_price: float) -> Tuple[float, int]:
        """
        Calculate position size from score

        Args:
            score: Composite score
            capital: Available capital
            current_price: Current stock price

        Returns:
            Tuple of (position_size_fraction, num_shares)
        """
        # Position size is the score, capped at max_position_size
        position_fraction = min(score, self.max_position_size)

        # Calculate number of shares
        position_value = capital * position_fraction
        num_shares = int(position_value / current_price)

        return position_fraction, num_shares

    def get_current_signals(self, date: Optional[str] = None,
                           min_signal_strength: float = 0.3) -> pd.DataFrame:
        """
        Get current trading signals from database

        Args:
            date: Date to get signals for (default: latest)
            min_signal_strength: Minimum signal strength to consider

        Returns:
            DataFrame with signals
        """
        conn = sqlite3.connect(self.db_path)

        if date is None:
            # Get latest date
            date_query = "SELECT MAX(signal_date) FROM trading_signals"
            date = pd.read_sql_query(date_query, conn).iloc[0, 0]

        # Get signals with price and volume data
        query = """
        SELECT
            s.signal_id,
            s.strategy_name,
            s.symbol_ticker as symbol,
            s.signal_date as date,
            s.signal_type as signal,
            s.strength as signal_strength,
            s.entry_price,
            s.stop_loss,
            s.take_profit,
            p.close,
            p.volume,
            p.high,
            p.low,
            COALESCE(m.volatility_30d, 0.15) as volatility_30d
        FROM trading_signals s
        JOIN raw_price_data p ON s.symbol_ticker = p.symbol_ticker AND s.signal_date = p.price_date
        LEFT JOIN ml_features m ON s.symbol_ticker = m.symbol_ticker AND s.signal_date = m.feature_date
        WHERE s.signal_date = ?
          AND s.signal_type != 'HOLD'
          AND ABS(s.strength) >= ?
        ORDER BY s.symbol_ticker
        """

        df = pd.read_sql_query(query, conn, params=[date, min_signal_strength])
        conn.close()

        df['date'] = pd.to_datetime(df['date'])

        # Convert signal to numeric for scoring
        df['signal_numeric'] = df['signal'].map({'BUY': 1, 'SELL': -1, 'HOLD': 0})

        return df

    def rank_signals(self, signals: pd.DataFrame,
                    current_vix: Optional[float] = None) -> pd.DataFrame:
        """
        Rank trading signals by composite score

        Args:
            signals: DataFrame of trading signals
            current_vix: Current VIX level (default: use 30-day volatility)

        Returns:
            Ranked signals with scores and position sizes
        """
        if len(signals) == 0:
            return signals

        # Get market volatility
        if current_vix is None:
            # Use average 30-day volatility as proxy
            current_vix = signals['volatility_30d'].median() * 100 if 'volatility_30d' in signals.columns else 15.0

        market_volatility = current_vix / 100.0

        # Calculate score for each signal
        scores = []

        for _, row in signals.iterrows():
            # Get strategy performance
            strategy_perf = self.get_strategy_performance(row['strategy_name'])

            # Calculate score
            score = self.score_signal(row, strategy_perf, market_volatility)

            scores.append({
                'symbol': row['symbol'],
                'strategy_name': row['strategy_name'],
                'signal': row['signal'],
                'signal_strength': row['signal_strength'],
                'score': score,
                'kelly_fraction': self.calculate_kelly_fraction(
                    strategy_perf['win_rate'],
                    strategy_perf['avg_win'],
                    strategy_perf['avg_loss']
                ) * self.kelly_fraction,
                'strategy_win_rate': strategy_perf['win_rate'],
                'strategy_sharpe': strategy_perf['sharpe_ratio'],
                'close': row['close'],
                'volume': row['volume']
            })

        # Create ranked DataFrame
        ranked = pd.DataFrame(scores)

        # Sort by score (descending)
        ranked = ranked.sort_values('score', ascending=False).reset_index(drop=True)

        return ranked

    def select_top_trades(self, signals: pd.DataFrame,
                         num_trades: int = 5,
                         total_capital: float = 100_000,
                         current_vix: Optional[float] = None) -> pd.DataFrame:
        """
        Select top N trades with position sizing

        Args:
            signals: DataFrame of trading signals
            num_trades: Number of trades to select (default 5)
            total_capital: Total available capital
            current_vix: Current VIX level

        Returns:
            Top N trades with position sizes and stop/take profit levels
        """
        print("\n" + "="*80)
        print("TRADE RANKING & SELECTION")
        print("="*80)

        # Rank all signals
        ranked = self.rank_signals(signals, current_vix)

        print(f"\nTotal Signals: {len(ranked)}")
        print(f"Selecting Top: {num_trades}")

        if len(ranked) == 0:
            print("\n❌ No signals to rank")
            return pd.DataFrame()

        # Select top N
        top_trades = ranked.head(num_trades).copy()

        # Calculate position sizes
        position_sizes = []
        num_shares_list = []

        for _, row in top_trades.iterrows():
            pos_fraction, num_shares = self.calculate_position_size(
                row['score'],
                total_capital,
                row['close']
            )
            position_sizes.append(pos_fraction)
            num_shares_list.append(num_shares)

        top_trades['position_size'] = position_sizes
        top_trades['num_shares'] = num_shares_list
        top_trades['position_value'] = top_trades['num_shares'] * top_trades['close']

        # Calculate stop-loss and take-profit levels (2:1 reward/risk)
        top_trades['stop_loss_pct'] = -0.02  # -2% stop loss
        top_trades['take_profit_pct'] = 0.04  # +4% take profit (2:1 ratio)

        top_trades['stop_loss_price'] = top_trades['close'] * (1 + top_trades['stop_loss_pct'])
        top_trades['take_profit_price'] = top_trades['close'] * (1 + top_trades['take_profit_pct'])

        # Adjust for short positions
        short_mask = top_trades['signal'] < 0
        top_trades.loc[short_mask, 'stop_loss_price'] = top_trades.loc[short_mask, 'close'] * (1 - top_trades.loc[short_mask, 'stop_loss_pct'])
        top_trades.loc[short_mask, 'take_profit_price'] = top_trades.loc[short_mask, 'close'] * (1 - top_trades.loc[short_mask, 'take_profit_pct'])

        # Print results
        print(f"\n{'─'*80}")
        print("TOP TRADES")
        print(f"{'─'*80}")

        for i, row in top_trades.iterrows():
            direction = "LONG" if row['signal'] > 0 else "SHORT"
            print(f"\n#{i+1} {row['symbol']} ({direction}) - {row['strategy_name']}")
            print(f"  Score:          {row['score']:.4f}")
            print(f"  Signal Strength: {row['signal_strength']:.2f}")
            print(f"  Position Size:   {row['position_size']:.2%} (${row['position_value']:,.0f})")
            print(f"  Shares:          {row['num_shares']:,}")
            print(f"  Entry Price:     ${row['close']:.2f}")
            print(f"  Stop Loss:       ${row['stop_loss_price']:.2f} ({row['stop_loss_pct']:.1%})")
            print(f"  Take Profit:     ${row['take_profit_price']:.2f} ({row['take_profit_pct']:.1%})")
            print(f"  Kelly Fraction:  {row['kelly_fraction']:.2%}")
            print(f"  Strategy Stats:  WR={row['strategy_win_rate']:.1%}, Sharpe={row['strategy_sharpe']:.2f}")

        # Summary
        total_allocation = top_trades['position_size'].sum()
        print(f"\n{'─'*80}")
        print(f"Total Capital Allocated: {total_allocation:.2%} (${total_allocation * total_capital:,.0f})")
        print(f"Cash Remaining: {1 - total_allocation:.2%} (${(1 - total_allocation) * total_capital:,.0f})")
        print(f"{'='*80}\n")

        return top_trades

    def get_current_top_trades(self, num_trades: int = 5,
                               total_capital: float = 100_000,
                               date: Optional[str] = None,
                               strategy_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Get current top trades (convenience method)

        Args:
            num_trades: Number of trades to select
            total_capital: Total available capital
            date: Date to get signals for (default: latest)
            strategy_filter: Filter by strategy name (e.g., 'EnsembleStrategy')

        Returns:
            Top N trades with full details
        """
        # Get current signals
        signals = self.get_current_signals(date=date)

        if len(signals) == 0:
            print("\n❌ No current signals found")
            return pd.DataFrame()

        # Filter by strategy if specified
        if strategy_filter:
            signals = signals[signals['strategy_name'] == strategy_filter].copy()
            if len(signals) == 0:
                print(f"\n❌ No signals found for strategy: {strategy_filter}")
                return pd.DataFrame()

        # Select top trades
        top_trades = self.select_top_trades(signals, num_trades, total_capital)

        return top_trades

    def get_ensemble_top_trades(self, num_trades: int = 5,
                                total_capital: float = 100_000,
                                date: Optional[str] = None) -> pd.DataFrame:
        """
        Get top trades from EnsembleStrategy only

        Args:
            num_trades: Number of trades to select
            total_capital: Total available capital
            date: Date to get signals for (default: latest)

        Returns:
            Top N trades from EnsembleStrategy
        """
        return self.get_current_top_trades(
            num_trades=num_trades,
            total_capital=total_capital,
            date=date,
            strategy_filter='EnsembleStrategy'
        )

    def export_trades_to_csv(self, trades: pd.DataFrame, output_path: str):
        """
        Export top trades to CSV for execution

        Args:
            trades: Top trades DataFrame
            output_path: Path to save CSV
        """
        if len(trades) == 0:
            print("No trades to export")
            return

        # Select relevant columns for execution
        execution_df = trades[[
            'symbol', 'strategy_name', 'signal', 'signal_strength',
            'score', 'position_size', 'num_shares', 'close',
            'stop_loss_price', 'take_profit_price',
            'kelly_fraction', 'strategy_win_rate', 'strategy_sharpe'
        ]].copy()

        # Add execution instructions
        execution_df['action'] = execution_df['signal'].apply(lambda x: 'BUY' if x > 0 else 'SELL')
        execution_df['entry_price'] = execution_df['close']

        # Reorder columns
        execution_df = execution_df[[
            'symbol', 'action', 'num_shares', 'entry_price',
            'stop_loss_price', 'take_profit_price', 'position_size',
            'score', 'signal_strength', 'strategy_name',
            'kelly_fraction', 'strategy_win_rate', 'strategy_sharpe'
        ]]

        # Save to CSV
        execution_df.to_csv(output_path, index=False)
        print(f"\n✅ Trades exported to: {output_path}")
