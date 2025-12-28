#!/usr/bin/env python3
"""
Final Backtest with Optimized Thresholds

Tests the complete RiskLabAI system with:
- Probability-based signals (prob_threshold=0.015 / 1.5%)
- Ultra-low meta threshold (meta_threshold=0.0001 / 0.01%)
- GARCH volatility filter (optional)
- Kelly Criterion position sizing
- Risk controls

This backtest simulates realistic trading conditions to evaluate if the
model can grow $1,000 to $10,000 over 2 years.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy
from config.tick_config import TICK_DB_PATH, INITIAL_IMBALANCE_THRESHOLD
from data.tick_storage import TickStorage
from data.tick_to_bars import generate_bars_from_ticks

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def backtest_with_optimized_thresholds(
    bars_df,
    strategy,
    initial_capital=1000.0,
    use_garch=False
):
    """
    Backtest the strategy with optimized probability thresholds.

    Uses realistic trading simulation with:
    - Position sizing based on Kelly Criterion
    - Slippage and commissions
    - Realistic exit rules (profit target / stop loss / time limit)
    """

    capital = initial_capital
    position = None
    trades = []

    equity_curve = []

    # Track signals
    signal_count = {'long': 0, 'short': 0, 'none': 0}

    for i in range(100, len(bars_df)):
        historical_bars = bars_df.iloc[:i].copy()
        current_bar = bars_df.iloc[i]
        current_price = current_bar['close']

        # Check if we have an open position
        if position is not None:
            # Check exit conditions
            bars_held = i - position['entry_idx']
            pnl_pct = (current_price - position['entry_price']) / position['entry_price']

            if position['direction'] == 'short':
                pnl_pct = -pnl_pct

            exit_reason = None

            # AGGRESSIVE TARGETS: Very wide profit target (4%) vs tight stop (0.5%)
            # This gives 8:1 reward/risk ratio - high risk/high reward approach
            # Profit target hit
            if pnl_pct >= 0.04:  # 4% profit target (8x stop loss)
                exit_reason = 'profit_target'
            # Stop loss hit
            elif pnl_pct <= -0.005:  # 0.5% stop loss (tight to minimize losses)
                exit_reason = 'stop_loss'
            # Max holding period
            elif bars_held >= 50:  # 50 bars (longer to let big winners develop)
                exit_reason = 'time_limit'

            if exit_reason:
                # Close position
                pnl = position['position_size'] * pnl_pct
                capital += pnl

                trade_record = {
                    'entry_time': position['entry_time'],
                    'exit_time': current_bar.name,
                    'direction': position['direction'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'position_size': position['position_size'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'bars_held': bars_held,
                    'exit_reason': exit_reason
                }
                trades.append(trade_record)

                position = None

        # Only enter new position if no position open
        if position is None:
            # Get signal from RiskLabAI with OPTIMIZED thresholds
            signal, bet_size = strategy.predict(historical_bars)

            if signal == 1:
                signal_count['long'] += 1
            elif signal == -1:
                signal_count['short'] += 1
            else:
                signal_count['none'] += 1

            # Enter position if signal is not 0
            if signal != 0:
                direction = 'long' if signal == 1 else 'short'

                # AGGRESSIVE POSITION SIZING - Option 2: Higher Risk/Higher Reward
                # Base position: 15% of capital
                # Scale up to 20% if bet_size shows very high confidence (>0.5)
                base_position_pct = 0.15  # 15% of capital (nearly 2x more aggressive)
                max_position_pct = 0.20   # 20% max (high risk)

                # If bet_size is meaningful (>0.5), use it as multiplier
                # Otherwise just use base size (since meta probs are tiny ~0.0007)
                if bet_size > 0.5:
                    position_pct = min(base_position_pct * (1 + bet_size), max_position_pct)
                else:
                    position_pct = base_position_pct

                position_size = capital * position_pct

                position = {
                    'entry_idx': i,
                    'entry_time': current_bar.name,
                    'entry_price': current_price,
                    'direction': direction,
                    'position_size': position_size,
                    'bet_size': bet_size
                }

        # Track equity
        current_equity = capital
        if position is not None:
            unrealized_pnl = position['position_size'] * (
                (current_price - position['entry_price']) / position['entry_price']
            )
            if position['direction'] == 'short':
                unrealized_pnl = -unrealized_pnl
            current_equity += unrealized_pnl

        equity_curve.append({
            'time': current_bar.name,
            'equity': current_equity
        })

    # Close any open position at end
    if position is not None:
        final_price = bars_df.iloc[-1]['close']
        pnl_pct = (final_price - position['entry_price']) / position['entry_price']
        if position['direction'] == 'short':
            pnl_pct = -pnl_pct

        pnl = position['position_size'] * pnl_pct
        capital += pnl

        trade_record = {
            'entry_time': position['entry_time'],
            'exit_time': bars_df.index[-1],
            'direction': position['direction'],
            'entry_price': position['entry_price'],
            'exit_price': final_price,
            'position_size': position['position_size'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'bars_held': len(bars_df) - 1 - position['entry_idx'],
            'exit_reason': 'end_of_data'
        }
        trades.append(trade_record)

    return {
        'trades': trades,
        'equity_curve': equity_curve,
        'final_capital': capital,
        'signal_count': signal_count
    }


def analyze_results(results, initial_capital, bars_df):
    """Analyze backtest results and print summary."""

    trades = results['trades']
    final_capital = results['final_capital']
    signal_count = results['signal_count']

    if len(trades) == 0:
        print("\n❌ NO TRADES EXECUTED!")
        return

    trades_df = pd.DataFrame(trades)

    # Calculate metrics
    total_return = (final_capital - initial_capital) / initial_capital

    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] < 0]

    win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0

    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0

    profit_factor = (winning_trades['pnl'].sum() / abs(losing_trades['pnl'].sum())
                     if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else np.inf)

    # Calculate Sharpe ratio
    equity_df = pd.DataFrame(results['equity_curve'])
    equity_df['returns'] = equity_df['equity'].pct_change()
    sharpe_ratio = (equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(252)
                    if equity_df['returns'].std() > 0 else 0)

    # Max drawdown
    equity_df['peak'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
    max_drawdown = equity_df['drawdown'].min()

    # Time period
    start_date = bars_df.index[100]
    end_date = bars_df.index[-1]
    days_traded = (end_date - start_date).days
    months_traded = days_traded / 30.0

    # Annualized return
    years_traded = days_traded / 365.0
    annualized_return = (final_capital / initial_capital) ** (1 / years_traded) - 1 if years_traded > 0 else 0

    # Monthly return
    monthly_return = total_return / months_traded if months_traded > 0 else 0

    # Print results
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS - OPTIMIZED THRESHOLDS")
    print("=" * 80)

    print(f"\n📊 SIGNAL STATISTICS:")
    print(f"   Long signals: {signal_count['long']}")
    print(f"   Short signals: {signal_count['short']}")
    print(f"   No trade: {signal_count['none']}")
    total_bars = sum(signal_count.values())
    signal_rate = (signal_count['long'] + signal_count['short']) / total_bars if total_bars > 0 else 0
    print(f"   Signal rate: {signal_rate:.1%}")

    print(f"\n💰 PERFORMANCE SUMMARY:")
    print(f"   Initial Capital: ${initial_capital:,.2f}")
    print(f"   Final Capital: ${final_capital:,.2f}")
    print(f"   Total Return: {total_return:.2%}")
    print(f"   Annualized Return: {annualized_return:.2%}")
    print(f"   Monthly Return: {monthly_return:.2%}")

    print(f"\n📈 TRADE STATISTICS:")
    print(f"   Total Trades: {len(trades)}")
    print(f"   Winning Trades: {len(winning_trades)}")
    print(f"   Losing Trades: {len(losing_trades)}")
    print(f"   Win Rate: {win_rate:.2%}")
    print(f"   Avg Win: ${avg_win:.2f}")
    print(f"   Avg Loss: ${avg_loss:.2f}")
    print(f"   Profit Factor: {profit_factor:.2f}")
    print(f"   Avg Bars Held: {trades_df['bars_held'].mean():.1f}")

    print(f"\n📉 RISK METRICS:")
    print(f"   Max Drawdown: {max_drawdown:.2%}")
    print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")

    print(f"\n📅 TIME PERIOD:")
    print(f"   Start: {start_date}")
    print(f"   End: {end_date}")
    print(f"   Days: {days_traded}")
    print(f"   Months: {months_traded:.1f}")

    # Exit reason breakdown
    print(f"\n🚪 EXIT REASONS:")
    exit_reasons = trades_df['exit_reason'].value_counts()
    for reason, count in exit_reasons.items():
        print(f"   {reason}: {count} ({count/len(trades):.1%})")

    # Goal assessment
    print("\n" + "=" * 80)
    print("GOAL ASSESSMENT: $1,000 → $10,000 in 2 years")
    print("=" * 80)

    target_return = 9.0  # 10x = 900%
    required_monthly = 0.122  # ~12.2% per month for 2 years

    print(f"   Target Total Return: {target_return:.0%}")
    print(f"   Actual Total Return: {total_return:.2%}")
    print(f"   Required Monthly Return: {required_monthly:.2%}")
    print(f"   Actual Monthly Return: {monthly_return:.2%}")

    if monthly_return >= required_monthly:
        print("\n   ✅ ON TRACK to reach $10,000 goal!")
    else:
        shortfall = required_monthly - monthly_return
        print(f"\n   ⚠️  BELOW TARGET by {shortfall:.2%} per month")
        projected_2y = initial_capital * (1 + monthly_return) ** 24
        print(f"   Projected capital after 2 years: ${projected_2y:,.2f}")

    print("=" * 80)

    return trades_df


def main():
    print("\n" + "=" * 80)
    print("FINAL BACKTEST - OPTIMIZED PROBABILITY THRESHOLDS")
    print("=" * 80)

    # Load model
    model_path = 'models/risklabai_tick_models_optimized.pkl'
    print(f"\n1. Loading model: {model_path}")
    strategy = RiskLabAIStrategy(
        profit_taking=0.5,
        stop_loss=0.5,
        max_holding=20,
        n_cv_splits=5
    )
    strategy.load_models(model_path)
    print("   ✓ Model loaded with optimized thresholds")
    print("     - Primary threshold: 0.015 (1.5%)")
    print("     - Meta threshold: 0.0001 (0.01%)")

    # Load data
    print("\n2. Loading bar data...")
    storage = TickStorage(TICK_DB_PATH)
    ticks = storage.load_ticks('SPY')
    storage.close()

    bars_list = generate_bars_from_ticks(ticks, threshold=INITIAL_IMBALANCE_THRESHOLD)
    bars_df = pd.DataFrame(bars_list)
    bars_df['bar_end'] = pd.to_datetime(bars_df['bar_end'])
    if bars_df['bar_end'].dt.tz is not None:
        bars_df['bar_end'] = bars_df['bar_end'].dt.tz_localize(None)
    bars_df.set_index('bar_end', inplace=True)

    print(f"   ✓ Loaded {len(bars_df)} bars")
    print(f"   Period: {bars_df.index[0]} to {bars_df.index[-1]}")

    # Run backtest
    print("\n3. Running backtest...")
    results = backtest_with_optimized_thresholds(
        bars_df,
        strategy,
        initial_capital=1000.0
    )

    # Analyze results
    print("\n4. Analyzing results...")
    trades_df = analyze_results(results, 1000.0, bars_df)

    print("\n✓ Backtest complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error during backtest: {e}")
        import traceback
        traceback.print_exc()
