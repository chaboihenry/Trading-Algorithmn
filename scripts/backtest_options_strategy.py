#!/usr/bin/env python3
"""
Options Trading Strategy Using RiskLabAI Signals

Uses the same profitable RiskLabAI signals but trades options instead of stock:
- Long signals → Buy ATM/OTM call options
- Short signals → Buy ATM/OTM put options

Options provide leverage to achieve higher returns with limited capital.

Key Parameters:
- DTE (Days to Expiration): 30-45 days to balance theta decay vs time for move
- Delta: 0.5-0.7 (ATM to slightly ITM for good leverage)
- Position Size: 5-10% of capital per trade (max loss = premium)
- Target: 50-100% gain on premium, -100% max loss
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy
from config.tick_config import TICK_DB_PATH, INITIAL_IMBALANCE_THRESHOLD
from data.tick_storage import TickStorage
from data.tick_to_bars import generate_bars_from_ticks

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class SimpleOptionsCalculator:
    """Simplified options pricing for backtesting."""

    @staticmethod
    def black_scholes_call(S, K, T, r, sigma):
        """
        Black-Scholes call option pricing.

        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility (annualized)
        """
        if T <= 0:
            return max(S - K, 0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

        # Calculate delta for reference
        delta = norm.cdf(d1)

        return call_price, delta

    @staticmethod
    def black_scholes_put(S, K, T, r, sigma):
        """Black-Scholes put option pricing."""
        if T <= 0:
            return max(K - S, 0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        # Calculate delta for reference (negative for puts)
        delta = -norm.cdf(-d1)

        return put_price, delta


def estimate_volatility(prices, window=20):
    """Estimate historical volatility (annualized)."""
    returns = prices.pct_change().dropna()
    if len(returns) < window:
        return 0.20  # Default 20% volatility

    recent_returns = returns.tail(window)
    volatility = recent_returns.std() * np.sqrt(252)  # Annualize

    return max(volatility, 0.10)  # Minimum 10% vol


def backtest_options_strategy(
    bars_df,
    strategy,
    initial_capital=1000.0,
    position_size_pct=0.08,  # 8% of capital per trade
    target_dte=35,  # Days to expiration
    target_delta=0.60,  # Target delta for options
    profit_target_pct=0.75,  # Take profit at 75% gain
    max_loss_pct=-0.90,  # Stop out at -90% (near worthless)
    max_holding_days=30  # Max 30 days (before expiration)
):
    """
    Backtest options trading strategy.

    For each signal:
    - Long → Buy ATM/ITM call option
    - Short → Buy ATM/ITM put option

    Options are held until:
    - +75% profit (take profit)
    - -90% loss (cut losses)
    - 30 days (approaching expiration)
    """

    capital = initial_capital
    position = None
    trades = []
    equity_curve = []
    signal_count = {'long': 0, 'short': 0, 'none': 0}

    # Risk-free rate (approximate)
    risk_free_rate = 0.04

    calc = SimpleOptionsCalculator()

    for i in range(100, len(bars_df)):
        historical_bars = bars_df.iloc[:i].copy()
        current_bar = bars_df.iloc[i]
        current_price = current_bar['close']
        current_date = current_bar.name

        # Check existing position
        if position is not None:
            # Days held
            days_held = (current_date - position['entry_date']).days

            # Calculate current option value
            time_to_expiry = max((position['expiration_date'] - current_date).days / 365.0, 0.001)
            current_vol = estimate_volatility(bars_df.iloc[max(0, i-20):i+1]['close'])

            if position['option_type'] == 'call':
                current_option_price, _ = calc.black_scholes_call(
                    current_price,
                    position['strike'],
                    time_to_expiry,
                    risk_free_rate,
                    current_vol
                )
            else:  # put
                current_option_price, _ = calc.black_scholes_put(
                    current_price,
                    position['strike'],
                    time_to_expiry,
                    risk_free_rate,
                    current_vol
                )

            # Calculate P&L
            pnl_pct = (current_option_price - position['entry_premium']) / position['entry_premium']

            exit_reason = None

            # Exit conditions
            if pnl_pct >= profit_target_pct:
                exit_reason = 'profit_target'
            elif pnl_pct <= max_loss_pct:
                exit_reason = 'stop_loss'
            elif days_held >= max_holding_days:
                exit_reason = 'time_limit'
            elif time_to_expiry < 5/365:  # Less than 5 days to expiry
                exit_reason = 'near_expiration'

            if exit_reason:
                # Close position
                num_contracts = position['num_contracts']
                total_entry_cost = position['entry_premium'] * num_contracts * 100
                total_exit_value = current_option_price * num_contracts * 100

                pnl = total_exit_value - total_entry_cost
                capital += pnl

                trade_record = {
                    'entry_date': position['entry_date'],
                    'exit_date': current_date,
                    'option_type': position['option_type'],
                    'strike': position['strike'],
                    'entry_price': current_price,
                    'exit_price': current_price,
                    'entry_premium': position['entry_premium'],
                    'exit_premium': current_option_price,
                    'num_contracts': num_contracts,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'days_held': days_held,
                    'exit_reason': exit_reason
                }
                trades.append(trade_record)

                position = None

        # Enter new position if no position open
        if position is None:
            # Get signal from RiskLabAI
            signal, bet_size = strategy.predict(historical_bars)

            if signal == 1:
                signal_count['long'] += 1
            elif signal == -1:
                signal_count['short'] += 1
            else:
                signal_count['none'] += 1

            # Enter options position
            if signal != 0:
                # Determine strike based on target delta
                # For simplicity: ATM = current price, slightly ITM for target delta 0.6
                if signal == 1:  # Buy call
                    option_type = 'call'
                    # Slightly ITM call (strike below current price)
                    strike = current_price * 0.98  # 2% ITM
                else:  # Buy put
                    option_type = 'put'
                    # Slightly ITM put (strike above current price)
                    strike = current_price * 1.02  # 2% ITM

                # Calculate option price
                time_to_expiry = target_dte / 365.0
                current_vol = estimate_volatility(bars_df.iloc[max(0, i-20):i+1]['close'])

                if option_type == 'call':
                    option_premium, actual_delta = calc.black_scholes_call(
                        current_price,
                        strike,
                        time_to_expiry,
                        risk_free_rate,
                        current_vol
                    )
                else:
                    option_premium, actual_delta = calc.black_scholes_put(
                        current_price,
                        strike,
                        time_to_expiry,
                        risk_free_rate,
                        current_vol
                    )

                # Position sizing: allocate % of capital
                position_value = capital * position_size_pct

                # Number of contracts (each contract = 100 shares)
                contract_cost = option_premium * 100
                num_contracts = max(1, int(position_value / contract_cost))

                # Actual cost
                actual_cost = num_contracts * contract_cost

                # Only enter if we can afford at least 1 contract
                if actual_cost <= capital:
                    capital -= actual_cost

                    expiration_date = current_date + pd.Timedelta(days=target_dte)

                    position = {
                        'entry_date': current_date,
                        'expiration_date': expiration_date,
                        'option_type': option_type,
                        'strike': strike,
                        'entry_premium': option_premium,
                        'num_contracts': num_contracts,
                        'entry_underlying_price': current_price,
                        'delta': actual_delta
                    }

        # Track equity
        current_equity = capital
        if position is not None:
            # Mark to market
            time_to_expiry = max((position['expiration_date'] - current_date).days / 365.0, 0.001)
            current_vol = estimate_volatility(bars_df.iloc[max(0, i-20):i+1]['close'])

            if position['option_type'] == 'call':
                current_option_price, _ = calc.black_scholes_call(
                    current_price,
                    position['strike'],
                    time_to_expiry,
                    risk_free_rate,
                    current_vol
                )
            else:
                current_option_price, _ = calc.black_scholes_put(
                    current_price,
                    position['strike'],
                    time_to_expiry,
                    risk_free_rate,
                    current_vol
                )

            current_equity += current_option_price * position['num_contracts'] * 100

        equity_curve.append({
            'time': current_date,
            'equity': current_equity
        })

    # Close any open position at end
    if position is not None:
        final_price = bars_df.iloc[-1]['close']
        final_date = bars_df.index[-1]

        time_to_expiry = max((position['expiration_date'] - final_date).days / 365.0, 0.001)
        current_vol = estimate_volatility(bars_df.iloc[-20:]['close'])

        if position['option_type'] == 'call':
            final_option_price, _ = calc.black_scholes_call(
                final_price,
                position['strike'],
                time_to_expiry,
                risk_free_rate,
                current_vol
            )
        else:
            final_option_price, _ = calc.black_scholes_put(
                final_price,
                position['strike'],
                time_to_expiry,
                risk_free_rate,
                current_vol
            )

        num_contracts = position['num_contracts']
        total_entry_cost = position['entry_premium'] * num_contracts * 100
        total_exit_value = final_option_price * num_contracts * 100

        pnl = total_exit_value - total_entry_cost
        capital += pnl

        pnl_pct = (final_option_price - position['entry_premium']) / position['entry_premium']

        trade_record = {
            'entry_date': position['entry_date'],
            'exit_date': final_date,
            'option_type': position['option_type'],
            'strike': position['strike'],
            'entry_price': position['entry_underlying_price'],
            'exit_price': final_price,
            'entry_premium': position['entry_premium'],
            'exit_premium': final_option_price,
            'num_contracts': num_contracts,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'days_held': (final_date - position['entry_date']).days,
            'exit_reason': 'end_of_data'
        }
        trades.append(trade_record)

    return {
        'trades': trades,
        'equity_curve': equity_curve,
        'final_capital': capital,
        'signal_count': signal_count
    }


def analyze_options_results(results, initial_capital, bars_df):
    """Analyze options backtest results."""

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
    losing_trades = trades_df[trades_df['pnl'] <= 0]

    win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0

    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0

    avg_win_pct = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
    avg_loss_pct = abs(losing_trades['pnl_pct'].mean()) if len(losing_trades) > 0 else 0

    profit_factor = (winning_trades['pnl'].sum() / abs(losing_trades['pnl'].sum())
                     if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else np.inf)

    # Sharpe ratio
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
    years_traded = days_traded / 365.0

    # Annualized return
    annualized_return = (final_capital / initial_capital) ** (1 / years_traded) - 1 if years_traded > 0 else 0
    monthly_return = total_return / months_traded if months_traded > 0 else 0

    # Print results
    print("\n" + "=" * 80)
    print("OPTIONS TRADING BACKTEST RESULTS")
    print("=" * 80)

    print(f"\n📊 SIGNAL STATISTICS:")
    print(f"   Long signals (calls): {signal_count['long']}")
    print(f"   Short signals (puts): {signal_count['short']}")
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
    print(f"   Winning Trades: {len(winning_trades)} ({win_rate:.1%})")
    print(f"   Losing Trades: {len(losing_trades)} ({(1-win_rate):.1%})")
    print(f"   Avg Win: ${avg_win:.2f} ({avg_win_pct:.1%})")
    print(f"   Avg Loss: ${avg_loss:.2f} ({avg_loss_pct:.1%})")
    print(f"   Profit Factor: {profit_factor:.2f}")
    print(f"   Avg Days Held: {trades_df['days_held'].mean():.1f}")

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

    # Best/worst trades
    print(f"\n🏆 BEST TRADES:")
    best_trades = trades_df.nlargest(3, 'pnl_pct')[['entry_date', 'option_type', 'pnl', 'pnl_pct', 'days_held']]
    for idx, trade in best_trades.iterrows():
        print(f"   {trade['entry_date'].date()} {trade['option_type'].upper()}: "
              f"${trade['pnl']:.2f} ({trade['pnl_pct']:.1%}) in {trade['days_held']} days")

    print(f"\n💔 WORST TRADES:")
    worst_trades = trades_df.nsmallest(3, 'pnl_pct')[['entry_date', 'option_type', 'pnl', 'pnl_pct', 'days_held']]
    for idx, trade in worst_trades.iterrows():
        print(f"   {trade['entry_date'].date()} {trade['option_type'].upper()}: "
              f"${trade['pnl']:.2f} ({trade['pnl_pct']:.1%}) in {trade['days_held']} days")

    # Goal assessment
    print("\n" + "=" * 80)
    print("GOAL ASSESSMENT: $1,000 → $10,000 in 2 years")
    print("=" * 80)

    target_return = 9.0  # 10x = 900%
    required_monthly = 0.122  # ~12.2% per month

    print(f"   Target Total Return: {target_return:.0%}")
    print(f"   Actual Total Return: {total_return:.2%}")
    print(f"   Required Monthly Return: {required_monthly:.2%}")
    print(f"   Actual Monthly Return: {monthly_return:.2%}")

    if monthly_return >= required_monthly:
        print("\n   ✅ ON TRACK to reach $10,000 goal!")
        gap_pct = (monthly_return / required_monthly - 1) * 100
        print(f"   Exceeding target by {gap_pct:.1f}%")
    else:
        shortfall = required_monthly - monthly_return
        print(f"\n   ⚠️  BELOW TARGET by {shortfall:.2%} per month")
        projected_2y = initial_capital * (1 + monthly_return) ** 24
        print(f"   Projected capital after 2 years: ${projected_2y:,.2f}")

        # How much closer are we?
        improvement = (monthly_return / required_monthly) * 100
        print(f"   Achievement: {improvement:.1f}% of target")

    print("=" * 80)

    return trades_df


def main():
    print("\n" + "=" * 80)
    print("OPTIONS TRADING STRATEGY - LEVERAGED RISKLABAI SIGNALS")
    print("=" * 80)

    # Load model
    model_path = 'models/risklabai_tick_models_optimized.pkl'
    print(f"\n1. Loading RiskLabAI model: {model_path}")
    strategy = RiskLabAIStrategy(
        profit_taking=0.5,
        stop_loss=0.5,
        max_holding=20,
        n_cv_splits=5
    )
    strategy.load_models(model_path)
    print("   ✓ Model loaded (signals optimized for stock trading)")

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

    # Run options backtest
    print("\n3. Running options backtest...")
    print("   Strategy: Buy ATM calls/puts based on RiskLabAI signals")
    print("   Position sizing: 8% of capital per trade")
    print("   DTE: 35 days")
    print("   Profit target: +75% on premium")
    print("   Stop loss: -90% on premium")

    results = backtest_options_strategy(
        bars_df,
        strategy,
        initial_capital=1000.0,
        position_size_pct=0.08,
        target_dte=35,
        profit_target_pct=0.75,
        max_loss_pct=-0.90,
        max_holding_days=30
    )

    # Analyze results
    print("\n4. Analyzing results...")
    trades_df = analyze_options_results(results, 1000.0, bars_df)

    print("\n✓ Options backtest complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error during backtest: {e}")
        import traceback
        traceback.print_exc()
