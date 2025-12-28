#!/usr/bin/env python3
"""
Comprehensive Backtest: GARCH Filter + RiskLabAI Integration

Tests the COMPLETE system to verify trade generation and return potential:
1. GARCH filter identifies high-volatility regimes
2. RiskLabAI generates signals during those regimes
3. Kelly Criterion position sizing
4. Full risk management

This will show if we can achieve 20% monthly returns.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy
from utils.garch_filter import GARCHVolatilityFilter
from config.tick_config import TICK_DB_PATH, INITIAL_IMBALANCE_THRESHOLD
from data.tick_storage import TickStorage
from data.tick_to_bars import generate_bars_from_ticks

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IntegratedSystemBacktest:
    """
    Backtest the COMPLETE integrated system:
    - GARCH volatility filter (activation gate)
    - RiskLabAI (signal generation)
    - Kelly Criterion (position sizing)
    - Risk controls (daily loss, drawdown)
    """

    def __init__(
        self,
        model_path: str,
        initial_capital: float = 1000.0,
        profit_target: float = 0.5,
        stop_loss: float = 0.5,
        kelly_fraction: float = 0.5,
        garch_percentile: float = 0.60,
        meta_threshold: float = 0.50
    ):
        """Initialize integrated backtest."""
        self.model_path = model_path
        self.initial_capital = initial_capital
        self.profit_target = profit_target / 100.0
        self.stop_loss = stop_loss / 100.0
        self.kelly_fraction = kelly_fraction
        self.garch_percentile = garch_percentile
        self.meta_threshold = meta_threshold

        # Load model
        logger.info(f"Loading model from {model_path}")
        self.strategy = RiskLabAIStrategy(
            profit_taking=profit_target,
            stop_loss=stop_loss,
            max_holding=20,
            n_cv_splits=5
        )
        self.strategy.load_models(model_path)
        logger.info("✓ Model loaded")

        # Initialize GARCH filter
        self.garch_filter = GARCHVolatilityFilter(
            lookback_period=100,
            volatility_percentile=garch_percentile,
            min_observations=50
        )
        logger.info(f"✓ GARCH filter initialized (percentile={garch_percentile:.0%})")

    def simulate_trading(
        self,
        bars_df: pd.DataFrame,
        estimated_win_rate: float = 0.5323
    ) -> Dict:
        """
        Simulate trading with FULL integrated system.
        """
        capital = self.initial_capital
        position = None
        trades = []
        portfolio_values = [capital]

        # Track activation
        garch_checks = {'activated': 0, 'blocked': 0}
        signal_checks = {'long': 0, 'short': 0, 'none': 0}

        # Risk tracking
        peak_capital = capital
        daily_values = {}

        logger.info(f"Starting backtest with ${capital:,.2f}")
        logger.info(f"Testing {len(bars_df)} bars")

        for i in range(100, len(bars_df)):
            if i % 1000 == 0:
                logger.info(f"Processing bar {i}/{len(bars_df)}")

            current_bar = bars_df.iloc[i]
            current_price = current_bar['close']
            current_date = current_bar.name.date()

            # Track daily values for daily returns
            if current_date not in daily_values:
                daily_values[current_date] = capital

            # Close existing position if conditions met
            if position is not None:
                bars_held = i - position['entry_bar']

                # Calculate P&L
                if position['direction'] == 'long':
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price']

                    # Check exit conditions
                    if pnl_pct >= self.profit_target:
                        # Take profit
                        exit_price = position['entry_price'] * (1 + self.profit_target)
                        pnl = position['shares'] * (exit_price - position['entry_price'])
                        capital += pnl
                        trades.append({
                            'entry_bar': position['entry_bar'],
                            'exit_bar': i,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'pnl_pct': self.profit_target,
                            'direction': 'long',
                            'outcome': 'take_profit',
                            'bars_held': bars_held
                        })
                        position = None
                        portfolio_values.append(capital)
                        peak_capital = max(peak_capital, capital)

                    elif pnl_pct <= -self.stop_loss:
                        # Stop loss
                        exit_price = position['entry_price'] * (1 - self.stop_loss)
                        pnl = position['shares'] * (exit_price - position['entry_price'])
                        capital += pnl
                        trades.append({
                            'entry_bar': position['entry_bar'],
                            'exit_bar': i,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'pnl_pct': -self.stop_loss,
                            'direction': 'long',
                            'outcome': 'stop_loss',
                            'bars_held': bars_held
                        })
                        position = None
                        portfolio_values.append(capital)

                    elif bars_held >= 20:
                        # Timeout
                        exit_price = current_price
                        pnl = position['shares'] * (exit_price - position['entry_price'])
                        capital += pnl
                        pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
                        trades.append({
                            'entry_bar': position['entry_bar'],
                            'exit_bar': i,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'direction': 'long',
                            'outcome': 'timeout',
                            'bars_held': bars_held
                        })
                        position = None
                        portfolio_values.append(capital)

            # Enter new position if no position held
            if position is None:
                # Get historical bars
                historical_bars = bars_df.iloc[:i].copy()
                prices = historical_bars['close']

                # STEP 1: Check GARCH filter
                should_trade_garch, garch_info = self.garch_filter.should_trade(prices)

                if not should_trade_garch:
                    garch_checks['blocked'] += 1
                    continue

                # GARCH says trade - proceed
                garch_checks['activated'] += 1

                # STEP 2: Get RiskLabAI signal
                try:
                    signal, bet_size = self.strategy.predict(historical_bars)

                    # Track signals
                    if signal == 1:
                        signal_checks['long'] += 1
                    elif signal == -1:
                        signal_checks['short'] += 1
                    else:
                        signal_checks['none'] += 1

                    # Get meta probability
                    features = self.strategy.prepare_features(historical_bars)
                    if len(features) == 0:
                        continue

                    X = features.iloc[[-1]]

                    if self.strategy.meta_model is not None:
                        meta_prob = self.strategy.meta_model.predict_proba(X)[0, 1]
                    else:
                        meta_prob = 0.5

                    # Check meta threshold
                    if meta_prob < self.meta_threshold:
                        continue

                    # Check signal
                    if signal != 1:  # Only long for now
                        continue

                    # STEP 3: Calculate Kelly position size
                    kelly_base = self.calculate_kelly(estimated_win_rate)

                    # Adjust for drawdown
                    drawdown = (capital - peak_capital) / peak_capital if peak_capital > 0 else 0
                    if drawdown < -0.05:
                        kelly_adjusted = kelly_base * max(0.5, 1.0 + (drawdown / 0.05))
                    else:
                        kelly_adjusted = kelly_base

                    # Apply meta confidence
                    position_fraction = kelly_adjusted * bet_size

                    # Calculate position size
                    position_value = capital * position_fraction
                    shares = position_value / current_price

                    # Enter position
                    position = {
                        'entry_bar': i,
                        'entry_price': current_price,
                        'shares': shares,
                        'direction': 'long',
                        'meta_prob': meta_prob,
                        'bet_size': bet_size,
                        'garch_vol': garch_info['forecasted_vol']
                    }

                except Exception as e:
                    logger.debug(f"Prediction failed at bar {i}: {e}")
                    continue

        # Close any remaining position
        if position is not None:
            exit_price = bars_df.iloc[-1]['close']
            pnl = position['shares'] * (exit_price - position['entry_price'])
            capital += pnl
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
            bars_held = len(bars_df) - 1 - position['entry_bar']
            trades.append({
                'entry_bar': position['entry_bar'],
                'exit_bar': len(bars_df) - 1,
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'direction': 'long',
                'outcome': 'final',
                'bars_held': bars_held
            })
            portfolio_values.append(capital)

        # Calculate metrics
        return self._calculate_metrics(
            trades, portfolio_values, daily_values, capital,
            garch_checks, signal_checks, bars_df
        )

    def calculate_kelly(self, win_rate: float) -> float:
        """Calculate Kelly fraction."""
        if win_rate <= 0 or win_rate >= 1:
            return 0.05

        loss_rate = 1 - win_rate
        win_loss_ratio = self.profit_target / self.stop_loss  # Symmetric

        kelly = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
        kelly_adjusted = max(0.01, min(kelly * self.kelly_fraction, 0.15))

        return kelly_adjusted

    def _calculate_metrics(
        self, trades, portfolio_values, daily_values, final_capital,
        garch_checks, signal_checks, bars_df
    ) -> Dict:
        """Calculate comprehensive performance metrics."""

        if len(trades) == 0:
            return {
                'num_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'avg_bars_held': 0.0,
                'total_return': 0.0,
                'monthly_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'final_capital': final_capital,
                'days_traded': (bars_df.index[-1] - bars_df.index[0]).days if len(bars_df) > 0 else 0,
                'months_traded': ((bars_df.index[-1] - bars_df.index[0]).days / 30.0) if len(bars_df) > 0 else 0,
                'garch_activation_rate': garch_checks['activated'] / (garch_checks['activated'] + garch_checks['blocked']) if (garch_checks['activated'] + garch_checks['blocked']) > 0 else 0,
                'garch_activated': garch_checks['activated'],
                'garch_blocked': garch_checks['blocked'],
                'signal_distribution': signal_checks,
                'signal_long_rate': 0.0,
                'outcomes': {},
                'trades_per_month': 0.0,
                'message': 'NO TRADES GENERATED'
            }

        trades_df = pd.DataFrame(trades)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]

        # Basic metrics
        win_rate = len(winning_trades) / len(trades)
        avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0
        avg_bars_held = trades_df['bars_held'].mean()

        total_return = (final_capital - self.initial_capital) / self.initial_capital

        # Time-based returns
        first_date = bars_df.index[0]
        last_date = bars_df.index[-1]
        days_traded = (last_date - first_date).days
        months_traded = days_traded / 30.0

        monthly_return = (1 + total_return) ** (1 / months_traded) - 1 if months_traded > 0 else 0

        # Calculate daily returns
        daily_return_values = sorted(daily_values.items())
        if len(daily_return_values) > 1:
            daily_returns = []
            for i in range(1, len(daily_return_values)):
                prev_val = daily_return_values[i-1][1]
                curr_val = daily_return_values[i][1]
                daily_ret = (curr_val - prev_val) / prev_val if prev_val > 0 else 0
                daily_returns.append(daily_ret)

            daily_returns = pd.Series(daily_returns)
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        if len(portfolio_values) > 1:
            portfolio_series = pd.Series(portfolio_values)
            cumulative = portfolio_series / portfolio_series.iloc[0]
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0.0

        # GARCH activation rate
        total_garch_checks = garch_checks['activated'] + garch_checks['blocked']
        garch_activation_rate = garch_checks['activated'] / total_garch_checks if total_garch_checks > 0 else 0

        # Signal distribution after GARCH activation
        total_signals = sum(signal_checks.values())
        signal_long_rate = signal_checks['long'] / total_signals if total_signals > 0 else 0

        return {
            'num_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_bars_held': avg_bars_held,
            'total_return': total_return,
            'monthly_return': monthly_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_capital': final_capital,
            'days_traded': days_traded,
            'months_traded': months_traded,
            'garch_activation_rate': garch_activation_rate,
            'garch_activated': garch_checks['activated'],
            'garch_blocked': garch_checks['blocked'],
            'signal_distribution': signal_checks,
            'signal_long_rate': signal_long_rate,
            'outcomes': trades_df['outcome'].value_counts().to_dict(),
            'trades_per_month': len(trades) / months_traded if months_traded > 0 else 0
        }

    def run(self, symbol: str = 'SPY') -> Dict:
        """Run comprehensive backtest."""

        print("\n" + "=" * 80)
        print("INTEGRATED SYSTEM BACKTEST")
        print("GARCH Filter + RiskLabAI + Kelly Criterion + Risk Controls")
        print("=" * 80)
        print(f"Symbol: {symbol}")
        print(f"Initial capital: ${self.initial_capital:,.2f}")
        print(f"Profit/Loss targets: {self.profit_target*100:.1f}% / {self.stop_loss*100:.1f}%")
        print(f"GARCH percentile: {self.garch_percentile:.0%}")
        print(f"Meta threshold: {self.meta_threshold:.2f}")
        print(f"Kelly fraction: {self.kelly_fraction:.0%} (Half-Kelly)")
        print("=" * 80)

        # Load data
        logger.info("\nLoading tick data...")
        storage = TickStorage(TICK_DB_PATH)
        ticks = storage.load_ticks(symbol)
        storage.close()

        if not ticks:
            raise ValueError(f"No tick data found for {symbol}")

        logger.info(f"Loaded {len(ticks):,} ticks")

        # Generate bars
        logger.info("Generating bars...")
        bars_list = generate_bars_from_ticks(ticks, threshold=INITIAL_IMBALANCE_THRESHOLD)
        logger.info(f"Generated {len(bars_list)} bars")

        # Convert to DataFrame
        bars_df = pd.DataFrame(bars_list)
        bars_df['bar_end'] = pd.to_datetime(bars_df['bar_end'])
        if bars_df['bar_end'].dt.tz is not None:
            bars_df['bar_end'] = bars_df['bar_end'].dt.tz_localize(None)
        bars_df.set_index('bar_end', inplace=True)

        # Run backtest
        logger.info("\nRunning integrated backtest...")
        results = self.simulate_trading(bars_df)

        # Print results
        print("\n" + "=" * 80)
        print("BACKTEST RESULTS")
        print("=" * 80)

        print(f"\n📊 GARCH FILTER PERFORMANCE:")
        print(f"  Activation rate: {results['garch_activation_rate']:.1%}")
        print(f"  Activated: {results['garch_activated']:,} bars")
        print(f"  Blocked: {results['garch_blocked']:,} bars")

        print(f"\n📈 SIGNAL GENERATION (after GARCH activation):")
        print(f"  Long signals: {results['signal_distribution']['long']:,}")
        print(f"  No signals: {results['signal_distribution']['none']:,}")
        print(f"  Signal rate: {results['signal_long_rate']:.1%}")

        print(f"\n💰 TRADING PERFORMANCE:")
        print(f"  Trades: {results['num_trades']}")
        print(f"  Trades per month: {results.get('trades_per_month', 0):.1f}")
        print(f"  Win rate: {results['win_rate']:.1%}")
        print(f"  Avg win: {results['avg_win']:.2%}")
        print(f"  Avg loss: {results['avg_loss']:.2%}")
        print(f"  Avg hold: {results['avg_bars_held']:.1f} bars")

        print(f"\n📅 RETURNS:")
        print(f"  Days traded: {results['days_traded']}")
        print(f"  Months traded: {results['months_traded']:.1f}")
        print(f"  Total return: {results['total_return']:.2%}")
        print(f"  Monthly return: {results['monthly_return']:.2%}")
        print(f"  Sharpe ratio: {results['sharpe_ratio']:.3f}")
        print(f"  Max drawdown: {results['max_drawdown']:.2%}")

        print(f"\n💵 CAPITAL:")
        print(f"  Start: ${self.initial_capital:,.2f}")
        print(f"  End: ${results['final_capital']:,.2f}")
        print(f"  P&L: ${results['final_capital'] - self.initial_capital:,.2f}")

        if 'outcomes' in results:
            print(f"\n🎯 TRADE OUTCOMES:")
            for outcome, count in results['outcomes'].items():
                print(f"  {outcome}: {count}")

        # Growth projection
        print("\n" + "=" * 80)
        print("GROWTH PROJECTION TO $10,000")
        print("=" * 80)

        if results['monthly_return'] > 0:
            months_to_10k = np.log(10000 / self.initial_capital) / np.log(1 + results['monthly_return'])
            print(f"At {results['monthly_return']:.2%} monthly return:")
            print(f"  Months to $10,000: {months_to_10k:.1f}")
            print(f"  Years to $10,000: {months_to_10k/12:.1f}")

            # Compare to 20% target
            print(f"\nTarget: 20% monthly return")
            print(f"Actual: {results['monthly_return']:.2%} monthly return")
            if results['monthly_return'] >= 0.20:
                print(f"✓ EXCEEDS TARGET!")
            elif results['monthly_return'] >= 0.15:
                print(f"⚠️  Close to target (within 5%)")
            else:
                print(f"❌ Below target - needs optimization")
        else:
            print("No positive returns - system not profitable in backtest")

        print("=" * 80)

        return results


def main():
    """Main entry point."""

    # Test integrated system
    model_path = 'models/risklabai_tick_models_optimized.pkl'

    # Test different GARCH percentiles to find optimal
    percentiles = [0.50, 0.55, 0.60, 0.65, 0.70]

    print("\n" + "=" * 80)
    print("TESTING DIFFERENT GARCH PERCENTILES")
    print("=" * 80)

    best_monthly_return = -1
    best_percentile = 0.60
    all_results = []

    for percentile in percentiles:
        print(f"\n{'='*80}")
        print(f"TESTING GARCH PERCENTILE: {percentile:.0%}")
        print(f"{'='*80}")

        backtest = IntegratedSystemBacktest(
            model_path=model_path,
            initial_capital=1000.0,
            profit_target=0.5,
            stop_loss=0.5,
            kelly_fraction=0.5,
            garch_percentile=percentile,
            meta_threshold=0.50
        )

        results = backtest.run('SPY')
        results['garch_percentile'] = percentile
        all_results.append(results)

        if results['monthly_return'] > best_monthly_return:
            best_monthly_return = results['monthly_return']
            best_percentile = percentile

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: ALL GARCH PERCENTILES")
    print("=" * 80)

    summary_df = pd.DataFrame([{
        'Percentile': r['garch_percentile'],
        'Trades': r['num_trades'],
        'Win Rate': f"{r['win_rate']:.1%}",
        'Monthly Return': f"{r['monthly_return']:.2%}",
        'Sharpe': f"{r['sharpe_ratio']:.2f}",
        'Max DD': f"{r['max_drawdown']:.2%}",
        'GARCH Activation': f"{r['garch_activation_rate']:.1%}"
    } for r in all_results])

    print(summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("OPTIMAL CONFIGURATION")
    print("=" * 80)
    print(f"Best GARCH percentile: {best_percentile:.0%}")
    print(f"Best monthly return: {best_monthly_return:.2%}")
    print(f"Target monthly return: 20.00%")

    if best_monthly_return >= 0.20:
        print("\n✓ SYSTEM CAN ACHIEVE 20% MONTHLY RETURN!")
    else:
        print(f"\n⚠️  System achieves {best_monthly_return:.2%} monthly")
        print(f"   Gap to target: {(0.20 - best_monthly_return):.2%}")
        print(f"\nConsider:")
        print(f"  • Adjusting meta threshold")
        print(f"  • Adjusting GARCH percentile further")
        print(f"  • Retraining with different parameters")

    print("=" * 80)

    # Save results
    output_path = project_root / f"integrated_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
