#!/usr/bin/env python3
"""
Optimize Meta Threshold for the NEW Optimized Model

Tests the optimized model (0.5% targets) with different meta thresholds
to find the sweet spot for signal quality vs quantity.

This should generate ACTUAL trades (unlike the old 2.0% model).
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy
from config.tick_config import TICK_DB_PATH, INITIAL_IMBALANCE_THRESHOLD
from data.tick_storage import TickStorage
from data.tick_to_bars import generate_bars_from_ticks

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizedModelBacktest:
    """
    Backtest optimized model with different meta thresholds.
    """

    def __init__(
        self,
        model_path: str,
        initial_capital: float = 1000.0,
        profit_target: float = 0.5,  # NEW: 0.5% profit target
        stop_loss: float = 0.5,      # NEW: 0.5% stop loss
        position_size_pct: float = 0.0656  # Kelly half-fraction
    ):
        """Initialize backtest."""
        self.model_path = model_path
        self.initial_capital = initial_capital
        self.profit_target = profit_target / 100.0
        self.stop_loss = stop_loss / 100.0
        self.position_size_pct = position_size_pct

        # Load optimized model
        logger.info(f"Loading optimized model from {model_path}")
        self.strategy = RiskLabAIStrategy(
            profit_taking=profit_target,
            stop_loss=stop_loss,
            max_holding=20,
            n_cv_splits=5
        )
        self.strategy.load_models(model_path)
        logger.info("✓ Optimized model loaded successfully")

    def simulate_trades(
        self,
        bars_df: pd.DataFrame,
        meta_threshold: float = 0.5
    ) -> Dict:
        """
        Simulate trading with a specific meta threshold.
        """
        capital = self.initial_capital
        position = None
        trades = []
        portfolio_values = [capital]
        signals_seen = {'long': 0, 'short': 0, 'none': 0}

        logger.debug(f"Simulating with threshold={meta_threshold:.2f}")

        for i in range(100, len(bars_df)):
            # Get historical bars for prediction
            historical_bars = bars_df.iloc[:i].copy()

            # Get signal and bet size from model
            try:
                signal, bet_size = self.strategy.predict(historical_bars)

                # Track signals
                if signal == 1:
                    signals_seen['long'] += 1
                elif signal == -1:
                    signals_seen['short'] += 1
                else:
                    signals_seen['none'] += 1

                # Get meta probability
                features = self.strategy.prepare_features(historical_bars)
                if len(features) == 0:
                    continue

                X = features.iloc[[-1]]

                if self.strategy.meta_model is not None:
                    meta_prob = self.strategy.meta_model.predict_proba(X)[0, 1]
                else:
                    meta_prob = 0.5

            except Exception as e:
                logger.warning(f"Prediction failed at bar {i}: {e}")
                continue

            current_bar = bars_df.iloc[i]
            current_price = current_bar['close']

            # Close existing position if conditions met
            if position is not None:
                bars_held = i - position['entry_bar']

                # Check stop loss / take profit
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']

                if position['direction'] == 'long':
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
                        # Timeout - exit at current price
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

            # Enter new position if signal meets threshold
            if position is None and meta_prob >= meta_threshold and signal == 1:
                # Enter long position
                position_value = capital * self.position_size_pct
                shares = position_value / current_price

                position = {
                    'entry_bar': i,
                    'entry_price': current_price,
                    'shares': shares,
                    'direction': 'long',
                    'meta_prob': meta_prob
                }

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
        if len(trades) == 0:
            return {
                'meta_threshold': meta_threshold,
                'num_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'final_capital': capital,
                'signals': signals_seen,
                'avg_bars_held': 0
            }

        # Calculate returns
        trades_df = pd.DataFrame(trades)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]

        win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
        avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0
        avg_bars_held = trades_df['bars_held'].mean()

        total_return = (capital - self.initial_capital) / self.initial_capital

        # Sharpe ratio
        if len(portfolio_values) > 1:
            returns = pd.Series(portfolio_values).pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

            # Max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            sharpe_ratio = 0.0
            max_drawdown = 0.0

        # Count outcomes
        outcomes = trades_df['outcome'].value_counts().to_dict()

        return {
            'meta_threshold': meta_threshold,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_capital': capital,
            'signals': signals_seen,
            'avg_bars_held': avg_bars_held,
            'outcomes': outcomes
        }

    def optimize(
        self,
        symbol: str = 'SPY',
        thresholds: List[float] = None
    ) -> pd.DataFrame:
        """
        Test multiple thresholds and find optimal.
        """
        if thresholds is None:
            thresholds = [0.45, 0.50, 0.55, 0.60, 0.65]

        logger.info("=" * 80)
        logger.info("OPTIMIZED MODEL META THRESHOLD BACKTEST")
        logger.info("=" * 80)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Profit/Loss: {self.profit_target*100:.1f}% / {self.stop_loss*100:.1f}%")
        logger.info(f"Position size: {self.position_size_pct:.2%} (Kelly half-fraction)")
        logger.info(f"Thresholds to test: {thresholds}")
        logger.info(f"Initial capital: ${self.initial_capital:,.2f}")
        logger.info("=" * 80)

        # Load tick data and generate bars
        logger.info("\nLoading tick data...")
        storage = TickStorage(TICK_DB_PATH)
        ticks = storage.load_ticks(symbol)
        storage.close()

        if not ticks:
            raise ValueError(f"No tick data found for {symbol}")

        logger.info(f"Loaded {len(ticks):,} ticks")

        # Generate bars
        logger.info("Generating bars from ticks...")
        bars_list = generate_bars_from_ticks(ticks, threshold=INITIAL_IMBALANCE_THRESHOLD)
        logger.info(f"Generated {len(bars_list)} bars")

        # Convert to DataFrame
        bars_df = pd.DataFrame(bars_list)
        bars_df['bar_end'] = pd.to_datetime(bars_df['bar_end'])
        if bars_df['bar_end'].dt.tz is not None:
            bars_df['bar_end'] = bars_df['bar_end'].dt.tz_localize(None)
        bars_df.set_index('bar_end', inplace=True)

        # Test each threshold
        results = []
        for threshold in thresholds:
            logger.info(f"\nTesting threshold: {threshold:.2f}")
            logger.info("-" * 60)

            metrics = self.simulate_trades(bars_df, meta_threshold=threshold)
            results.append(metrics)

            # Print results
            logger.info(f"  Signals: Long={metrics['signals']['long']}, None={metrics['signals']['none']}")
            logger.info(f"  Trades: {metrics['num_trades']}")
            logger.info(f"  Win Rate: {metrics['win_rate']:.2%}")
            logger.info(f"  Avg Win: {metrics['avg_win']:.2%}")
            logger.info(f"  Avg Loss: {metrics['avg_loss']:.2%}")
            logger.info(f"  Avg Hold: {metrics['avg_bars_held']:.1f} bars")
            logger.info(f"  Total Return: {metrics['total_return']:.2%}")
            logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
            logger.info(f"  Final Capital: ${metrics['final_capital']:,.2f}")
            if 'outcomes' in metrics:
                logger.info(f"  Outcomes: {metrics['outcomes']}")

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST RESULTS SUMMARY")
        logger.info("=" * 80)

        print("\n" + results_df[[
            'meta_threshold', 'num_trades', 'win_rate', 'total_return',
            'sharpe_ratio', 'max_drawdown', 'final_capital'
        ]].to_string(index=False))

        # Find optimal threshold (maximize Sharpe ratio)
        if results_df['num_trades'].sum() > 0:
            # Only consider thresholds with trades
            valid_results = results_df[results_df['num_trades'] > 0]
            if len(valid_results) > 0:
                optimal_idx = valid_results['sharpe_ratio'].idxmax()
                optimal = valid_results.loc[optimal_idx]

                logger.info("\n" + "=" * 80)
                logger.info("✅ OPTIMAL THRESHOLD FOUND")
                logger.info("=" * 80)
                logger.info(f"Threshold: {optimal['meta_threshold']:.2f}")
                logger.info(f"Trades: {optimal['num_trades']}")
                logger.info(f"Win Rate: {optimal['win_rate']:.2%}")
                logger.info(f"Total Return: {optimal['total_return']:.2%}")
                logger.info(f"Sharpe Ratio: {optimal['sharpe_ratio']:.3f}")
                logger.info(f"Final Capital: ${optimal['final_capital']:,.2f}")
                logger.info("=" * 80)
        else:
            logger.warning("\n⚠️  NO TRADES GENERATED - Model may still be too conservative")

        # Save results
        output_path = project_root / f"optimized_model_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_path, index=False)
        logger.info(f"\nResults saved to: {output_path}")

        return results_df


def main():
    """Main entry point."""
    # Test optimized model
    model_path = 'models/risklabai_tick_models_optimized.pkl'

    backtest = OptimizedModelBacktest(
        model_path=model_path,
        initial_capital=1000.0,
        profit_target=0.5,  # 0.5% targets
        stop_loss=0.5,
        position_size_pct=0.0656  # 6.56% Kelly half-fraction
    )

    # Optimize
    results = backtest.optimize(
        symbol='SPY',
        thresholds=[0.45, 0.50, 0.55, 0.60, 0.65]
    )

    print("\n✓ Backtest complete!")


if __name__ == "__main__":
    main()
