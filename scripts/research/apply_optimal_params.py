#!/usr/bin/env python3
"""
Apply Optimal Parameters from Parameter Sweep

Updates the trading bot configuration with the best parameters found
from the parameter sweep analysis.
"""

import sys
from pathlib import Path

# Best parameters from sweep (highest Sharpe ratio)
OPTIMAL_PARAMS = {
    'meta_threshold': 0.001,  # 0.1% - Signal quality filter
    'profit_target': 0.04,     # 4% - Take profit level
    'stop_loss': 0.02,         # 2% - Stop loss level
    'max_holding': 20,         # bars - Maximum holding period
}

PERFORMANCE_METRICS = {
    'sharpe_ratio': 3.53,
    'win_rate': 0.731,
    'num_trades': 52,
    'max_drawdown': -0.0779,
}

def print_parameters():
    """Display the optimal parameters."""
    print("=" * 80)
    print("OPTIMAL TRADING PARAMETERS")
    print("=" * 80)
    print("\nFrom Parameter Sweep Analysis:")
    print(f"  Configurations tested: 48")
    print(f"  Best configuration (by Sharpe ratio):")
    print()
    print(f"  Meta Threshold:    {OPTIMAL_PARAMS['meta_threshold']:.4f} ({OPTIMAL_PARAMS['meta_threshold']*100:.2f}%)")
    print(f"  Profit Target:     {OPTIMAL_PARAMS['profit_target']:.4f} ({OPTIMAL_PARAMS['profit_target']*100:.1f}%)")
    print(f"  Stop Loss:         {OPTIMAL_PARAMS['stop_loss']:.4f} ({OPTIMAL_PARAMS['stop_loss']*100:.1f}%)")
    print(f"  Risk:Reward Ratio: 1:{OPTIMAL_PARAMS['profit_target']/OPTIMAL_PARAMS['stop_loss']:.1f}")
    print(f"  Max Holding:       {OPTIMAL_PARAMS['max_holding']} bars")
    print()
    print("Performance Metrics (from backtest):")
    print(f"  Sharpe Ratio:  {PERFORMANCE_METRICS['sharpe_ratio']:.2f}")
    print(f"  Win Rate:      {PERFORMANCE_METRICS['win_rate']:.1%}")
    print(f"  Num Trades:    {PERFORMANCE_METRICS['num_trades']}")
    print(f"  Max Drawdown:  {PERFORMANCE_METRICS['max_drawdown']:.2%}")
    print()
    print("=" * 80)
    print()


def update_trading_strategy():
    """
    Instructions for applying these parameters to your trading bot.
    """
    print("TO APPLY THESE PARAMETERS:")
    print()
    print("1. Update your RiskLabAICombined strategy initialization:")
    print()
    print("   In core/risklabai_combined.py, update the predict() call:")
    print()
    print("   # Use optimal meta threshold")
    print(f"   signal, bet_size = self.strategy.predict(")
    print(f"       bars,")
    print(f"       prob_threshold=0.015,  # Keep model default")
    print(f"       meta_threshold={OPTIMAL_PARAMS['meta_threshold']}")
    print(f"   )")
    print()
    print("2. Update exit parameters in your position management:")
    print()
    print(f"   profit_target = {OPTIMAL_PARAMS['profit_target']}  # {OPTIMAL_PARAMS['profit_target']*100:.1f}%")
    print(f"   stop_loss = {OPTIMAL_PARAMS['stop_loss']}      # {OPTIMAL_PARAMS['stop_loss']*100:.1f}%")
    print(f"   max_holding = {OPTIMAL_PARAMS['max_holding']}         # bars")
    print()
    print("3. Test in paper trading mode first:")
    print()
    print("   python run_live_trading.py")
    print()
    print("4. Monitor performance for 30 days before going live")
    print()
    print("=" * 80)


def create_config_snippet():
    """Create a code snippet file for easy copy-paste."""
    snippet = f"""# Optimal Parameters from Parameter Sweep
# Date: 2025-12-30
# Sharpe: {PERFORMANCE_METRICS['sharpe_ratio']:.2f} | Win Rate: {PERFORMANCE_METRICS['win_rate']:.1%} | Trades: {PERFORMANCE_METRICS['num_trades']}

# In your strategy predict() call:
signal, bet_size = self.strategy.predict(
    bars,
    prob_threshold=0.015,  # Model's optimized threshold
    meta_threshold={OPTIMAL_PARAMS['meta_threshold']}     # Optimized: 0.1% confidence filter
)

# In your position management:
profit_target = {OPTIMAL_PARAMS['profit_target']}  # {OPTIMAL_PARAMS['profit_target']*100:.1f}% take profit
stop_loss = {OPTIMAL_PARAMS['stop_loss']}      # {OPTIMAL_PARAMS['stop_loss']*100:.1f}% stop loss
max_holding = {OPTIMAL_PARAMS['max_holding']}         # {OPTIMAL_PARAMS['max_holding']} bars max hold
"""

    snippet_file = Path('config/optimal_params_snippet.py')
    snippet_file.parent.mkdir(exist_ok=True)
    snippet_file.write_text(snippet)

    print(f"✓ Code snippet saved to: {snippet_file}")
    print()


if __name__ == "__main__":
    print_parameters()
    update_trading_strategy()
    create_config_snippet()

    print("\n⚠️  IMPORTANT REMINDERS:")
    print("  - These parameters are optimized on ~3-4 days of data")
    print("  - Always paper trade new parameters for 30+ days")
    print("  - Monitor Sharpe ratio - if it drops below 1.5, re-optimize")
    print("  - Set max drawdown alerts at 15%")
    print()
