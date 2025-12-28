#!/usr/bin/env python3
"""
Live Trading Bot with Tick-Based RiskLabAI Models

This script runs the trading bot in paper trading mode with:
- Tick imbalance bar-based models
- Profitability tracking (win rate, Sharpe ratio, drawdown)
- 30-day validation period before considering live trading

Usage:
    python run_live_trading.py
"""

import logging
from datetime import datetime
from lumibot.brokers import Alpaca
from lumibot.traders import Trader

from core.risklabai_combined import RiskLabAICombined
from backup.settings import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_PAPER

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/live_trading_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """
    Run live trading bot with profitability tracking.
    """
    print("=" * 80)
    print("RISKLABAI LIVE TRADING BOT")
    print("=" * 80)
    print(f"Mode: {'PAPER TRADING' if ALPACA_PAPER else 'LIVE TRADING'}")
    print(f"Started: {datetime.now()}")
    print("=" * 80)

    # Configure Alpaca broker
    alpaca_config = {
        "API_KEY": ALPACA_API_KEY,
        "API_SECRET": ALPACA_API_SECRET,
        "PAPER": ALPACA_PAPER
    }

    broker = Alpaca(alpaca_config)

    # Create trader first
    trader = Trader()

    # Initialize strategy with OPTIMIZED tick-based models
    # Set parameters as class attribute before creating instance
    RiskLabAICombined.parameters = {
        # Trading symbols
        'symbols': ['SPY'],

        # OPTIMIZED MODEL: Tighter targets for more frequent signals
        'model_path': 'models/risklabai_tick_models_optimized.pkl',
        'profit_taking': 0.5,  # 0.5% profit target (was 2.0%)
        'stop_loss': 0.5,      # 0.5% stop loss (was 2.0%)
        'max_holding': 20,     # 20 bars max (was 10)

        # Strategy settings
        'use_tick_bars': True,
        'enable_profitability_tracking': True,
        'min_training_bars': 100,
        'retrain_days': 30,

        # KELLY CRITERION POSITION SIZING
        'use_kelly_sizing': True,
        'kelly_fraction': 0.5,  # Half-Kelly for safety
        'estimated_win_rate': 0.5323,  # From optimized model meta accuracy

        # RISK MANAGEMENT CONTROLS
        'daily_loss_limit_pct': 0.03,  # 3% max daily loss
        'max_drawdown_pct': 0.10,      # 10% max drawdown (hard stop)
        'drawdown_warning_pct': 0.05,  # 5% drawdown warning level
        'max_consecutive_losses': 3,    # Pause after 3 losses
        'max_trades_per_day': 15,      # Prevent overtrading

        # GARCH VOLATILITY FILTER (Prediction Activation)
        # Complements CUSUM (training) with GARCH (prediction)
        # Only activate RiskLabAI during high-volatility regimes
        'use_garch_filter': True,
        'garch_lookback': 100,         # Lookback period for volatility estimation
        'garch_percentile': 0.60       # Activate when vol > 60th percentile
    }

    strategy = RiskLabAICombined(broker=broker)
    trader.add_strategy(strategy)

    # Run trading bot
    logger.info("Starting trading bot...")
    logger.info("Profitability logs: logs/profitability_logs/")
    logger.info("Press Ctrl+C to stop")

    try:
        trader.run_all()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        raise
    finally:
        # Print final profitability summary
        if hasattr(strategy, 'profitability_tracker'):
            strategy.profitability_tracker.save_summary()
            strategy.profitability_tracker.print_summary()


if __name__ == "__main__":
    main()
