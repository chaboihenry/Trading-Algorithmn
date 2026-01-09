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
import signal
import sys
from datetime import datetime
from lumibot.brokers import Alpaca
from lumibot.traders import Trader
from utils.model_downloader import download_models

from core.risklabai_combined import RiskLabAICombined
from config.tick_config import (
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    ALPACA_PAPER,
    OPTIMAL_PROFIT_TARGET,
    OPTIMAL_STOP_LOSS,
    OPTIMAL_MAX_HOLDING_BARS,
    OPTIMAL_META_THRESHOLD,
    OPTIMAL_PROB_THRESHOLD,
    OPTIMAL_FRACTIONAL_D,
    should_use_tick_bars  # Auto-detect tick database for portability
)
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

# Global flag for graceful shutdown
shutdown_requested = False
force_exit = False


def signal_handler(sig, frame):
    """
    Handle Ctrl+C (SIGINT) gracefully.

    First Ctrl+C: Initiates graceful shutdown (saves models, logs, etc.)
    Second Ctrl+C: Forces immediate exit without cleanup
    """
    global shutdown_requested, force_exit

    if force_exit:
        # Third Ctrl+C or already force exiting
        logger.warning("\n⚠️  Force exit already in progress...")
        return

    if shutdown_requested:
        # Second Ctrl+C - force immediate exit
        logger.warning("\n🚨 FORCE EXIT: Second Ctrl+C detected!")
        logger.warning("⏭️  Skipping cleanup and exiting immediately...")
        force_exit = True
        sys.exit(1)
    else:
        # First Ctrl+C - graceful shutdown
        logger.info("\n🛑 Shutdown requested (Ctrl+C)")
        logger.info("⏳ Cleaning up... (Press Ctrl+C again to force quit)")
        shutdown_requested = True
        # Let the exception propagate to trigger cleanup
        raise KeyboardInterrupt()


def main():
    """
    Run live trading bot with profitability tracking.
    """
    print("=" * 80)
    print("RISKLABAI LIVE TRADING BOT")
    print("=" * 80)

    download_models()

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

    # Choose which tier of US stocks to trade
    # tier_1: Top 100 most liquid (AAPL, MSFT, etc.) - START HERE
    # tier_2: Top 500 (S&P 500 level)
    # tier_3: Top 1000 (Russell 1000)
    # tier_4: Top 2000
    # tier_5: ALL liquid US stocks (2000-5000 symbols)

    try:
        from config.all_symbols import get_symbols_by_tier
        ACTIVE_SYMBOLS = get_symbols_by_tier('tier_1')  # Start with top 100
    except ImportError:
        logger.error("all_symbols.py not found!")
        logger.error("Run: python scripts/setup/fetch_all_symbols.py first")
        return 1

    logger.info(f"Trading {len(ACTIVE_SYMBOLS)} symbols from tier_1")
    logger.info(f"First 10: {', '.join(ACTIVE_SYMBOLS[:10])}")

    # AUTO-DETECT: Use tick bars if database exists, otherwise use Alpaca API
    USE_TICK_BARS = should_use_tick_bars()
    logger.info("=" * 80)
    logger.info(f"DATA SOURCE: {'Tick Imbalance Bars (Database)' if USE_TICK_BARS else 'Alpaca API Real-Time Bars'}")
    logger.info("=" * 80)

    # Initialize strategy with OPTIMIZED tick-based models
    # Set parameters as class attribute before creating instance
    RiskLabAICombined.parameters = {
        # Trading symbols - Multi-asset universe
        # Each symbol will load its own model: models/risklabai_{symbol}_models.pkl
        'symbols': ACTIVE_SYMBOLS,

        # OPTIMAL PARAMETERS from parameter sweep (Sharpe 3.53, Win Rate 73.1%)
        # Note: Each symbol loads its own model from models/risklabai_{symbol}_models.pkl
        'profit_taking': OPTIMAL_PROFIT_TARGET,  # 4.0% (0.04) profit target
        'stop_loss': OPTIMAL_STOP_LOSS,          # 2.0% (0.02) stop loss
        'max_holding': OPTIMAL_MAX_HOLDING_BARS, # 20 bars max hold
        'd': OPTIMAL_FRACTIONAL_D,               # 0.30 - Fractional differencing (preserves 70% memory)

        # Signal thresholds
        'meta_threshold': OPTIMAL_META_THRESHOLD,  # 0.001 (0.1%) - Meta model confidence
        'prob_threshold': OPTIMAL_PROB_THRESHOLD,  # 0.015 (1.5%) - Primary model probability

        # Strategy settings - AUTO-DETECTED for portability
        'use_tick_bars': USE_TICK_BARS,  # Auto-detect: True if DB exists, False otherwise
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
        'max_trades_per_day': 15       # Prevent overtrading
    }

    strategy = RiskLabAICombined(broker=broker)
    trader.add_strategy(strategy)

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)  # Docker stop signal

    # Run trading bot
    logger.info("Starting trading bot...")
    logger.info("Profitability logs: logs/profitability_logs/")
    logger.info("Press Ctrl+C once to stop gracefully")
    logger.info("Press Ctrl+C twice to force quit immediately")

    try:
        trader.run_all()
    except KeyboardInterrupt:
        if not shutdown_requested:
            logger.info("🛑 Shutdown initiated by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        raise
    finally:
        # Only do cleanup if not force exiting
        if not force_exit:
            try:
                # Print final profitability summary
                if hasattr(strategy, 'profitability_tracker'):
                    logger.info("📊 Generating profitability summary...")
                    strategy.profitability_tracker.save_summary()
                    strategy.profitability_tracker.print_summary()
                logger.info("✅ Cleanup complete")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
        else:
            logger.warning("⏭️  Cleanup skipped (force exit)")


if __name__ == "__main__":
    main()
