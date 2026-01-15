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
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
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
    SYMBOLS,
    SYMBOL_TIER,
    MAX_SYMBOLS,
    TICK_DB_PATH
)
from config.tick_config import ensure_alpaca_credentials
LOG_DIR = Path(os.environ.get("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
(LOG_DIR / "profitability_logs").mkdir(parents=True, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f'live_trading_{datetime.now().strftime("%Y%m%d")}.log'),
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
    ensure_alpaca_credentials()
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

    # Choose which tier of US stocks to trade
    # tier_1: Top 100 most liquid (AAPL, MSFT, etc.) - START HERE
    # tier_2: Top 500 (S&P 500 level)
    # tier_3: Top 1000 (Russell 1000)
    # tier_4: Top 2000
    # tier_5: ALL liquid US stocks (2000-5000 symbols)

    ACTIVE_SYMBOLS = SYMBOLS

    tier_desc = SYMBOL_TIER or "tier_1"
    if MAX_SYMBOLS:
        tier_desc += f" (max {MAX_SYMBOLS})"
    logger.info(f"Trading {len(ACTIVE_SYMBOLS)} symbols from {tier_desc}")
    logger.info(f"First 10: {', '.join(ACTIVE_SYMBOLS[:10])}")

    try:
        download_models(symbols=ACTIVE_SYMBOLS, require_s3=True)
    except Exception as exc:
        logger.error(f"Model download failed: {exc}")
        return 1

    models_dir = Path(os.environ.get("MODELS_PATH", "models"))
    if not models_dir.exists():
        logger.error(f"Models directory missing: {models_dir}")
        return 1

    missing_models = [
        symbol for symbol in ACTIVE_SYMBOLS
        if not any(models_dir.glob(f"risklabai_{symbol}_*.pkl"))
    ]
    if missing_models:
        logger.error(
            "Missing models after S3 download for: "
            f"{', '.join(missing_models)}"
        )
        logger.error("Update the S3 bucket or reduce the symbol list.")
        return 1

    logger.info("=" * 80)
    logger.info(f"DATA SOURCE: Tick Imbalance Bars (Database @ {TICK_DB_PATH})")
    logger.info("=" * 80)
    if not TICK_DB_PATH.exists() or TICK_DB_PATH.stat().st_size == 0:
        logger.error(f"Tick database missing or empty: {TICK_DB_PATH}")
        logger.error("Run tick backfill before starting live trading.")
        return 1

    # Initialize strategy with OPTIMIZED tick-based models
    # Set parameters as class attribute before creating instance
    RiskLabAICombined.parameters = {
        # Trading symbols - Multi-asset universe
        # Each symbol will load its own model: models/risklabai_{symbol}_models.pkl
        'symbols': ACTIVE_SYMBOLS,

        # OPTIMAL PARAMETERS (Locked for Tuning)
        # Note: Each symbol loads its own model from models/risklabai_{symbol}_models.pkl
        'profit_taking': OPTIMAL_PROFIT_TARGET,  # 2.5x volatility profit target
        'stop_loss': OPTIMAL_STOP_LOSS,          # 2.5x volatility stop loss
        'max_holding': OPTIMAL_MAX_HOLDING_BARS, # 10 bars max hold
        'd': None,                               # Auto-calculated per symbol

        # Signal thresholds
        'meta_threshold': OPTIMAL_META_THRESHOLD,  # 0.001 (0.1%) - Meta model confidence
        'prob_threshold': OPTIMAL_PROB_THRESHOLD,  # 0.015 (1.5%) - Primary model probability

        # Strategy settings
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
