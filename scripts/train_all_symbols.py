#!/usr/bin/env python3
"""
Multi-Symbol Model Training

Trains RiskLabAI models for ALL US stocks.
Each symbol gets its own dedicated model stored as models/risklabai_{symbol}_models.pkl

Usage:
    python scripts/train_all_symbols.py [--tier TIER] [--parallel N]

Arguments:
    --tier: Which tier to train (tier_1, tier_2, tier_3, tier_4, tier_5)
            tier_1: Top 100 most liquid (START HERE)
            tier_2: Top 500 (S&P 500 level)
            tier_3: Top 1000 (Russell 1000)
            tier_4: Top 2000
            tier_5: ALL liquid stocks
            Default: tier_1
    --parallel: Number of parallel training jobs (default: 1)
                Warning: Training is memory-intensive!
"""

import sys
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from config.tick_config import (
    TICK_DB_PATH,
    INITIAL_IMBALANCE_THRESHOLD,
    OPTIMAL_PROFIT_TARGET,
    OPTIMAL_STOP_LOSS,
    OPTIMAL_MAX_HOLDING_BARS,
    OPTIMAL_FRACTIONAL_D
)
from data.tick_storage import TickStorage
from data.tick_to_bars import generate_bars_from_ticks
from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_symbol(symbol: str) -> dict:
    """
    Train RiskLabAI model for a single symbol.

    Args:
        symbol: Stock/ETF symbol

    Returns:
        dict with symbol, success status, and metrics
    """
    logger.info("=" * 80)
    logger.info(f"TRAINING MODEL FOR {symbol}")
    logger.info("=" * 80)

    try:
        # Step 1: Load tick data
        logger.info(f"[{symbol}] Step 1: Loading tick data...")
        storage = TickStorage(TICK_DB_PATH)

        # Check if data exists
        date_range = storage.get_date_range(symbol)
        if not date_range:
            storage.close()
            logger.error(f"[{symbol}] ✗ No tick data found! Run backfill first.")
            return {
                'symbol': symbol,
                'success': False,
                'message': 'No tick data found',
                'metrics': {}
            }

        earliest, latest = date_range
        logger.info(f"[{symbol}] Available data: {earliest} to {latest}")

        # Load ticks
        ticks = storage.load_ticks(symbol, limit=5000000)
        storage.close()

        if not ticks:
            logger.error(f"[{symbol}] ✗ Failed to load ticks")
            return {
                'symbol': symbol,
                'success': False,
                'message': 'Failed to load ticks',
                'metrics': {}
            }

        logger.info(f"[{symbol}] ✓ Loaded {len(ticks):,} ticks")

        # Step 2: Generate bars
        logger.info(f"[{symbol}] Step 2: Generating tick imbalance bars...")
        bars_list = generate_bars_from_ticks(ticks, threshold=INITIAL_IMBALANCE_THRESHOLD)

        if not bars_list:
            logger.error(f"[{symbol}] ✗ Failed to generate bars")
            return {
                'symbol': symbol,
                'success': False,
                'message': 'Failed to generate bars',
                'metrics': {}
            }

        logger.info(f"[{symbol}] ✓ Generated {len(bars_list)} bars")

        # Convert to DataFrame
        bars_df = pd.DataFrame(bars_list)
        bars_df['bar_end'] = pd.to_datetime(bars_df['bar_end'])
        if bars_df['bar_end'].dt.tz is not None:
            bars_df['bar_end'] = bars_df['bar_end'].dt.tz_localize(None)
        bars_df.set_index('bar_end', inplace=True)

        # Step 3: Initialize strategy
        logger.info(f"[{symbol}] Step 3: Initializing RiskLabAI strategy...")
        strategy = RiskLabAIStrategy(
            profit_taking=OPTIMAL_PROFIT_TARGET,
            stop_loss=OPTIMAL_STOP_LOSS,
            max_holding=OPTIMAL_MAX_HOLDING_BARS,
            d=OPTIMAL_FRACTIONAL_D,
            n_cv_splits=5,
            force_directional=True,  # Force directional labels
            neutral_threshold=0.00001
        )

        # Step 4: Train model
        logger.info(f"[{symbol}] Step 4: Training model...")
        logger.info(f"[{symbol}]   - Profit target: {OPTIMAL_PROFIT_TARGET:.2%}")
        logger.info(f"[{symbol}]   - Stop loss: {OPTIMAL_STOP_LOSS:.2%}")
        logger.info(f"[{symbol}]   - Max holding: {OPTIMAL_MAX_HOLDING_BARS} bars")
        logger.info(f"[{symbol}]   - Fractional d: {OPTIMAL_FRACTIONAL_D}")
        logger.info(f"[{symbol}]   - Force directional: True")

        # Train the model
        strategy.train(bars_df)

        # Step 5: Evaluate model
        logger.info(f"[{symbol}] Step 5: Evaluating model...")

        # Get primary model accuracy
        if hasattr(strategy, 'primary_accuracy'):
            primary_acc = strategy.primary_accuracy
        else:
            primary_acc = 0.0

        # Get meta model accuracy
        if hasattr(strategy, 'meta_accuracy'):
            meta_acc = strategy.meta_accuracy
        else:
            meta_acc = 0.0

        logger.info(f"[{symbol}] Primary model accuracy: {primary_acc:.2%}")
        logger.info(f"[{symbol}] Meta model accuracy: {meta_acc:.2%}")

        # Step 6: Save model
        model_path = f"models/risklabai_{symbol}_models.pkl"
        logger.info(f"[{symbol}] Step 6: Saving model to {model_path}...")
        strategy.save_models(model_path)

        logger.info(f"[{symbol}] ✓ Training completed successfully!")

        return {
            'symbol': symbol,
            'success': True,
            'message': 'Training completed',
            'metrics': {
                'primary_accuracy': primary_acc,
                'meta_accuracy': meta_acc,
                'num_bars': len(bars_df),
                'num_ticks': len(ticks),
                'data_range': f"{earliest} to {latest}"
            }
        }

    except Exception as e:
        logger.error(f"[{symbol}] ✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'symbol': symbol,
            'success': False,
            'message': str(e),
            'metrics': {}
        }


def main():
    """Run multi-symbol training."""
    parser = argparse.ArgumentParser(description='Train models for all US stocks')
    parser.add_argument('--tier', type=str, default='tier_1',
                        choices=['tier_1', 'tier_2', 'tier_3', 'tier_4', 'tier_5'],
                        help='Which tier to train (tier_1=top 100, tier_5=all)')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel training jobs (WARNING: memory intensive!)')

    args = parser.parse_args()

    # Import and get symbols for this tier
    try:
        from config.all_symbols import get_symbols_by_tier
        symbols = get_symbols_by_tier(args.tier)
    except ImportError:
        logger.error("all_symbols.py not found!")
        logger.error("Run: python scripts/fetch_all_symbols.py first")
        return 1

    logger.info("=" * 80)
    logger.info(f"TRAINING MODELS FOR ALL US STOCKS - {args.tier.upper()}")
    logger.info("=" * 80)
    logger.info(f"Symbols to train: {len(symbols)}")
    logger.info(f"First 10 symbols: {', '.join(symbols[:10])}")
    logger.info(f"Parallel jobs: {args.parallel}")
    logger.info(f"Started: {datetime.now()}")
    logger.info(f"Estimated time: {len(symbols) * 15 / 60:.1f} hours")
    logger.info("=" * 80)
    logger.info("")

    # Train models (sequentially for now to avoid memory issues)
    results = []
    for symbol in symbols:
        result = train_symbol(symbol)
        results.append(result)
        logger.info("")

    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    logger.info(f"Total symbols: {len(results)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info("")

    if successful:
        logger.info("✓ Successfully trained:")
        for r in successful:
            symbol = r['symbol']
            metrics = r['metrics']
            logger.info(f"  - {symbol}: "
                       f"Primary={metrics.get('primary_accuracy', 0):.2%}, "
                       f"Meta={metrics.get('meta_accuracy', 0):.2%}, "
                       f"Bars={metrics.get('num_bars', 0)}")

    if failed:
        logger.info("")
        logger.error("✗ Failed to train:")
        for r in failed:
            logger.error(f"  - {r['symbol']}: {r['message']}")

    logger.info("")
    logger.info(f"Completed: {datetime.now()}")
    logger.info("=" * 80)

    # Return non-zero exit code if any failed
    return 0 if len(failed) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
