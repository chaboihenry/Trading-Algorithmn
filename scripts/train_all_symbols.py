#!/usr/bin/env python3
"""
Train RiskLabAI Models for All Symbols

This script trains models for all 102 symbols using the tier 1 fixes:
- C1: Extended data (365 days, 2000+ samples)
- C2: CUSUM filter (~35% filter rate with 2.5x volatility threshold)
- C3: Robust fractional differentiation (optimal d per symbol)

Usage:
    # Train all symbols:
    python scripts/train_all_symbols.py

    # Train specific symbol:
    python scripts/train_all_symbols.py --symbol AAPL

    # Resume from last failed symbol:
    python scripts/train_all_symbols.py --resume
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.tick_config import SYMBOLS
from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_progress.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def train_symbol(symbol: str, save_models: bool = True) -> dict:
    """
    Train RiskLabAI model for a single symbol.

    Args:
        symbol: Stock ticker
        save_models: Whether to save trained models

    Returns:
        Training results dictionary
    """
    logger.info("=" * 80)
    logger.info(f"TRAINING: {symbol}")
    logger.info("=" * 80)

    try:
        # Initialize strategy (d=None means auto-calculate per symbol)
        strategy = RiskLabAIStrategy(
            profit_taking=2.0,
            stop_loss=2.0,
            max_holding=10,
            d=None,  # Auto-calculate optimal d
            n_cv_splits=5
        )

        # Train from tick data
        results = strategy.train_from_ticks(
            symbol=symbol,
            threshold=None,  # Use config default
            min_samples=100
        )

        if results['success']:
            if save_models:
                model_path = project_root / 'models' / f'risklabai_{symbol}_models.pkl'
                strategy.save_models(str(model_path))
                logger.info(f"✓ Models saved: {model_path}")

            logger.info("=" * 80)
            logger.info(f"✓ SUCCESS: {symbol}")
            logger.info("=" * 80)
            logger.info(f"  Samples: {results['n_samples']}")
            logger.info(f"  Primary accuracy: {results['primary_accuracy']:.3f}")
            logger.info(f"  Meta accuracy: {results['meta_accuracy']:.3f}")
            logger.info("=" * 80)

            return {
                'symbol': symbol,
                'success': True,
                'n_samples': results['n_samples'],
                'primary_accuracy': results['primary_accuracy'],
                'meta_accuracy': results['meta_accuracy']
            }
        else:
            logger.warning(f"⚠️ FAILED: {symbol} - {results.get('reason', 'unknown')}")
            return {
                'symbol': symbol,
                'success': False,
                'reason': results.get('reason', 'unknown')
            }

    except Exception as e:
        logger.error(f"❌ ERROR: {symbol} - {e}")
        return {
            'symbol': symbol,
            'success': False,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Train RiskLabAI models for all symbols'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        help='Train specific symbol only'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last failed symbol'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save trained models (testing only)'
    )

    args = parser.parse_args()

    # Determine which symbols to train
    if args.symbol:
        symbols_to_train = [args.symbol.upper()]
    else:
        symbols_to_train = SYMBOLS

    logger.info("=" * 80)
    logger.info("TRAINING ALL SYMBOLS WITH TIER 1 FIXES")
    logger.info("=" * 80)
    logger.info(f"Total symbols: {len(symbols_to_train)}")
    logger.info(f"C1: Extended data (365 days)")
    logger.info(f"C2: CUSUM filter (2.5x volatility threshold)")
    logger.info(f"C3: Robust fractional differentiation")
    logger.info("=" * 80)

    start_time = datetime.now()
    results = []

    for i, symbol in enumerate(symbols_to_train, 1):
        logger.info(f"\n[{i}/{len(symbols_to_train)}] Training {symbol}...")

        result = train_symbol(symbol, save_models=not args.no_save)
        results.append(result)

        # Show progress
        successes = sum(1 for r in results if r['success'])
        failures = len(results) - successes

        logger.info(f"\nProgress: {i}/{len(symbols_to_train)} complete")
        logger.info(f"  Successes: {successes}")
        logger.info(f"  Failures: {failures}")

    # Final summary
    elapsed = datetime.now() - start_time
    successes = [r for r in results if r['success']]
    failures = [r for r in results if not r['success']]

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total symbols: {len(results)}")
    logger.info(f"Successful: {len(successes)}")
    logger.info(f"Failed: {len(failures)}")
    logger.info(f"Elapsed time: {elapsed}")
    logger.info("=" * 80)

    if successes:
        logger.info("\nSUCCESSFUL SYMBOLS:")
        for r in successes:
            logger.info(f"  ✓ {r['symbol']}: {r['n_samples']} samples, "
                       f"accuracy={r['primary_accuracy']:.3f}")

    if failures:
        logger.info("\nFAILED SYMBOLS:")
        for r in failures:
            reason = r.get('reason', r.get('error', 'unknown'))
            logger.info(f"  ❌ {r['symbol']}: {reason}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ All training jobs complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
