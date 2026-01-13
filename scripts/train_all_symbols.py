#!/usr/bin/env python3
"""
Train RiskLabAI Models for All Symbols

This script trains models for all 102 symbols using the tier 1 fixes:
- C1: Extended data (365 days, 2000+ samples)
- C2: CUSUM filter (~35-40% filter rate with 5.5x volatility threshold)
- C3: Robust fractional differentiation (optimal d per symbol)
- C7: max_holding=30 bars for balanced label distribution (~25-35% neutral)

Usage:
    # Train all symbols:
    python scripts/train_all_symbols.py

    # Train specific symbol:
    python scripts/train_all_symbols.py --symbol AAPL

    # Deep tune primary/meta models:
    python scripts/train_all_symbols.py --symbol SPY --deep-search --tune-primary-trials 200

    # Time-ordered split with fixed primary params:
    python scripts/train_all_symbols.py --symbol SPY --split-method time --no-tune-primary \
        --primary-params-json '{"max_depth":3,"min_child_weight":1,"reg_alpha":0.6,"reg_lambda":2.0,"gamma":0.4,"subsample":0.6,"colsample_bytree":0.8,"colsample_bylevel":0.4,"learning_rate":0.03,"n_estimators":150,"early_stopping_rounds":30}'

    # Resume from last failed symbol:
    python scripts/train_all_symbols.py --resume
"""

import sys
import json
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


def train_symbol(
    symbol: str,
    save_models: bool = True,
    tune_primary: bool = True,
    tune_meta: bool = True,
    tune_primary_trials: int = 25,
    tune_primary_seed: int = 42,
    primary_gap_tolerance: float = 0.05,
    primary_gap_penalty: float = 0.75,
    meta_c_candidates: list[float] | None = None,
    primary_search_depth: str = "standard",
    meta_search_depth: str = "standard",
    meta_l1_ratios: list[float] | None = None,
    split_method: str = "stratified",
    split_test_size: float = 0.2,
    primary_params_override: dict | None = None,
    meta_params_override: dict | None = None
) -> dict:
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
            max_holding=20,  # Balanced neutral rate with tick-level labeling
            d=None,  # Auto-calculate optimal d
            n_cv_splits=5,
            tune_primary=tune_primary,
            tune_meta=tune_meta,
            tune_primary_trials=tune_primary_trials,
            tune_primary_seed=tune_primary_seed,
            primary_gap_tolerance=primary_gap_tolerance,
            primary_gap_penalty=primary_gap_penalty,
            meta_c_candidates=meta_c_candidates,
            primary_search_depth=primary_search_depth,
            meta_search_depth=meta_search_depth,
            meta_l1_ratios=meta_l1_ratios,
            split_method=split_method,
            split_test_size=split_test_size,
            primary_params_override=primary_params_override,
            meta_params_override=meta_params_override
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
            logger.info(f"  Primary balanced accuracy: {results['primary_accuracy']:.3f}")
            logger.info(f"  Meta balanced accuracy: {results['meta_accuracy']:.3f}")
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
    parser.add_argument(
        '--tune-primary-trials',
        type=int,
        default=None,
        help='Number of primary model parameter samples'
    )
    parser.add_argument(
        '--tune-primary-seed',
        type=int,
        default=42,
        help='Random seed for primary model parameter sampling'
    )
    parser.add_argument(
        '--primary-gap-tolerance',
        type=float,
        default=0.05,
        help='Allowed train/test gap before penalty'
    )
    parser.add_argument(
        '--primary-gap-penalty',
        type=float,
        default=0.75,
        help='Penalty applied to gap beyond tolerance'
    )
    parser.add_argument(
        '--meta-c-candidates',
        type=str,
        help='Comma-separated C values for meta model tuning'
    )
    parser.add_argument(
        '--primary-search-depth',
        type=str,
        choices=['standard', 'deep'],
        default='standard',
        help='Primary param search depth'
    )
    parser.add_argument(
        '--meta-search-depth',
        type=str,
        choices=['standard', 'deep'],
        default='standard',
        help='Meta param search depth'
    )
    parser.add_argument(
        '--meta-l1-ratios',
        type=str,
        help='Comma-separated l1_ratio values for elasticnet meta tuning'
    )
    parser.add_argument(
        '--split-method',
        type=str,
        choices=['stratified', 'time'],
        default='stratified',
        help='Train/test split method'
    )
    parser.add_argument(
        '--split-test-size',
        type=float,
        default=0.2,
        help='Fraction of data reserved for test'
    )
    parser.add_argument(
        '--primary-params-json',
        type=str,
        help='JSON dict of primary model params to use (overrides tuning)'
    )
    parser.add_argument(
        '--meta-params-json',
        type=str,
        help='JSON dict of meta model params to use (overrides tuning)'
    )
    parser.add_argument(
        '--deep-search',
        action='store_true',
        help='Enable deep search for primary and meta params'
    )
    parser.add_argument(
        '--no-tune-primary',
        dest='tune_primary',
        action='store_false',
        help='Disable primary model tuning'
    )
    parser.add_argument(
        '--no-tune-meta',
        dest='tune_meta',
        action='store_false',
        help='Disable meta model tuning'
    )
    parser.set_defaults(tune_primary=True, tune_meta=True)

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

    meta_c_candidates = None
    if args.meta_c_candidates:
        meta_c_candidates = [float(v) for v in args.meta_c_candidates.split(',') if v.strip()]

    meta_l1_ratios = None
    if args.meta_l1_ratios:
        meta_l1_ratios = [float(v) for v in args.meta_l1_ratios.split(',') if v.strip()]

    primary_params_override = None
    if args.primary_params_json:
        try:
            primary_params_override = json.loads(args.primary_params_json)
        except json.JSONDecodeError as exc:
            parser.error(f"Invalid JSON for --primary-params-json: {exc}")
        if not isinstance(primary_params_override, dict):
            parser.error("--primary-params-json must be a JSON object")

    meta_params_override = None
    if args.meta_params_json:
        try:
            meta_params_override = json.loads(args.meta_params_json)
        except json.JSONDecodeError as exc:
            parser.error(f"Invalid JSON for --meta-params-json: {exc}")
        if not isinstance(meta_params_override, dict):
            parser.error("--meta-params-json must be a JSON object")

    primary_search_depth = args.primary_search_depth
    meta_search_depth = args.meta_search_depth
    if args.deep_search:
        primary_search_depth = 'deep'
        meta_search_depth = 'deep'

    tune_primary_trials = args.tune_primary_trials
    if tune_primary_trials is None:
        tune_primary_trials = 200 if args.deep_search else 25

    for i, symbol in enumerate(symbols_to_train, 1):
        logger.info(f"\n[{i}/{len(symbols_to_train)}] Training {symbol}...")

        result = train_symbol(
            symbol,
            save_models=not args.no_save,
            tune_primary=args.tune_primary,
            tune_meta=args.tune_meta,
            tune_primary_trials=tune_primary_trials,
            tune_primary_seed=args.tune_primary_seed,
            primary_gap_tolerance=args.primary_gap_tolerance,
            primary_gap_penalty=args.primary_gap_penalty,
            meta_c_candidates=meta_c_candidates,
            primary_search_depth=primary_search_depth,
            meta_search_depth=meta_search_depth,
            meta_l1_ratios=meta_l1_ratios,
            split_method=args.split_method,
            split_test_size=args.split_test_size,
            primary_params_override=primary_params_override,
            meta_params_override=meta_params_override
        )
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
