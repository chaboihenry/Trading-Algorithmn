#!/usr/bin/env python3
"""
Train RiskLabAI models from stored imbalance bars and save via ModelStorage.

Usage:
    python scripts/train_models.py --symbol SPY
    python scripts/train_models.py --tier tier_1 --max-symbols 50
    python scripts/train_models.py --symbol SPY --split-method walk_forward
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv

from config.tick_config import TICK_DB_PATH
from config.universe import build_universe
from data.tick_storage import TickStorage
from data.model_storage import ModelStorage
from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RiskLabAI models from imbalance bars")
    parser.add_argument("--symbol", help="Single symbol to train")
    parser.add_argument("--symbols", help="Comma-separated symbol list")
    parser.add_argument("--tier", default=None, help="Symbol tier (tier_1..tier_5)")
    parser.add_argument("--max-symbols", type=int, default=0, help="Max symbols for tier")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--min-samples", type=int, default=100, help="Minimum samples for training")
    parser.add_argument("--profit-taking", type=float, default=2.5, help="Triple-barrier profit multiplier")
    parser.add_argument("--stop-loss", type=float, default=2.5, help="Triple-barrier stop-loss multiplier")
    parser.add_argument("--max-holding", type=int, default=10, help="Triple-barrier max holding bars")
    parser.add_argument("--d", type=float, default=None, help="Fixed fractional diff d (default: auto)")
    parser.add_argument("--split-method", choices=["walk_forward", "time"], default="walk_forward")
    parser.add_argument("--split-test-size", type=float, default=0.2, help="Holdout size for time split")
    parser.add_argument("--walk-forward-splits", type=int, default=5, help="Number of walk-forward splits")
    parser.add_argument("--tune-primary", action="store_true", help="Enable primary model tuning")
    parser.add_argument("--tune-meta", action="store_true", help="Enable meta model tuning")
    parser.add_argument("--tune-primary-trials", type=int, default=25, help="Primary tuning trials")
    parser.add_argument("--primary-search-depth", choices=["standard", "deep"], default="standard")
    parser.add_argument("--meta-search-depth", choices=["standard", "deep"], default="standard")
    parser.add_argument("--primary-params-json", type=str, help="Override primary params (JSON dict)")
    parser.add_argument("--meta-params-json", type=str, help="Override meta params (JSON dict)")
    parser.add_argument("--no-save", action="store_true", help="Skip saving models")
    parser.add_argument("--no-s3", action="store_true", help="Do not upload to S3")
    return parser.parse_args()


def _load_json_arg(value: Optional[str]) -> Optional[Dict[str, Any]]:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {value}") from exc


def _select_symbols(args: argparse.Namespace) -> List[str]:
    if args.symbol:
        return [args.symbol]
    if args.symbols:
        return [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    tier = args.tier or "tier_1"
    max_symbols = args.max_symbols or None
    symbols, _, _ = build_universe(tier=tier, max_symbols=max_symbols)
    return symbols


def _build_model_payload(strategy: RiskLabAIStrategy, results: Dict[str, Any]) -> Dict[str, Any]:
    primary_params = strategy.primary_model.get_params() if strategy.primary_model else {}
    meta_params = strategy.meta_model.get_params() if strategy.meta_model else {}

    return {
        "primary_model": strategy.primary_model,
        "meta_model": strategy.meta_model,
        "scaler": strategy.scaler,
        "label_encoder": strategy.label_encoder,
        "feature_names": strategy.feature_names,
        "important_features": strategy.important_features,
        "frac_diff_d": strategy.frac_diff.d,
        "model_type": "XGBoost_primary_LR_meta",
        "hyperparameters": {
            "xgb_max_depth": primary_params.get("max_depth"),
            "xgb_learning_rate": primary_params.get("learning_rate"),
            "xgb_reg_alpha": primary_params.get("reg_alpha"),
            "xgb_reg_lambda": primary_params.get("reg_lambda"),
            "xgb_gamma": primary_params.get("gamma"),
            "xgb_min_child_weight": primary_params.get("min_child_weight"),
            "xgb_subsample": primary_params.get("subsample"),
            "xgb_colsample_bytree": primary_params.get("colsample_bytree"),
            "xgb_colsample_bylevel": primary_params.get("colsample_bylevel"),
            "xgb_n_estimators": primary_params.get("n_estimators"),
            "lr_C": meta_params.get("C"),
            "lr_class_weight": meta_params.get("class_weight"),
        },
        "training_metrics": {
            "primary_accuracy": results.get("primary_accuracy"),
            "meta_accuracy": results.get("meta_accuracy"),
            "n_samples": results.get("n_samples"),
            "purged_cv_mean": results.get("purged_cv_mean"),
            "purged_cv_std": results.get("purged_cv_std"),
        },
    }


def main() -> int:
    args = parse_args()
    symbols = _select_symbols(args)
    if not symbols:
        logger.error("No symbols selected")
        return 1

    log_dir = Path(os.environ.get("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "train_models.log"),
            logging.StreamHandler(),
        ],
    )

    primary_params_override = _load_json_arg(args.primary_params_json)
    meta_params_override = _load_json_arg(args.meta_params_json)

    storage = TickStorage(str(TICK_DB_PATH))
    model_storage = ModelStorage(local_dir=os.environ.get("MODELS_PATH", "models"))

    successes = 0
    failures = 0

    try:
        for symbol in symbols:
            logger.info("=" * 80)
            logger.info(f"TRAINING: {symbol}")
            logger.info("=" * 80)

            bars = storage.load_bars(
                symbol,
                start=args.start,
                end=args.end,
                timestamp_format="datetime"
            )

            if bars.empty:
                logger.warning(f"{symbol}: no bars found. Run scripts/build_bars.py first.")
                failures += 1
                continue

            strategy = RiskLabAIStrategy(
                profit_taking=args.profit_taking,
                stop_loss=args.stop_loss,
                max_holding=args.max_holding,
                d=args.d,
                n_cv_splits=args.walk_forward_splits,
                tune_primary=args.tune_primary,
                tune_meta=args.tune_meta,
                tune_primary_trials=args.tune_primary_trials,
                primary_search_depth=args.primary_search_depth,
                meta_search_depth=args.meta_search_depth,
                split_method=args.split_method,
                split_test_size=args.split_test_size,
                primary_params_override=primary_params_override,
                meta_params_override=meta_params_override,
            )

            results = strategy.train(bars, min_samples=args.min_samples, symbol=symbol)

            if not results.get("success"):
                logger.warning(f"{symbol}: training failed ({results.get('reason', 'unknown')})")
                failures += 1
                continue

            if args.no_save:
                logger.info(f"{symbol}: trained (no-save)")
                successes += 1
                continue

            if strategy.primary_model is None or strategy.meta_model is None:
                logger.warning(f"{symbol}: missing trained models, skipping save")
                failures += 1
                continue

            model_payload = _build_model_payload(strategy, results)
            try:
                model_storage.save_model(symbol, model_payload, upload_to_s3=not args.no_s3)
                logger.info(f"{symbol}: model saved")
                successes += 1
            except Exception as exc:
                logger.error(f"{symbol}: failed to save model ({exc})")
                failures += 1

    finally:
        storage.close()

    logger.info("=" * 80)
    logger.info(f"Training complete: {successes} succeeded, {failures} failed")
    logger.info("=" * 80)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
