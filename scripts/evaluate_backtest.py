#!/usr/bin/env python3
"""
Out-of-sample backtest using imbalance bars with transaction costs.

Usage:
    python scripts/evaluate_backtest.py --symbol SPY
    python scripts/evaluate_backtest.py --tier tier_1 --max-symbols 20 --cost-bps 1.0
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from config.tick_config import TICK_DB_PATH
from config.universe import build_universe
from data.tick_storage import TickStorage
from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate out-of-sample backtest")
    parser.add_argument("--symbol", help="Single symbol to test")
    parser.add_argument("--symbols", help="Comma-separated symbol list")
    parser.add_argument("--tier", default=None, help="Symbol tier (tier_1..tier_5)")
    parser.add_argument("--max-symbols", type=int, default=0, help="Max symbols for tier")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction")
    parser.add_argument("--min-samples", type=int, default=100, help="Minimum samples")
    parser.add_argument("--profit-taking", type=float, default=2.5, help="Triple-barrier profit multiplier")
    parser.add_argument("--stop-loss", type=float, default=2.5, help="Triple-barrier stop-loss multiplier")
    parser.add_argument("--max-holding", type=int, default=10, help="Triple-barrier max holding bars")
    parser.add_argument("--prob-threshold", type=float, default=0.015, help="Primary prob threshold")
    parser.add_argument("--meta-threshold", type=float, default=0.001, help="Meta prob threshold")
    parser.add_argument("--margin-threshold", type=float, default=0.03, help="Winner margin threshold")
    parser.add_argument("--cost-bps", type=float, default=0.5, help="Commission cost in bps per side")
    parser.add_argument("--slippage-bps", type=float, default=0.5, help="Slippage in bps per side")
    parser.add_argument("--annualization", type=float, default=252, help="Sharpe annualization factor")
    return parser.parse_args()


def _select_symbols(args: argparse.Namespace) -> List[str]:
    if args.symbol:
        return [args.symbol]
    if args.symbols:
        return [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    tier = args.tier or "tier_1"
    max_symbols = args.max_symbols or None
    symbols, _, _ = build_universe(tier=tier, max_symbols=max_symbols)
    return symbols


def _compute_signals(
    strategy: RiskLabAIStrategy,
    features: pd.DataFrame,
    prob_threshold: float,
    meta_threshold: float,
    margin_threshold: float,
) -> Dict[str, Any]:
    X_scaled = strategy.scaler.transform(features)
    proba = strategy.primary_model.predict_proba(X_scaled)
    pred_encoded = strategy.primary_model.predict(X_scaled)

    classes = strategy.label_encoder.classes_
    proba_df = pd.DataFrame(proba, index=features.index, columns=classes)

    prob_short = proba_df[-1].values if -1 in proba_df.columns else np.zeros(len(proba_df))
    prob_neutral = proba_df[0].values if 0 in proba_df.columns else np.zeros(len(proba_df))
    prob_long = proba_df[1].values if 1 in proba_df.columns else np.zeros(len(proba_df))

    prob_stack = np.stack([prob_short, prob_neutral, prob_long], axis=1)
    winner_idx = np.argmax(prob_stack, axis=1)
    winners = np.array([-1, 0, 1])[winner_idx]

    sorted_probs = np.sort(prob_stack, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2] if prob_stack.shape[1] > 1 else sorted_probs[:, -1]

    direction = winners.copy()
    keep_mask = (
        (direction != 0)
        & (margin >= margin_threshold)
        & (np.maximum(prob_long, prob_short) > prob_threshold)
    )
    direction = np.where(keep_mask, direction, 0)

    if strategy.meta_model is not None:
        proba_encoded = pd.DataFrame(proba, index=features.index, columns=range(proba.shape[1]))
        meta_features = strategy._build_meta_features(
            proba_encoded,
            pd.Series(pred_encoded, index=features.index)
        ).fillna(0.0)
        meta_proba = strategy.meta_model.predict_proba(meta_features)[:, 1]
        bet_size = np.where(meta_proba >= meta_threshold, meta_proba, 0.0)
        direction = np.where(meta_proba >= meta_threshold, direction, 0)
    else:
        bet_size = np.where(direction != 0, 0.5, 0.0)

    return {
        "direction": direction,
        "bet_size": bet_size,
        "primary_proba": proba,
        "pred_encoded": pred_encoded,
    }


def _compute_metrics(net_returns: np.ndarray, trade_mask: np.ndarray, annualization: float) -> Dict[str, Any]:
    equity = np.cumprod(1 + net_returns)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    mean_ret = float(np.mean(net_returns)) if len(net_returns) else 0.0
    std_ret = float(np.std(net_returns, ddof=1)) if len(net_returns) > 1 else 0.0
    sharpe = (mean_ret / std_ret) * np.sqrt(annualization) if std_ret > 0 else 0.0

    trade_returns = net_returns[trade_mask]
    win_rate = float((trade_returns > 0).mean()) if trade_returns.size else 0.0

    return {
        "total_return": float(equity[-1] - 1) if len(equity) else 0.0,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "trade_count": int(trade_mask.sum()),
        "win_rate": win_rate,
    }


def backtest_symbol(symbol: str, args: argparse.Namespace, storage: TickStorage) -> Dict[str, Any]:
    bars = storage.load_bars(
        symbol,
        start=args.start,
        end=args.end,
        timestamp_format="datetime"
    )

    if bars.empty:
        return {"symbol": symbol, "success": False, "reason": "no_bars"}

    bars = bars.sort_index()
    split_idx = int(len(bars) * (1 - args.test_size))
    split_idx = max(1, min(split_idx, len(bars) - 1))

    train_bars = bars.iloc[:split_idx]
    test_bars = bars.iloc[split_idx:]

    strategy = RiskLabAIStrategy(
        profit_taking=args.profit_taking,
        stop_loss=args.stop_loss,
        max_holding=args.max_holding,
        split_method="time",
        split_test_size=args.test_size,
    )

    results = strategy.train(train_bars, min_samples=args.min_samples, symbol=symbol)
    if not results.get("success"):
        return {"symbol": symbol, "success": False, "reason": results.get("reason", "train_failed")}

    bars_all = pd.concat([train_bars, test_bars]).sort_index()
    features_all = strategy.prepare_features(bars_all, symbol=symbol)
    features_test = features_all.loc[features_all.index.intersection(test_bars.index)]

    if features_test.empty:
        return {"symbol": symbol, "success": False, "reason": "no_features"}

    test_bars = test_bars.loc[features_test.index]

    signals = _compute_signals(
        strategy,
        features_test,
        prob_threshold=args.prob_threshold,
        meta_threshold=args.meta_threshold,
        margin_threshold=args.margin_threshold,
    )

    future_returns = test_bars["close"].pct_change().shift(-1)
    future_returns = future_returns.reindex(features_test.index)

    valid_mask = future_returns.notna().values
    direction = signals["direction"][valid_mask]
    bet_size = signals["bet_size"][valid_mask]
    returns = future_returns.values[valid_mask]

    position = direction * bet_size

    round_trip_cost = (args.cost_bps + args.slippage_bps) / 10000.0 * 2
    trade_cost = round_trip_cost * np.abs(position)

    net_returns = position * returns - trade_cost
    trade_mask = position != 0

    metrics = _compute_metrics(net_returns, trade_mask, args.annualization)
    metrics.update({
        "symbol": symbol,
        "success": True,
        "bars": len(bars),
        "train_bars": len(train_bars),
        "test_bars": len(test_bars),
    })

    return metrics


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
            logging.FileHandler(log_dir / "evaluate_backtest.log"),
            logging.StreamHandler(),
        ],
    )

    storage = TickStorage(str(TICK_DB_PATH))
    results = []

    try:
        for symbol in symbols:
            logger.info("=" * 80)
            logger.info(f"BACKTEST: {symbol}")
            logger.info("=" * 80)
            outcome = backtest_symbol(symbol, args, storage)
            results.append(outcome)

            if outcome.get("success"):
                logger.info(
                    f"{symbol}: return={outcome['total_return']:.2%}, "
                    f"sharpe={outcome['sharpe']:.2f}, "
                    f"max_dd={outcome['max_drawdown']:.2%}, "
                    f"trades={outcome['trade_count']}"
                )
            else:
                logger.warning(f"{symbol}: backtest failed ({outcome.get('reason')})")
    finally:
        storage.close()

    successes = [r for r in results if r.get("success")]
    if successes:
        avg_return = float(np.mean([r["total_return"] for r in successes]))
        avg_sharpe = float(np.mean([r["sharpe"] for r in successes]))
        avg_drawdown = float(np.mean([r["max_drawdown"] for r in successes]))
        logger.info("=" * 80)
        logger.info(
            f"Aggregate ({len(successes)} symbols): return={avg_return:.2%}, "
            f"sharpe={avg_sharpe:.2f}, max_dd={avg_drawdown:.2%}"
        )
        logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
