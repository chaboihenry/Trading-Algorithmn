#!/usr/bin/env python3
"""
Build tick imbalance bars from stored tick data.

Usage:
    python scripts/build_bars.py --symbol SPY
    python scripts/build_bars.py --tier tier_1 --max-symbols 50
    python scripts/build_bars.py --symbol SPY --start 2024-01-01 --end 2024-03-01
"""

import argparse
import logging
from datetime import datetime

from config.tick_config import TICK_DB_PATH, INITIAL_IMBALANCE_THRESHOLD
from config.universe import build_universe
from data.tick_storage import TickStorage, to_epoch_ms
from data.tick_to_bars import TickImbalanceBarGenerator

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build imbalance bars from ticks")
    parser.add_argument("--symbol", help="Single symbol to process")
    parser.add_argument("--symbols", help="Comma-separated symbol list")
    parser.add_argument("--tier", default=None, help="Symbol tier (tier_1..tier_5)")
    parser.add_argument("--max-symbols", type=int, default=0, help="Max symbols for tier")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--threshold", type=float, default=None, help="Initial imbalance threshold")
    parser.add_argument("--ewma-alpha", type=float, default=0.05, help="EWMA alpha for imbalance")
    parser.add_argument("--chunk-days", type=int, default=1, help="Days per tick load chunk")
    parser.add_argument("--batch-size", type=int, default=5000, help="Bars per DB write")
    return parser.parse_args()


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def _select_symbols(args: argparse.Namespace) -> list:
    if args.symbol:
        return [args.symbol]
    if args.symbols:
        return [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    tier = args.tier or "tier_1"
    max_symbols = args.max_symbols or None
    symbols, _, _ = build_universe(tier=tier, max_symbols=max_symbols)
    return symbols


def build_bars_for_symbol(
    storage: TickStorage,
    symbol: str,
    start_ms: int,
    end_ms: int,
    threshold: float,
    ewma_alpha: float,
    chunk_days: int,
    batch_size: int,
) -> int:
    generator = TickImbalanceBarGenerator(threshold=threshold, ewma_alpha=ewma_alpha)
    bars_buffer = []
    total_bars = 0
    total_ticks = 0

    chunk_ms = chunk_days * 24 * 60 * 60 * 1000
    cursor_ms = start_ms

    while cursor_ms <= end_ms:
        window_end = min(cursor_ms + chunk_ms - 1, end_ms)
        ticks = storage.load_ticks(
            symbol,
            start=cursor_ms,
            end=window_end,
            timestamp_format="epoch_ms"
        )

        if ticks:
            for timestamp_ms, price, size in ticks:
                total_ticks += 1
                bar = generator.process_tick(timestamp_ms, price, size)
                if bar:
                    bars_buffer.append(bar)
                    total_bars += 1

                if len(bars_buffer) >= batch_size:
                    storage.save_bars(symbol, bars_buffer)
                    bars_buffer = []
        else:
            logger.debug(
                f"{symbol}: no ticks for {datetime.utcfromtimestamp(cursor_ms/1000).date()}"
            )

        cursor_ms = window_end + 1

    if bars_buffer:
        storage.save_bars(symbol, bars_buffer)

    logger.info(
        f"{symbol}: processed {total_ticks:,} ticks, generated {total_bars:,} bars"
    )
    return total_bars


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    symbols = _select_symbols(args)
    if not symbols:
        logger.error("No symbols selected")
        return 1

    threshold = args.threshold if args.threshold is not None else INITIAL_IMBALANCE_THRESHOLD

    storage = TickStorage(str(TICK_DB_PATH))
    try:
        for symbol in symbols:
            date_range = storage.get_date_range(symbol)
            if not date_range:
                logger.warning(f"{symbol}: no ticks available")
                continue

            earliest, latest = date_range
            start_ms = to_epoch_ms(_parse_date(args.start)) if args.start else earliest
            end_ms = to_epoch_ms(_parse_date(args.end)) if args.end else latest

            if start_ms > end_ms:
                logger.warning(f"{symbol}: start date after end date, skipping")
                continue

            logger.info(
                f"{symbol}: building bars from {datetime.utcfromtimestamp(start_ms/1000).date()} "
                f"to {datetime.utcfromtimestamp(end_ms/1000).date()}"
            )

            build_bars_for_symbol(
                storage=storage,
                symbol=symbol,
                start_ms=start_ms,
                end_ms=end_ms,
                threshold=threshold,
                ewma_alpha=args.ewma_alpha,
                chunk_days=max(1, args.chunk_days),
                batch_size=max(100, args.batch_size)
            )

    finally:
        storage.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
