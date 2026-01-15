"""
Backfill historical ticks into the SQLite database.

Usage:
    python scripts/backfill_ticks.py --symbol SPY --days 30
    python scripts/backfill_ticks.py --symbol AAPL --start 2024-01-01 --end 2024-01-31
"""

import argparse
import logging
from datetime import datetime, timedelta

from config.tick_config import BACKFILL_DAYS, TICK_DB_PATH
from config.universe import build_universe
from data.alpaca_tick_client import AlpacaTickClient
from data.tick_storage import TickStorage, to_epoch_ms

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill tick data from Alpaca")
    parser.add_argument("--symbol", help="Stock symbol to backfill (e.g., SPY)")
    parser.add_argument("--tier", default=None, help="Symbol tier (tier_1..tier_5)")
    parser.add_argument("--max-symbols", type=int, default=0, help="Max symbols when using tier")
    parser.add_argument("--days", type=int, default=None, help="Days to backfill (overrides tier days)")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--extended-hours",
        action="store_true",
        help="Include pre/post market data",
    )
    return parser.parse_args()


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    if args.symbol:
        symbols = [args.symbol]
        backfill_days = {args.symbol: args.days or BACKFILL_DAYS}
    else:
        tier = args.tier or "tier_1"
        max_symbols = args.max_symbols or None
        symbols, _, backfill_days = build_universe(tier=tier, max_symbols=max_symbols)
        if args.days is not None:
            backfill_days = {symbol: args.days for symbol in symbols}

    if not symbols:
        logger.error("No symbols selected for backfill")
        return 1

    client = AlpacaTickClient()
    storage = TickStorage(str(TICK_DB_PATH))

    for symbol in symbols:
        if args.start and args.end:
            start_date = parse_date(args.start)
            end_date = parse_date(args.end)
        else:
            days = backfill_days.get(symbol, BACKFILL_DAYS)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

        ticks = client.fetch_ticks_range(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            extended_hours=args.extended_hours,
        )

        saved = storage.save_ticks(symbol, ticks)

        if ticks:
            timestamps_ms = [to_epoch_ms(t["timestamp"]) for t in ticks]
            earliest = min(timestamps_ms)
            latest = max(timestamps_ms)
            total = storage.get_tick_count(symbol)
            storage.update_backfill_status(symbol, earliest, latest, total)

        logger.info(f"{symbol}: saved {saved} ticks")

    storage.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
