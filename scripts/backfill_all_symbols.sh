#!/usr/bin/env bash
# Deprecated: use scripts/backfill_ticks.py instead.

set -euo pipefail

echo "[DEPRECATED] scripts/backfill_all_symbols.sh has been replaced by scripts/backfill_ticks.py. Redirecting..."
exec python3 scripts/backfill_ticks.py "$@"
