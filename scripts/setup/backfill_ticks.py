#!/usr/bin/env python3
"""Deprecated: use scripts/backfill_ticks.py instead."""

import subprocess
import sys


def main() -> int:
    print(
        "[DEPRECATED] scripts/setup/backfill_ticks.py has been replaced by "
        "scripts/backfill_ticks.py. Redirecting..."
    )
    cmd = [sys.executable, "scripts/backfill_ticks.py"] + sys.argv[1:]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
