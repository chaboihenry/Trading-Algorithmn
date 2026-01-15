#!/usr/bin/env python3
"""Deprecated: use scripts/init_tick_tables.py instead."""

import subprocess
import sys


def main() -> int:
    print(
        "[DEPRECATED] scripts/setup/init_tick_tables.py has been replaced by "
        "scripts/init_tick_tables.py. Redirecting..."
    )
    cmd = [sys.executable, "scripts/init_tick_tables.py"] + sys.argv[1:]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
