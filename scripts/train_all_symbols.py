#!/usr/bin/env python3
"""Deprecated: use scripts/train_models.py instead."""

import subprocess
import sys


def main() -> int:
    print(
        "[DEPRECATED] scripts/train_all_symbols.py has been replaced by "
        "scripts/train_models.py. Redirecting..."
    )
    cmd = [sys.executable, "scripts/train_models.py"] + sys.argv[1:]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
