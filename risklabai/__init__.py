"""
Local shim for RiskLabAI to avoid eager imports.

This prevents the upstream RiskLabAI package from importing optional
modules at import time (e.g., backtest), while still allowing
RiskLabAI submodules to resolve from site-packages.
"""

from __future__ import annotations

import os
import sysconfig

# Extend this package's path to include the installed RiskLabAI package.
_site_pkg = os.environ.get("RISKLABAI_SITE_PACKAGES")
if not _site_pkg:
    _site_pkg = sysconfig.get_paths().get("purelib")

if _site_pkg:
    _real_pkg = os.path.join(_site_pkg, "RiskLabAI")
    if os.path.isdir(_real_pkg) and _real_pkg not in __path__:
        __path__.append(_real_pkg)

__all__ = []
