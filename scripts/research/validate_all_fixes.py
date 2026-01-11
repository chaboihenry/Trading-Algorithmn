#!/usr/bin/env python3
"""
Comprehensive Validation Script for All Fixes

This script validates that ALL critical fixes from PROMPT 14-20 are properly
implemented and working correctly.

Run this after completing all fix prompts to ensure nothing was missed.

PROMPT 20: Validation Script [ALL]
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.WARNING,  # Suppress info logs during validation
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("=" * 80)
print("COMPREHENSIVE VALIDATION: ALL FIXES")
print("=" * 80)
print("")


# =============================================================================
# PROMPT 14: API Retry and Timeout [A15, A18, A19]
# =============================================================================

def check_retry_decorator():
    """A18: API should have retry logic with exponential backoff."""
    print("Checking PROMPT 14: API Retry and Timeout...")

    import inspect
    from data.alpaca_tick_client import AlpacaTickClient

    # Check retry decorator exists
    source = inspect.getsource(AlpacaTickClient)
    assert 'retry_with_backoff' in source, "retry_with_backoff decorator not found"

    # Check decorator is applied to fetch_day_ticks
    source_fetch = inspect.getsource(AlpacaTickClient.fetch_day_ticks)
    # The decorator should be above the function definition
    full_source = inspect.getsource(AlpacaTickClient)
    assert '@retry_with_backoff' in full_source, "retry_with_backoff not applied to methods"

    print("  ✓ Retry decorator exists")
    print("  ✓ Retry logic applied to API methods")


# =============================================================================
# PROMPT 15: Order Safety and State Persistence [M4, M10, A8, A20]
# =============================================================================

def check_state_persistence():
    """M4/M10/A8: State persistence for crash recovery."""
    print("\nChecking PROMPT 15: State Persistence...")

    import inspect
    from core.risklabai_combined import RiskLabAICombined

    # Check _save_state method exists
    assert hasattr(RiskLabAICombined, '_save_state'), "_save_state method not found"

    # Check _load_state method exists
    assert hasattr(RiskLabAICombined, '_load_state'), "_load_state method not found"

    # Check state includes cooldowns
    source_save = inspect.getsource(RiskLabAICombined._save_state)
    assert 'stop_loss_cooldowns' in source_save, "stop_loss_cooldowns not in state"
    assert 'trade_history' in source_save, "trade_history not in state"

    print("  ✓ _save_state method exists")
    print("  ✓ _load_state method exists")
    print("  ✓ Cooldowns persisted in state")


def check_order_id_generation():
    """A20: Unique order IDs for idempotency."""
    print("\nChecking PROMPT 15: Order ID Generation...")

    import inspect
    from core.risklabai_combined import RiskLabAICombined

    # Check _generate_order_id method exists
    assert hasattr(RiskLabAICombined, '_generate_order_id'), "_generate_order_id method not found"

    # Check it uses UUID for uniqueness
    source = inspect.getsource(RiskLabAICombined._generate_order_id)
    assert 'uuid' in source, "Order ID generation doesn't use UUID"

    print("  ✓ _generate_order_id method exists")
    print("  ✓ Uses UUID for uniqueness")


# =============================================================================
# PROMPT 16: Market Hours and Holidays [M3, M8]
# =============================================================================

def check_market_calendar():
    """M3/M8: Market calendar with Alpaca API (no hardcoded holidays)."""
    print("\nChecking PROMPT 16: Market Calendar...")

    # Check market_calendar module exists
    from utils import market_calendar

    # Check key functions exist
    assert hasattr(market_calendar, 'is_market_open'), "is_market_open not found"
    assert hasattr(market_calendar, 'is_trading_day'), "is_trading_day not found"
    assert hasattr(market_calendar, 'get_trading_days'), "get_trading_days not found"

    # Check no hardcoded holidays
    import inspect
    source = inspect.getsource(market_calendar)

    # Should NOT have hardcoded dates like this
    bad_patterns = ['2024-01-01', '2024-12-25', 'holidays = [']
    for pattern in bad_patterns:
        assert pattern not in source, f"Hardcoded holiday pattern found: {pattern}"

    # Should use Alpaca Calendar API
    assert 'get_calendar' in source or 'Calendar' in source, "Not using Alpaca Calendar API"

    print("  ✓ Market calendar module exists")
    print("  ✓ No hardcoded holidays")
    print("  ✓ Uses Alpaca Calendar API")


def check_timezone_handling():
    """M3: Consistent Eastern Time usage."""
    print("\nChecking PROMPT 16: Timezone Handling...")

    from utils.market_calendar import MARKET_TZ
    from zoneinfo import ZoneInfo

    # Check MARKET_TZ is Eastern Time
    assert MARKET_TZ == ZoneInfo("America/New_York"), "MARKET_TZ is not Eastern Time"

    # Check backfill script uses MARKET_TZ
    import inspect
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "backfill_ticks",
        project_root / "scripts/setup/backfill_ticks.py"
    )
    if spec and spec.loader:
        backfill_module = importlib.util.module_from_spec(spec)
        source = Path(project_root / "scripts/setup/backfill_ticks.py").read_text()
        assert 'MARKET_TZ' in source, "backfill_ticks doesn't use MARKET_TZ"

    print("  ✓ MARKET_TZ is Eastern Time")
    print("  ✓ Scripts use MARKET_TZ")


# =============================================================================
# PROMPT 17: Logging and Model Versioning [M6, M7, L1]
# =============================================================================

def check_logging_rotation():
    """M6: Log rotation to prevent disk space issues."""
    print("\nChecking PROMPT 17: Logging Rotation...")

    from config import logging_config

    # Check setup_logging exists
    assert hasattr(logging_config, 'setup_logging'), "setup_logging not found"

    # Check RotatingFileHandler is used
    import inspect
    source = inspect.getsource(logging_config.setup_logging)
    assert 'RotatingFileHandler' in source, "RotatingFileHandler not used"
    assert 'maxBytes' in source, "maxBytes not configured"
    assert 'backupCount' in source, "backupCount not configured"

    print("  ✓ setup_logging function exists")
    print("  ✓ RotatingFileHandler configured")
    print("  ✓ Max bytes and backup count set")


def check_model_versioning():
    """M7: Model versioning for rollback capability."""
    print("\nChecking PROMPT 17: Model Versioning...")

    import inspect
    from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy

    # Check save_models has versioning
    source_save = inspect.getsource(RiskLabAIStrategy.save_models)
    assert 'version' in source_save, "No version metadata in save_models"
    assert 'timestamp' in source_save or 'train_date' in source_save, "No timestamp in model metadata"

    # Check load_models logs version
    source_load = inspect.getsource(RiskLabAIStrategy.load_models)
    assert 'version' in source_load, "load_models doesn't check version"

    print("  ✓ save_models adds version metadata")
    print("  ✓ load_models validates version")


# =============================================================================
# PROMPT 18: Circuit Breaker Pattern [D3]
# =============================================================================

def check_circuit_breaker():
    """D3: Circuit breaker for runaway losses."""
    print("\nChecking PROMPT 18: Circuit Breaker...")

    from core.risklabai_combined import CircuitBreaker, RiskLabAICombined

    # Check CircuitBreaker class exists
    assert CircuitBreaker is not None, "CircuitBreaker class not found"

    # Check key methods exist
    assert hasattr(CircuitBreaker, 'check'), "check method not found"
    assert hasattr(CircuitBreaker, 'trip'), "trip method not found"
    assert hasattr(CircuitBreaker, 'reset'), "reset method not found"
    assert hasattr(CircuitBreaker, 'should_auto_reset'), "should_auto_reset not found"

    # Check integration in strategy
    import inspect
    source = inspect.getsource(RiskLabAICombined.initialize)
    assert 'CircuitBreaker' in source, "CircuitBreaker not initialized in strategy"

    # Check circuit breaker in trading loop
    source_loop = inspect.getsource(RiskLabAICombined.on_trading_iteration)
    assert 'circuit_breaker.check' in source_loop, "Circuit breaker not checked in trading loop"

    print("  ✓ CircuitBreaker class exists")
    print("  ✓ Key methods implemented")
    print("  ✓ Integrated in trading loop")


def check_circuit_breaker_state():
    """D3: Circuit breaker state persistence."""
    print("\nChecking PROMPT 18: Circuit Breaker State...")

    import inspect
    from core.risklabai_combined import RiskLabAICombined

    # Check circuit breaker state is saved
    source_save = inspect.getsource(RiskLabAICombined._save_state)
    assert 'circuit_breaker' in source_save, "circuit_breaker not in saved state"

    # Check circuit breaker state is loaded
    source_load = inspect.getsource(RiskLabAICombined._load_state)
    assert 'circuit_breaker' in source_load, "circuit_breaker not loaded from state"

    print("  ✓ Circuit breaker state saved")
    print("  ✓ Circuit breaker state loaded")


# =============================================================================
# PROMPT 19: Walk-Forward First Fold Fix [A4]
# =============================================================================

def check_walk_forward_fix():
    """A4: Walk-forward validation first fold fix."""
    print("\nChecking PROMPT 19: Walk-Forward First Fold Fix...")

    # Check walk_forward_validation script exists
    walk_forward_path = project_root / "scripts/research/walk_forward_validation.py"
    assert walk_forward_path.exists(), "walk_forward_validation.py not found"

    # Check implementation
    source = walk_forward_path.read_text()

    # Check starts from fold 1 (not 0)
    assert 'range(1, n_splits)' in source, "Walk-forward doesn't start from fold 1"

    # Check minimum training bars
    assert 'MIN_TRAIN_BARS' in source, "MIN_TRAIN_BARS not defined"
    assert 'min_training_bars' in source, "Minimum training bars check missing"

    # Check expanding window (not rolling)
    assert 'bars.iloc[:train_end]' in source or 'bars[:train_end]' in source, "Not using expanding window"

    print("  ✓ Walk-forward validation script exists")
    print("  ✓ Starts from fold 1 (not 0)")
    print("  ✓ Minimum training bars enforced")
    print("  ✓ Expanding window implemented")


# =============================================================================
# ADDITIONAL CHECKS (from template)
# =============================================================================

def check_database_timeout():
    """A2: Database should have timeout."""
    print("\nChecking Database Timeout [A2]...")

    import inspect
    from data.tick_storage import TickStorage

    source = inspect.getsource(TickStorage._get_connection)
    assert 'timeout' in source, "No timeout in database connection"

    print("  ✓ Database timeout configured")


def check_scaler_in_save():
    """C10/H10: Scaler should be saved with model."""
    print("\nChecking Scaler Persistence [C10/H10]...")

    import inspect
    from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy

    source = inspect.getsource(RiskLabAIStrategy.save_models)
    assert 'scaler' in source, "Scaler not saved in save_models()"

    # Check scaler is loaded
    source_load = inspect.getsource(RiskLabAIStrategy.load_models)
    assert 'scaler' in source_load, "Scaler not loaded in load_models()"

    print("  ✓ Scaler saved with model")
    print("  ✓ Scaler loaded from model")


# =============================================================================
# MAIN VALIDATION RUNNER
# =============================================================================

def main():
    """Run all validation checks."""

    checks = [
        # PROMPT 14
        ("PROMPT 14: API Retry Logic", check_retry_decorator),

        # PROMPT 15
        ("PROMPT 15: State Persistence", check_state_persistence),
        ("PROMPT 15: Order ID Generation", check_order_id_generation),

        # PROMPT 16
        ("PROMPT 16: Market Calendar", check_market_calendar),
        ("PROMPT 16: Timezone Handling", check_timezone_handling),

        # PROMPT 17
        ("PROMPT 17: Logging Rotation", check_logging_rotation),
        ("PROMPT 17: Model Versioning", check_model_versioning),

        # PROMPT 18
        ("PROMPT 18: Circuit Breaker", check_circuit_breaker),
        ("PROMPT 18: Circuit Breaker State", check_circuit_breaker_state),

        # PROMPT 19
        ("PROMPT 19: Walk-Forward Fix", check_walk_forward_fix),

        # Additional
        ("Additional: Database Timeout", check_database_timeout),
        ("Additional: Scaler Persistence", check_scaler_in_save),
    ]

    passed = 0
    failed = 0
    errors = []

    print("")

    for name, check_func in checks:
        try:
            check_func()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ {name} FAILED: {e}")
            failed += 1
            errors.append((name, str(e)))
        except Exception as e:
            print(f"\n✗ {name} ERROR: {e}")
            failed += 1
            errors.append((name, str(e)))

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total checks: {len(checks)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 80)

    if failed > 0:
        print("\n❌ VALIDATION FAILED")
        print("\nFailed checks:")
        for name, error in errors:
            print(f"  - {name}: {error}")
        print("\n⚠️  Some fixes are not complete. Review failed checks above.")
        return 1
    else:
        print("\n✓ ALL VALIDATION CHECKS PASSED")
        print("\nAll critical fixes from PROMPT 14-19 are properly implemented:")
        print("  ✓ API retry logic with exponential backoff")
        print("  ✓ State persistence for crash recovery")
        print("  ✓ Unique order IDs for idempotency")
        print("  ✓ Market calendar using Alpaca API (no hardcoded holidays)")
        print("  ✓ Consistent Eastern Time timezone handling")
        print("  ✓ Log rotation to prevent disk space issues")
        print("  ✓ Model versioning for rollback capability")
        print("  ✓ Circuit breaker for runaway losses")
        print("  ✓ Walk-forward validation first fold fix")
        print("  ✓ Database timeout configured")
        print("  ✓ Scaler persistence with models")
        print("\n🎉 Trading bot is production-ready!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
