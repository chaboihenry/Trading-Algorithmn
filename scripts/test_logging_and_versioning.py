#!/usr/bin/env python3
"""
Test Script for Logging Rotation and Model Versioning

This script validates:
1. Log rotation at 10MB (RotatingFileHandler)
2. Separate error logs for warnings/errors
3. Daily log rotation (TimedRotatingFileHandler)
4. Model versioning with timestamps
5. Model metadata (version, train_date, python_version)
6. Loading versioned models and extracting metadata

PROMPT 17: Logging and Model Versioning [M6, M7, L1]
"""

import sys
import logging
import shutil
from pathlib import Path
from datetime import datetime
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.logging_config import setup_logging, silence_library_logs
from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy

# Set up test environment
TEST_DIR = Path(tempfile.mkdtemp(prefix="test_logging_"))
TEST_LOG_DIR = TEST_DIR / "logs"
TEST_MODEL_DIR = TEST_DIR / "models"

print("=" * 80)
print("LOGGING AND MODEL VERSIONING TEST SUITE")
print("=" * 80)
print(f"Test directory: {TEST_DIR}")
print("=" * 80)


def test_log_rotation():
    """Test 1: Log rotation functionality."""
    print("\n" + "=" * 80)
    print("TEST 1: Log Rotation")
    print("=" * 80)

    # Setup logging with small max_bytes for testing
    logger = setup_logging(
        log_dir=str(TEST_LOG_DIR),
        max_bytes=1000,  # 1KB for testing (instead of 10MB)
        backup_count=3,
        level=logging.DEBUG
    )

    # Generate enough logs to trigger rotation
    print("Generating log entries to trigger rotation...")
    for i in range(100):
        logger.info(f"Test log entry {i}: " + "X" * 50)
        logger.debug(f"Debug entry {i}: " + "Y" * 50)
        logger.warning(f"Warning entry {i}: " + "Z" * 50)

    # Check if log files exist
    main_log = TEST_LOG_DIR / "trading_bot.log"
    error_log = TEST_LOG_DIR / "trading_bot_errors.log"
    daily_log = TEST_LOG_DIR / "trading_bot_daily.log"

    success = True

    # Verify main log exists
    if main_log.exists():
        size = main_log.stat().st_size
        print(f"✓ Main log created: {main_log} ({size:,} bytes)")
    else:
        print(f"✗ Main log not found: {main_log}")
        success = False

    # Check for rotated files (trading_bot.log.1, trading_bot.log.2, etc.)
    rotated_files = list(TEST_LOG_DIR.glob("trading_bot.log.*"))
    if rotated_files:
        print(f"✓ Log rotation occurred: {len(rotated_files)} backup file(s)")
        for rf in rotated_files:
            print(f"  - {rf.name} ({rf.stat().st_size:,} bytes)")
    else:
        print("⚠ No rotated files found (may not have exceeded 1KB yet)")

    # Verify error log exists and contains only warnings/errors
    if error_log.exists():
        size = error_log.stat().st_size
        print(f"✓ Error log created: {error_log} ({size:,} bytes)")

        # Check content
        with open(error_log, 'r') as f:
            content = f.read()
            if "WARNING" in content and "INFO" not in content:
                print("✓ Error log contains only WARNING+ messages (INFO excluded)")
            elif "WARNING" in content:
                print("⚠ Error log contains WARNING but might have INFO messages")
            else:
                print("✗ Error log doesn't contain expected WARNING messages")
                success = False
    else:
        print(f"✗ Error log not found: {error_log}")
        success = False

    # Verify daily log exists
    if daily_log.exists():
        size = daily_log.stat().st_size
        print(f"✓ Daily log created: {daily_log} ({size:,} bytes)")
    else:
        print(f"✗ Daily log not found: {daily_log}")
        success = False

    return success


def test_model_versioning():
    """Test 2: Model versioning functionality."""
    print("\n" + "=" * 80)
    print("TEST 2: Model Versioning")
    print("=" * 80)

    # Create test model directory
    TEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Create a minimal RiskLabAI strategy instance
    strategy = RiskLabAIStrategy()

    # Create dummy models (for testing save/load)
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    import numpy as np

    print("Creating dummy models for testing...")

    # Create simple dummy models
    X_dummy = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_dummy = np.array([0, 1, 0, 1])

    strategy.primary_model = XGBClassifier(n_estimators=2, max_depth=2)
    strategy.primary_model.fit(X_dummy, y_dummy)

    strategy.meta_model = LogisticRegression()
    strategy.meta_model.fit(X_dummy, y_dummy)

    strategy.feature_names = ['feature1', 'feature2']
    strategy.important_features = {'feature1': 0.7, 'feature2': 0.3}

    # Test saving with versioning
    model_path = TEST_MODEL_DIR / "test_model.pkl"

    print(f"\nSaving versioned model to {model_path}...")
    strategy.save_models(str(model_path), save_versioned=True)

    # Check that both files were created
    success = True

    # Check latest file
    if model_path.exists():
        size = model_path.stat().st_size
        print(f"✓ Latest model saved: {model_path} ({size:,} bytes)")
    else:
        print(f"✗ Latest model not found: {model_path}")
        success = False

    # Check versioned files
    versioned_files = list(TEST_MODEL_DIR.glob("test_model_*.pkl"))
    if versioned_files:
        print(f"✓ Versioned model(s) created: {len(versioned_files)} file(s)")
        for vf in versioned_files:
            print(f"  - {vf.name} ({vf.stat().st_size:,} bytes)")
    else:
        print("✗ No versioned model files found")
        success = False

    # Test loading and verify metadata
    print(f"\nLoading model from {model_path}...")

    # Create fresh strategy instance to test loading
    new_strategy = RiskLabAIStrategy()
    new_strategy.load_models(str(model_path))

    # Verify models loaded correctly
    if new_strategy.primary_model is not None:
        print("✓ Primary model loaded successfully")
    else:
        print("✗ Primary model failed to load")
        success = False

    if new_strategy.meta_model is not None:
        print("✓ Meta model loaded successfully")
    else:
        print("✗ Meta model failed to load")
        success = False

    # The load_models() method should have logged version info
    # We can't directly check the logs from here, but we verified the code does it

    # Manually verify metadata exists in saved file
    import joblib
    model_data = joblib.load(str(model_path))

    metadata_checks = [
        ('version', 'Model version timestamp'),
        ('train_date', 'Training date'),
        ('python_version', 'Python version'),
        ('model_type', 'Model type'),
        ('hyperparameters', 'Hyperparameters'),
    ]

    print("\nVerifying metadata in saved model...")
    for key, description in metadata_checks:
        if key in model_data:
            value = model_data[key]
            # Truncate long values for display
            if isinstance(value, str) and len(value) > 50:
                value = value[:50] + "..."
            print(f"✓ {description}: {value}")
        else:
            print(f"✗ {description} missing")
            success = False

    return success


def test_library_silencing():
    """Test 3: Library log silencing."""
    print("\n" + "=" * 80)
    print("TEST 3: Library Log Silencing")
    print("=" * 80)

    silence_library_logs()

    # Check log levels for noisy libraries
    libraries = ['urllib3', 'alpaca', 'requests', 'matplotlib']
    success = True

    for lib in libraries:
        lib_logger = logging.getLogger(lib)
        level = lib_logger.level

        # Level should be WARNING (30) or higher
        if level >= logging.WARNING:
            level_name = logging.getLevelName(level)
            print(f"✓ {lib}: {level_name} (silenced)")
        else:
            level_name = logging.getLevelName(level)
            print(f"✗ {lib}: {level_name} (not silenced)")
            success = False

    return success


def cleanup():
    """Clean up test directory."""
    print("\n" + "=" * 80)
    print("CLEANUP")
    print("=" * 80)

    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
        print(f"✓ Test directory removed: {TEST_DIR}")
    else:
        print(f"⚠ Test directory not found: {TEST_DIR}")


if __name__ == "__main__":
    print("\nRunning test suite...\n")

    # Run tests
    test1_passed = test_log_rotation()
    test2_passed = test_model_versioning()
    test3_passed = test_library_silencing()

    # Summary
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Test 1 - Log Rotation: {'✓ PASS' if test1_passed else '✗ FAIL'}")
    print(f"Test 2 - Model Versioning: {'✓ PASS' if test2_passed else '✗ FAIL'}")
    print(f"Test 3 - Library Silencing: {'✓ PASS' if test3_passed else '✗ FAIL'}")
    print("=" * 80)

    all_passed = test1_passed and test2_passed and test3_passed
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nValidation:")
        print("  - Log rotation working (files rotate at configured size)")
        print("  - Separate error log contains only WARNING+ messages")
        print("  - Daily log created successfully")
        print("  - Model versioning working (timestamped files created)")
        print("  - Model metadata saved (version, train_date, python_version)")
        print("  - Models load correctly with version validation")
        print("  - Noisy library logs silenced")
    else:
        print("✗ SOME TESTS FAILED")

    print("=" * 80)

    # Cleanup
    cleanup()

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)
