#!/Users/henry/miniconda3/envs/trading/bin/python
"""
Wait for External Drive and Run Intraday Pipeline Continuously (Python Version)

This script:
1. Waits for /Volumes/Vault to be mounted (up to 2 hours)
2. Runs intraday_runner.py in continuous mode
3. Runs from 9:30 AM until market close (4:00 PM ET)
"""

import os
import sys
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path


DB_PATH = "/Volumes/Vault/85_assets_prediction.db"
DRIVE_PATH = "/Volumes/Vault"
MAX_WAIT_HOURS = 2
CHECK_INTERVAL_SECONDS = 30
PYTHON_BIN = "/Users/henry/miniconda3/envs/trading/bin/python"
PROJECT_ROOT = "/Users/henry/Desktop/Programming & Networking/Personal Projects/Integrated Trading Agent"


def is_drive_mounted() -> bool:
    """Check if the external drive is mounted"""
    return os.path.exists(DRIVE_PATH) and os.path.ismount(DRIVE_PATH)


def is_database_accessible() -> bool:
    """Check if the database file is accessible"""
    if not is_drive_mounted():
        return False

    db_file = Path(DB_PATH)
    return db_file.exists() and os.access(DB_PATH, os.R_OK | os.W_OK)


def wait_for_drive(max_wait_hours: float = MAX_WAIT_HOURS) -> bool:
    """Wait for the external drive to be mounted"""
    print("="*80)
    print(f"🔍 Checking for external drive: {DRIVE_PATH}")
    print("="*80)

    if is_database_accessible():
        print(f"✅ Drive already mounted and database accessible!")
        return True

    print(f"⏳ Drive not detected. Waiting up to {max_wait_hours} hours...")
    print(f"   Checking every {CHECK_INTERVAL_SECONDS} seconds...")

    start_time = datetime.now()
    max_wait = timedelta(hours=max_wait_hours)
    check_count = 0

    while datetime.now() - start_time < max_wait:
        check_count += 1
        elapsed = datetime.now() - start_time
        elapsed_str = f"{int(elapsed.total_seconds() / 60)}m {int(elapsed.total_seconds() % 60)}s"

        print(f"\n[Check #{check_count}] Elapsed: {elapsed_str}")

        if is_drive_mounted():
            print(f"   📁 Drive mounted at {DRIVE_PATH}")
            print(f"   ⏸️  Waiting 5s for filesystem to settle...")
            time.sleep(5)

            if is_database_accessible():
                print(f"   ✅ Database accessible at {DB_PATH}")
                print(f"\n{'='*80}")
                print(f"✅ READY - Drive mounted and database accessible!")
                print(f"{'='*80}\n")
                return True
            else:
                print(f"   ⚠️  Drive mounted but database not accessible yet...")
        else:
            print(f"   ⏳ Drive not mounted yet...")

        time.sleep(CHECK_INTERVAL_SECONDS)

    print(f"\n{'='*80}")
    print(f"❌ TIMEOUT - Drive not mounted after {max_wait_hours} hours")
    print(f"{'='*80}\n")
    return False


def main():
    """Main entry point"""
    # Check/wait for drive
    if not wait_for_drive():
        print(f"❌ Drive not mounted. Exiting.")
        sys.exit(1)

    # Change to project root
    os.chdir(PROJECT_ROOT)

    # Run intraday pipeline in continuous mode
    print("="*80)
    print("🔄 Starting Intraday Pipeline (Continuous Mode)")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Database: {DB_PATH}")
    print(f"Mode: Continuous (9:30 AM - 4:00 PM ET)")
    print(f"Check Interval: 30 seconds")
    print("="*80)
    print()

    intraday_runner = Path(PROJECT_ROOT) / "master_orchestrator" / "intraday_runner.py"

    cmd = [
        PYTHON_BIN,
        str(intraday_runner),
        "--db", DB_PATH,
        "--interval", "30"
    ]

    try:
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    except Exception as e:
        print(f"❌ Error running intraday pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
