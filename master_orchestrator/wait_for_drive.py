#!/Users/henry/miniconda3/envs/trading/bin/python
"""
Wait for External Drive Wrapper

Waits up to 2 hours for the external drive (/Volumes/Vault) to be mounted
before running the trading system. This handles scenarios where the Mac
starts the scheduled pipeline but the drive isn't connected yet.

Usage:
    wait_for_drive.py --daily
    wait_for_drive.py --intraday
    wait_for_drive.py --full
"""

import os
import sys
import time
import subprocess
import argparse
from datetime import datetime, timedelta
from pathlib import Path


DB_PATH = "/Volumes/Vault/85_assets_prediction.db"
DRIVE_PATH = "/Volumes/Vault"
MAX_WAIT_HOURS = 2
CHECK_INTERVAL_SECONDS = 30


def is_drive_mounted() -> bool:
    """Check if the external drive is mounted"""
    return os.path.exists(DRIVE_PATH) and os.path.ismount(DRIVE_PATH)


def is_database_accessible() -> bool:
    """Check if the database file is accessible"""
    if not is_drive_mounted():
        return False

    # Check if database file exists and is readable
    db_file = Path(DB_PATH)
    return db_file.exists() and os.access(DB_PATH, os.R_OK | os.W_OK)


def wait_for_drive(max_wait_hours: float = MAX_WAIT_HOURS) -> bool:
    """
    Wait for the external drive to be mounted

    Args:
        max_wait_hours: Maximum hours to wait (default: 2)

    Returns:
        True if drive is mounted, False if timeout
    """
    print(f"="*80)
    print(f"🔍 Checking for external drive: {DRIVE_PATH}")
    print(f"="*80)

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

            # Wait a few seconds for filesystem to settle
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

        # Sleep before next check
        time.sleep(CHECK_INTERVAL_SECONDS)

    # Timeout
    print(f"\n{'='*80}")
    print(f"❌ TIMEOUT - Drive not mounted after {max_wait_hours} hours")
    print(f"{'='*80}\n")
    return False


def run_orchestrator(mode: str) -> int:
    """
    Run the orchestrator in specified mode

    Args:
        mode: One of 'daily', 'intraday', 'full'

    Returns:
        Exit code from orchestrator
    """
    script_dir = Path(__file__).parent
    orchestrator = script_dir / "orchestrator.py"
    python_bin = "/Users/henry/miniconda3/envs/trading/bin/python"

    print(f"="*80)
    print(f"🚀 Starting Trading System ({mode} mode)")
    print(f"="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Database: {DB_PATH}")
    print(f"="*80 + "\n")

    # Run orchestrator
    cmd = [python_bin, str(orchestrator), f"--{mode}"]

    try:
        result = subprocess.run(cmd)
        return result.returncode
    except Exception as e:
        print(f"❌ Error running orchestrator: {e}")
        return 1


def rotate_logs():
    """Rotate logs if they're too large (>10MB)"""
    script_dir = Path(__file__).parent
    log_rotation_script = script_dir / "log_rotation.sh"

    if log_rotation_script.exists():
        try:
            subprocess.run([str(log_rotation_script)], check=False)
        except Exception as e:
            print(f"⚠️  Log rotation failed (non-critical): {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Wait for external drive, then run trading system'
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--daily', action='store_true',
                           help='Run daily pipeline')
    mode_group.add_argument('--intraday', action='store_true',
                           help='Run intraday pipeline')
    mode_group.add_argument('--full', action='store_true',
                           help='Run full system')

    parser.add_argument('--max-wait-hours', type=float, default=MAX_WAIT_HOURS,
                       help=f'Max hours to wait for drive (default: {MAX_WAIT_HOURS})')
    parser.add_argument('--skip-wait', action='store_true',
                       help='Skip waiting for drive (fail immediately if not mounted)')

    args = parser.parse_args()

    # Rotate logs before starting (prevents log files from growing too large)
    rotate_logs()

    # Determine mode
    if args.daily:
        mode = 'daily'
    elif args.intraday:
        mode = 'intraday'
    else:
        mode = 'full'

    # Check/wait for drive
    if args.skip_wait:
        if not is_database_accessible():
            print(f"❌ Drive not mounted and --skip-wait specified. Exiting.")
            sys.exit(1)
    else:
        if not wait_for_drive(max_wait_hours=args.max_wait_hours):
            print(f"❌ Drive not mounted after {args.max_wait_hours} hours. Exiting.")
            sys.exit(1)

    # Run orchestrator
    exit_code = run_orchestrator(mode)

    print(f"\n{'='*80}")
    if exit_code == 0:
        print(f"✅ Trading system completed successfully")
    else:
        print(f"❌ Trading system completed with errors (exit code: {exit_code})")
    print(f"{'='*80}\n")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
