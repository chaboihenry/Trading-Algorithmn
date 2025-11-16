#!/bin/bash
#
# Wait for External Drive and Run Intraday Pipeline Continuously
#
# This script:
# 1. Waits for /Volumes/Vault to be mounted (up to 2 hours)
# 2. Runs intraday_runner.py in continuous mode
# 3. Runs from 9:30 AM until market close (4:00 PM ET)
#

set -e

PYTHON_BIN="/Users/henry/miniconda3/envs/trading/bin/python"
PROJECT_ROOT="/Users/henry/Desktop/Programming & Networking/Personal Projects/Integrated Trading Agent"
DRIVE_PATH="/Volumes/Vault"
DB_PATH="/Volumes/Vault/85_assets_prediction.db"
MAX_WAIT_HOURS=2
CHECK_INTERVAL_SECONDS=30

echo "================================================================================"
echo "🔍 Checking for external drive: ${DRIVE_PATH}"
echo "================================================================================"

# Function to check if drive is mounted and database is accessible
is_ready() {
    if [ -d "${DRIVE_PATH}" ] && [ -f "${DB_PATH}" ] && [ -r "${DB_PATH}" ] && [ -w "${DB_PATH}" ]; then
        return 0
    else
        return 1
    fi
}

# Check if already ready
if is_ready; then
    echo "✅ Drive already mounted and database accessible!"
else
    echo "⏳ Drive not detected. Waiting up to ${MAX_WAIT_HOURS} hours..."
    echo "   Checking every ${CHECK_INTERVAL_SECONDS} seconds..."

    START_TIME=$(date +%s)
    MAX_WAIT_SECONDS=$((MAX_WAIT_HOURS * 3600))
    CHECK_COUNT=0

    while true; do
        CURRENT_TIME=$(date +%s)
        ELAPSED=$((CURRENT_TIME - START_TIME))

        if [ $ELAPSED -ge $MAX_WAIT_SECONDS ]; then
            echo ""
            echo "================================================================================"
            echo "❌ TIMEOUT - Drive not mounted after ${MAX_WAIT_HOURS} hours"
            echo "================================================================================"
            exit 1
        fi

        CHECK_COUNT=$((CHECK_COUNT + 1))
        ELAPSED_MIN=$((ELAPSED / 60))
        ELAPSED_SEC=$((ELAPSED % 60))

        echo ""
        echo "[Check #${CHECK_COUNT}] Elapsed: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"

        if [ -d "${DRIVE_PATH}" ]; then
            echo "   📁 Drive mounted at ${DRIVE_PATH}"
            echo "   ⏸️  Waiting 5s for filesystem to settle..."
            sleep 5

            if is_ready; then
                echo "   ✅ Database accessible at ${DB_PATH}"
                echo ""
                echo "================================================================================"
                echo "✅ READY - Drive mounted and database accessible!"
                echo "================================================================================"
                break
            else
                echo "   ⚠️  Drive mounted but database not accessible yet..."
            fi
        else
            echo "   ⏳ Drive not mounted yet..."
        fi

        sleep ${CHECK_INTERVAL_SECONDS}
    done
fi

echo ""
echo "================================================================================"
echo "🔄 Starting Intraday Pipeline (Continuous Mode)"
echo "================================================================================"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Database: ${DB_PATH}"
echo "Mode: Continuous (9:30 AM - 4:00 PM ET)"
echo "Check Interval: 30 seconds"
echo "================================================================================"
echo ""

# Change to project root
cd "${PROJECT_ROOT}"

# Run intraday pipeline in continuous mode
exec "${PYTHON_BIN}" "${PROJECT_ROOT}/master_orchestrator/intraday_runner.py" \
    --db "${DB_PATH}" \
    --interval 30
