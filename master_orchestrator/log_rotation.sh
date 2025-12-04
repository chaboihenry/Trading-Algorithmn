#!/bin/bash
# Log Rotation Script for Trading Automation
# Rotates logs when they exceed 10MB and keeps last 5 versions
# Run this via launchd or manually before daily pipeline

LOG_DIR="/tmp"
MAX_SIZE_MB=10
KEEP_VERSIONS=5

# Function to rotate a log file
rotate_log() {
    local logfile=$1
    local max_size_bytes=$((MAX_SIZE_MB * 1024 * 1024))

    if [ ! -f "$logfile" ]; then
        return
    fi

    local size=$(stat -f%z "$logfile" 2>/dev/null || echo 0)

    if [ "$size" -gt "$max_size_bytes" ]; then
        echo "Rotating $logfile (size: $(($size / 1024 / 1024))MB)"

        # Remove oldest backup if exists
        if [ -f "${logfile}.${KEEP_VERSIONS}" ]; then
            rm -f "${logfile}.${KEEP_VERSIONS}"
        fi

        # Shift existing backups
        for i in $(seq $((KEEP_VERSIONS - 1)) -1 1); do
            if [ -f "${logfile}.$i" ]; then
                mv "${logfile}.$i" "${logfile}.$((i + 1))"
            fi
        done

        # Rotate current log
        mv "$logfile" "${logfile}.1"
        touch "$logfile"

        echo "✓ Rotated $logfile"
    fi
}

# Rotate all trading-related logs
echo "=================================================================================="
echo "LOG ROTATION - $(date)"
echo "=================================================================================="

rotate_log "${LOG_DIR}/trading_daily.log"
rotate_log "${LOG_DIR}/trading_daily_error.log"
rotate_log "${LOG_DIR}/trading_intraday.log"
rotate_log "${LOG_DIR}/trading_intraday_error.log"
rotate_log "${LOG_DIR}/orchestrator.log"
rotate_log "${LOG_DIR}/daily_runner.log"

echo "=================================================================================="
echo "Log rotation complete"
echo "=================================================================================="
