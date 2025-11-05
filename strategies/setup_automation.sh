#!/bin/bash

###############################################################################
# Setup Script for Automated Daily Pairs Trading
###############################################################################
# This script sets up automated daily execution of the pairs trading system
# using macOS LaunchAgent or provides instructions for cron setup
###############################################################################

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_PATH="/Users/henry/miniconda3/envs/trading/bin/python"
TRADING_SCRIPT="$SCRIPT_DIR/run_daily_trading.py"
LOG_DIR="$SCRIPT_DIR/logs"

# Create logs directory
mkdir -p "$LOG_DIR"

echo "========================================"
echo "Pairs Trading Automation Setup"
echo "========================================"
echo ""

# Check if Python environment exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "ERROR: Python environment not found at $PYTHON_PATH"
    echo "Please update PYTHON_PATH in this script"
    exit 1
fi

# Check if trading script exists
if [ ! -f "$TRADING_SCRIPT" ]; then
    echo "ERROR: Trading script not found at $TRADING_SCRIPT"
    exit 1
fi

echo "Python Environment: $PYTHON_PATH"
echo "Trading Script: $TRADING_SCRIPT"
echo "Log Directory: $LOG_DIR"
echo ""

# Install required Python package
echo "Installing required Python package (schedule)..."
$PYTHON_PATH -m pip install schedule --quiet

echo ""
echo "========================================"
echo "Setup Options"
echo "========================================"
echo ""
echo "Choose your automation method:"
echo ""
echo "1. macOS LaunchAgent (recommended for macOS)"
echo "   - Runs automatically at specified time daily"
echo "   - Survives reboots"
echo "   - Easy to enable/disable"
echo ""
echo "2. cron (Unix/Linux/macOS)"
echo "   - Traditional Unix scheduling"
echo "   - Simple and reliable"
echo ""
echo "3. Manual (run the scheduler yourself)"
echo "   - You control when it runs"
echo "   - Good for testing"
echo ""

read -p "Enter your choice (1/2/3): " choice

case $choice in
    1)
        # macOS LaunchAgent
        PLIST_NAME="com.pairstrading.daily"
        PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_NAME}.plist"

        echo ""
        read -p "What time should trading run daily? (format HH:MM, e.g., 09:30): " run_time

        # Parse time
        IFS=':' read -r hour minute <<< "$run_time"

        # Create LaunchAgent plist
        cat > "$PLIST_PATH" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_NAME}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON_PATH}</string>
        <string>${TRADING_SCRIPT}</string>
        <string>--run-once</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>${hour}</integer>
        <key>Minute</key>
        <integer>${minute}</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/daily_trading.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/daily_trading_error.log</string>
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
EOF

        # Load the LaunchAgent
        launchctl unload "$PLIST_PATH" 2>/dev/null
        launchctl load "$PLIST_PATH"

        echo ""
        echo "✓ LaunchAgent installed successfully!"
        echo ""
        echo "The trading system will run daily at ${run_time}"
        echo ""
        echo "Management commands:"
        echo "  Stop:    launchctl unload $PLIST_PATH"
        echo "  Start:   launchctl load $PLIST_PATH"
        echo "  Remove:  launchctl unload $PLIST_PATH && rm $PLIST_PATH"
        echo ""
        echo "Logs location: $LOG_DIR/"
        ;;

    2)
        # cron setup
        echo ""
        read -p "What time should trading run daily? (format HH:MM, e.g., 09:30): " run_time

        # Parse time
        IFS=':' read -r hour minute <<< "$run_time"

        CRON_COMMAND="$minute $hour * * * cd $SCRIPT_DIR && $PYTHON_PATH $TRADING_SCRIPT --run-once >> $LOG_DIR/daily_trading.log 2>&1"

        echo ""
        echo "Add this line to your crontab:"
        echo ""
        echo "$CRON_COMMAND"
        echo ""
        read -p "Would you like me to add it automatically? (y/n): " add_cron

        if [ "$add_cron" = "y" ] || [ "$add_cron" = "Y" ]; then
            (crontab -l 2>/dev/null; echo "$CRON_COMMAND") | crontab -
            echo "✓ Cron job added successfully!"
            echo ""
            echo "View crontab: crontab -l"
            echo "Edit crontab: crontab -e"
            echo "Remove crontab: crontab -r"
        else
            echo "Run 'crontab -e' and add the line manually"
        fi
        ;;

    3)
        # Manual instructions
        echo ""
        echo "Manual Execution Instructions"
        echo "=============================="
        echo ""
        echo "Run once immediately:"
        echo "  cd $SCRIPT_DIR"
        echo "  $PYTHON_PATH run_daily_trading.py --run-once"
        echo ""
        echo "Run continuously (checks daily at 9:30 AM):"
        echo "  cd $SCRIPT_DIR"
        echo "  $PYTHON_PATH run_daily_trading.py --time 09:30"
        echo ""
        echo "View dashboard:"
        echo "  cd $SCRIPT_DIR"
        echo "  $PYTHON_PATH portfolio_dashboard.py"
        ;;

    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Monitor logs at: $LOG_DIR/"
echo "2. Check portfolio: $PYTHON_PATH $SCRIPT_DIR/portfolio_dashboard.py"
echo "3. Manual run: $PYTHON_PATH $TRADING_SCRIPT --run-once"
echo ""