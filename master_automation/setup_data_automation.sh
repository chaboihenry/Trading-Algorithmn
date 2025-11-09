#!/bin/bash

###############################################################################
# Setup Script for Automated Daily Database Updates
###############################################################################
# This script sets up automated daily execution of database updates
# at 9:30 AM using macOS LaunchAgent or cron
###############################################################################

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_PATH="/Users/henry/miniconda3/envs/trading/bin/python"
UPDATE_SCRIPT="$SCRIPT_DIR/update_database.py"
LOG_DIR="$SCRIPT_DIR/logs"

# Create logs directory
mkdir -p "$LOG_DIR"

echo "========================================"
echo "Database Update Automation Setup"
echo "========================================"
echo ""

# Check if Python environment exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "ERROR: Python environment not found at $PYTHON_PATH"
    echo "Please update PYTHON_PATH in this script"
    exit 1
fi

# Check if update script exists
if [ ! -f "$UPDATE_SCRIPT" ]; then
    echo "ERROR: Update script not found at $UPDATE_SCRIPT"
    exit 1
fi

echo "Python Environment: $PYTHON_PATH"
echo "Update Script: $UPDATE_SCRIPT"
echo "Log Directory: $LOG_DIR"
echo ""

echo "========================================"
echo "Setup Options"
echo "========================================"
echo ""
echo "Choose your automation method:"
echo ""
echo "1. macOS LaunchAgent (recommended for macOS)"
echo "   - Runs automatically at 9:30 AM daily"
echo "   - Waits for external drive if disconnected"
echo "   - Survives reboots"
echo ""
echo "2. cron (Unix/Linux/macOS)"
echo "   - Traditional Unix scheduling"
echo "   - Simple and reliable"
echo ""
echo "3. Manual (run the script yourself)"
echo "   - You control when it runs"
echo "   - Good for testing"
echo ""

read -p "Enter your choice (1/2/3): " choice

case $choice in
    1)
        # macOS LaunchAgent
        PLIST_NAME="com.tradingalgo.dataupdate"
        PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_NAME}.plist"

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
        <string>${UPDATE_SCRIPT}</string>
        <string>--wait-for-drive</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>9</integer>
        <key>Minute</key>
        <integer>30</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/data_update.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/data_update_error.log</string>
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
        echo "Database will update daily at 9:30 AM"
        echo ""
        echo "Management commands:"
        echo "  Stop:    launchctl unload $PLIST_PATH"
        echo "  Start:   launchctl load $PLIST_PATH"
        echo "  Remove:  launchctl unload $PLIST_PATH && rm $PLIST_PATH"
        echo ""
        echo "Logs location: $LOG_DIR/"
        echo ""
        echo "⚠️  IMPORTANT: Script will wait up to 60 minutes for external drive"
        ;;

    2)
        # cron setup
        CRON_COMMAND="30 9 * * * cd $SCRIPT_DIR && $PYTHON_PATH $UPDATE_SCRIPT --wait-for-drive >> $LOG_DIR/data_update.log 2>&1"

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
        echo "Run update immediately (fail if drive not connected):"
        echo "  cd $SCRIPT_DIR"
        echo "  $PYTHON_PATH update_database.py"
        echo ""
        echo "Run update with drive wait (wait up to 60 min for drive):"
        echo "  cd $SCRIPT_DIR"
        echo "  $PYTHON_PATH update_database.py --wait-for-drive"
        echo ""
        echo "View logs:"
        echo "  tail -f $LOG_DIR/database_update.log"
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
echo "1. Test manual run: $PYTHON_PATH $UPDATE_SCRIPT"
echo "2. Check logs at: $LOG_DIR/"
echo "3. Automation will run daily at 9:30 AM"
echo ""