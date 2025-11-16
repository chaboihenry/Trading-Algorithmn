#!/bin/bash
#
# Install LaunchAgents for Automated Trading System
#
# This script installs macOS LaunchAgents that will automatically
# run the trading system Monday-Friday at 9:30 AM ET.
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"

echo "================================================================================"
echo "Installing Trading System LaunchAgents"
echo "================================================================================"
echo ""

# Create LaunchAgents directory if it doesn't exist
if [ ! -d "$LAUNCH_AGENTS_DIR" ]; then
    echo "Creating LaunchAgents directory..."
    mkdir -p "$LAUNCH_AGENTS_DIR"
fi

# Make wrapper script executable
chmod +x "$SCRIPT_DIR/wait_and_run_intraday.sh"

# Copy plist files
echo "Installing LaunchAgent files..."
cp "$SCRIPT_DIR/com.trading.daily.plist" "$LAUNCH_AGENTS_DIR/"
cp "$SCRIPT_DIR/com.trading.intraday.plist" "$LAUNCH_AGENTS_DIR/"

echo "✅ Copied plist files to $LAUNCH_AGENTS_DIR"
echo "✅ Wrapper scripts are executable"
echo ""

# Load the LaunchAgents
echo "Loading LaunchAgents..."
launchctl load "$LAUNCH_AGENTS_DIR/com.trading.daily.plist" 2>/dev/null || true
launchctl load "$LAUNCH_AGENTS_DIR/com.trading.intraday.plist" 2>/dev/null || true

echo "✅ LaunchAgents loaded"
echo ""

# Show status
echo "================================================================================"
echo "Installation Complete!"
echo "================================================================================"
echo ""
echo "The trading system will now run automatically:"
echo "  • Every day at 9:30 AM ET (7 days/week for 24/7 crypto data)"
echo "  • Waits up to 2 hours for external drive to be connected"
echo ""
echo "Logs:"
echo "  • Daily:    /tmp/trading_daily.log"
echo "  • Intraday: /tmp/trading_intraday.log"
echo ""
echo "Management Commands:"
echo "  • Check status:  launchctl list | grep trading"
echo "  • View logs:     tail -f /tmp/trading_daily.log"
echo "  • Unload:        launchctl unload ~/Library/LaunchAgents/com.trading.*.plist"
echo "  • Reload:        launchctl load ~/Library/LaunchAgents/com.trading.*.plist"
echo ""
echo "================================================================================"
