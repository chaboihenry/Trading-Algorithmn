#!/bin/bash

# Fix macOS socket exhaustion issues for long-running trading bot
# Run this script with sudo to apply system-level fixes

echo "Configuring macOS for long-running trading bot..."
echo "================================================"

# Check current settings
echo ""
echo "Current socket settings:"
sysctl -a | grep -E "net.inet.tcp.tw_reuse|net.inet.tcp.msl|kern.maxfiles|kern.maxfilesperproc"

echo ""
echo "Recommended changes:"
echo "1. Reduce TIME_WAIT duration (tcp.msl)"
echo "2. Enable socket reuse (tcp.tw_reuse)"
echo "3. Increase file descriptor limits"

echo ""
read -p "Apply these changes? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Applying system tuning..."

    # Reduce TIME_WAIT duration (default 15000ms -> 1000ms)
    sudo sysctl -w net.inet.tcp.msl=1000

    # Increase file descriptor limits
    sudo sysctl -w kern.maxfiles=65536
    sudo sysctl -w kern.maxfilesperproc=65536

    echo ""
    echo "Settings applied! These will reset on reboot."
    echo ""
    echo "To make permanent, add these lines to /etc/sysctl.conf:"
    echo "  net.inet.tcp.msl=1000"
    echo "  kern.maxfiles=65536"
    echo "  kern.maxfilesperproc=65536"

else
    echo "Changes not applied."
fi

echo ""
echo "You should also consider:"
echo "1. Restarting the trading bot daily via cron"
echo "2. Monitoring socket usage with: netstat -an | grep TIME_WAIT | wc -l"
echo "3. Using shorter sleep times in the strategy (already implemented)"
