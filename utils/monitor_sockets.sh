#!/bin/bash

# Monitor socket usage for the trading bot
# Run this while your bot is running to track connection health

echo "Trading Bot Socket Monitor"
echo "=========================="
echo ""

while true; do
    clear
    echo "Trading Bot Socket Monitor - $(date)"
    echo "========================================"
    echo ""

    # Count Python processes (your bot)
    BOT_PIDS=$(pgrep -f "python.*live_trader.py")
    if [ -z "$BOT_PIDS" ]; then
        echo "⚠️  Trading bot is not running!"
    else
        echo "✅ Trading bot running (PID: $BOT_PIDS)"
    fi

    echo ""
    echo "Socket Statistics:"
    echo "------------------"

    # Count sockets in different states
    ESTABLISHED=$(netstat -an | grep ESTABLISHED | wc -l | tr -d ' ')
    TIME_WAIT=$(netstat -an | grep TIME_WAIT | wc -l | tr -d ' ')
    CLOSE_WAIT=$(netstat -an | grep CLOSE_WAIT | wc -l | tr -d ' ')
    TOTAL=$(netstat -an | grep -E 'ESTABLISHED|TIME_WAIT|CLOSE_WAIT' | wc -l | tr -d ' ')

    echo "  ESTABLISHED: $ESTABLISHED"
    echo "  TIME_WAIT:   $TIME_WAIT"
    echo "  CLOSE_WAIT:  $CLOSE_WAIT"
    echo "  Total:       $TOTAL"

    # Warning thresholds
    if [ "$TIME_WAIT" -gt 1000 ]; then
        echo ""
        echo "⚠️  WARNING: High number of TIME_WAIT connections!"
        echo "   Consider restarting the bot or applying system tuning."
    fi

    if [ "$CLOSE_WAIT" -gt 100 ]; then
        echo ""
        echo "⚠️  WARNING: High number of CLOSE_WAIT connections!"
        echo "   The application may not be closing connections properly."
    fi

    echo ""
    echo "Alpaca Connections:"
    echo "-------------------"

    # Count connections to Alpaca
    ALPACA_CONNS=$(netstat -an | grep -E 'api.alpaca.markets|data.alpaca.markets|stream.data.alpaca.markets' | wc -l | tr -d ' ')
    echo "  Active Alpaca connections: $ALPACA_CONNS"

    if [ "$ALPACA_CONNS" -gt 10 ]; then
        echo "  ⚠️  Multiple Alpaca connections detected (expected: 1-3)"
    fi

    echo ""
    echo "Recent Errors (from logs):"
    echo "--------------------------"
    LATEST_LOG=$(ls -t logs/live_trading_*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo "  Latest log: $LATEST_LOG"
        ERROR_COUNT=$(grep -c "Can't assign requested address" "$LATEST_LOG" 2>/dev/null || echo 0)
        if [ "$ERROR_COUNT" -gt 0 ]; then
            echo "  🔴 Socket errors found: $ERROR_COUNT"
            echo "  Last occurrence:"
            grep "Can't assign requested address" "$LATEST_LOG" | tail -1
        else
            echo "  ✅ No socket errors"
        fi
    else
        echo "  (No log files found in logs/ directory)"
    fi

    echo ""
    echo "Press Ctrl+C to exit. Refreshing in 10 seconds..."
    sleep 10
done
