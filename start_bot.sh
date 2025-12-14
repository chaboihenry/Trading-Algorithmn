#!/bin/bash
# Start Trading Bot - Run this from anywhere
# This script ensures the bot starts from the correct directory

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to project root
cd "$SCRIPT_DIR"

# Activate conda environment and run bot
echo "Starting trading bot..."
echo "Project root: $SCRIPT_DIR"
echo "Strategy: ${1:-combined}"
echo ""

# Run with trading environment
/Users/henry/miniconda3/envs/trading/bin/python core/live_trader.py --strategy "${1:-combined}"
