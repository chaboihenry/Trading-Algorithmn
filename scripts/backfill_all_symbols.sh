#!/bin/bash
# Backfill All 102 Symbols (99 stocks + 3 ETFs)
# This will take 4-8 hours to complete
# Recommendation: Run overnight or in a tmux/screen session

echo "================================================================================"
echo "BACKFILLING ALL 102 SYMBOLS (365 DAYS)"
echo "================================================================================"
echo "Tier 1 Fixes Applied:"
echo "  ✓ C1: Extended to 365 days (need 2000+ samples)"
echo "  ✓ C2: CUSUM filter threshold = 2.5x volatility (~35% filter rate)"
echo "  ✓ C3: Robust fractional differentiation with ADF tests"
echo ""
echo "Estimated time: 4-8 hours (overnight recommended)"
echo "================================================================================"

# Navigate to project root
cd "$(dirname "$0")/.."

# Run backfill for all symbols
# Using --days 365 from config/tick_config.py
/Users/henry/miniconda3/envs/trading/bin/python scripts/setup/backfill_ticks.py --days 365

echo ""
echo "================================================================================"
echo "✓ BACKFILL COMPLETE"
echo "================================================================================"
echo "Next step: Run scripts/train_all_symbols.py to train models with tier 1 fixes"
echo "================================================================================"
