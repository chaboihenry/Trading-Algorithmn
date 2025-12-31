#!/bin/bash
# Launch Parallel Parameter Sweep
# Runs 6 workers simultaneously to test all 1,152 configurations
# Expected time: 4-6 hours (vs 30+ hours sequential)

echo "================================"
echo "PARALLEL PARAMETER SWEEP LAUNCHER"
echo "================================"
echo ""
echo "Configuration:"
echo "- Total configs: 48"
echo "- Workers: 6"
echo "- Configs per worker: ~8"
echo "- Dataset: Last 2000 bars (FAST MODE)"
echo "- Expected time: 15-30 minutes"
echo ""
echo "Starting workers..."
echo ""

cd "/Users/henry/Desktop/Programming & Networking/Personal Projects/Integrated Trading Agent"

# Use the correct Python from conda environment
PYTHON="/Users/henry/miniconda3/envs/trading/bin/python"

# Launch 6 workers in background
$PYTHON scripts/parameter_sweep_parallel.py --worker-id 0 --total-workers 6 > results/worker_0.log 2>&1 &
$PYTHON scripts/parameter_sweep_parallel.py --worker-id 1 --total-workers 6 > results/worker_1.log 2>&1 &
$PYTHON scripts/parameter_sweep_parallel.py --worker-id 2 --total-workers 6 > results/worker_2.log 2>&1 &
$PYTHON scripts/parameter_sweep_parallel.py --worker-id 3 --total-workers 6 > results/worker_3.log 2>&1 &
$PYTHON scripts/parameter_sweep_parallel.py --worker-id 4 --total-workers 6 > results/worker_4.log 2>&1 &
$PYTHON scripts/parameter_sweep_parallel.py --worker-id 5 --total-workers 6 > results/worker_5.log 2>&1 &

echo "✓ All 6 workers launched!"
echo ""
echo "Monitor progress with:"
echo "  tail -f results/worker_0.log"
echo "  tail -f results/worker_1.log"
echo "  ... etc"
echo ""
echo "Or check all at once:"
echo "  watch 'grep Progress results/worker_*.log | tail -20'"
echo ""
echo "When complete, run:"
echo "  python scripts/combine_sweep_results.py"
