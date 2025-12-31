#!/usr/bin/env python3
"""
Aggressive Model Retraining Script

This script addresses the conservative model bias by:
1. Using TIGHT barriers (0.5% instead of 4%/2%) for tick bars
2. Forcing directional labels - only label as neutral if truly flat
3. Using class_weight='balanced' to handle any remaining imbalance
4. Training on tick imbalance bars for better signal quality

The goal: Reduce neutral predictions from 97% to ~33% (balanced classes)
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy
from config.tick_config import (
    TICK_DB_PATH,
    INITIAL_IMBALANCE_THRESHOLD,
    OPTIMAL_PROFIT_TARGET,
    OPTIMAL_STOP_LOSS,
    OPTIMAL_MAX_HOLDING_BARS,
    OPTIMAL_FRACTIONAL_D
)
from data.tick_storage import TickStorage
from data.tick_to_bars import generate_bars_from_ticks
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Train aggressive RiskLabAI model with tight barriers and forced directional labels."""

    logger.info("=" * 80)
    logger.info("AGGRESSIVE MODEL RETRAINING")
    logger.info("=" * 80)
    logger.info("Goal: Fix 97% neutral predictions by using tight barriers + force_directional")
    logger.info("")

    # Configuration
    SYMBOL = "SPY"
    TICK_THRESHOLD = INITIAL_IMBALANCE_THRESHOLD
    MODEL_PATH = "models/risklabai_tick_models_aggressive.pkl"

    # PARAMETERS - Use your ACTUAL optimized values from tick_config.py
    # The KEY change is force_directional=True, not changing your sweep results
    PROFIT_TAKING = OPTIMAL_PROFIT_TARGET  # 0.04 (4%) - from sweep
    STOP_LOSS = OPTIMAL_STOP_LOSS          # 0.02 (2%) - from sweep
    MAX_HOLDING = OPTIMAL_MAX_HOLDING_BARS # 20 bars - from sweep
    D_PARAM = OPTIMAL_FRACTIONAL_D         # 0.30 - preserves 70% memory
    FORCE_DIRECTIONAL = True               # KEY: Force directional labels to fix 97% neutral
    NEUTRAL_THRESHOLD = 0.00001            # Only label as neutral if |return| < 0.001%

    logger.info("Training Parameters:")
    logger.info(f"  Symbol: {SYMBOL}")
    logger.info(f"  Profit Taking: {PROFIT_TAKING:.2%} ({PROFIT_TAKING}) - from sweep")
    logger.info(f"  Stop Loss: {STOP_LOSS:.2%} ({STOP_LOSS}) - from sweep")
    logger.info(f"  Max Holding: {MAX_HOLDING} bars - from sweep")
    logger.info(f"  D (frac diff): {D_PARAM} (preserves {(1-D_PARAM)*100:.0f}% memory)")
    logger.info(f"  Force Directional: {FORCE_DIRECTIONAL} ← KEY FIX")
    logger.info(f"  Neutral Threshold: {NEUTRAL_THRESHOLD:.5f} (0.001%)")
    logger.info("")

    # Step 1: Load tick data
    logger.info("=" * 60)
    logger.info("STEP 1: Loading tick data from database")
    logger.info("=" * 60)

    storage = TickStorage(TICK_DB_PATH)

    # Check available data
    date_range = storage.get_date_range(SYMBOL)
    if not date_range:
        logger.error(f"No tick data found for {SYMBOL}")
        logger.error("Run scripts/backfill_ticks.py first to collect tick data")
        return 1

    earliest, latest = date_range
    logger.info(f"Available data: {earliest} to {latest}")

    # Load all ticks
    ticks = storage.load_ticks(SYMBOL)
    storage.close()

    if not ticks:
        logger.error(f"Failed to load ticks for {SYMBOL}")
        return 1

    logger.info(f"✓ Loaded {len(ticks):,} ticks")
    logger.info("")

    # Step 2: Generate tick imbalance bars
    logger.info("=" * 60)
    logger.info("STEP 2: Generating tick imbalance bars")
    logger.info("=" * 60)

    bars_list = generate_bars_from_ticks(ticks, threshold=TICK_THRESHOLD)

    if not bars_list:
        logger.error(f"Failed to generate bars from {len(ticks)} ticks")
        return 1

    logger.info(f"✓ Generated {len(bars_list)} bars")
    logger.info(f"  Compression ratio: {len(ticks)/len(bars_list):.1f} ticks per bar")
    logger.info("")

    # Step 3: Convert to DataFrame
    logger.info("=" * 60)
    logger.info("STEP 3: Converting to DataFrame")
    logger.info("=" * 60)

    bars_df = pd.DataFrame(bars_list)

    # Set datetime index (remove timezone for compatibility)
    bars_df['bar_end'] = pd.to_datetime(bars_df['bar_end'])
    if bars_df['bar_end'].dt.tz is not None:
        bars_df['bar_end'] = bars_df['bar_end'].dt.tz_localize(None)
    bars_df.set_index('bar_end', inplace=True)

    logger.info(f"✓ DataFrame shape: {bars_df.shape}")
    logger.info(f"  Columns: {bars_df.columns.tolist()}")
    logger.info(f"  Date range: {bars_df.index[0]} to {bars_df.index[-1]}")
    logger.info("")

    # Step 4: Initialize aggressive strategy
    logger.info("=" * 60)
    logger.info("STEP 4: Initializing RiskLabAI with aggressive settings")
    logger.info("=" * 60)

    strategy = RiskLabAIStrategy(
        profit_taking=PROFIT_TAKING,
        stop_loss=STOP_LOSS,
        max_holding=MAX_HOLDING,
        d=D_PARAM,
        n_cv_splits=5,
        force_directional=FORCE_DIRECTIONAL,
        neutral_threshold=NEUTRAL_THRESHOLD
    )

    logger.info("✓ Strategy initialized with aggressive labeling")
    logger.info("")

    # Step 5: Train models
    logger.info("=" * 60)
    logger.info("STEP 5: Training models")
    logger.info("=" * 60)
    logger.info("Expected: Balanced label distribution (not 97% neutral)")
    logger.info("")

    results = strategy.train(bars_df, min_samples=100)

    if not results['success']:
        logger.error(f"Training failed: {results.get('reason', 'Unknown error')}")
        return 1

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 6: Training Results")
    logger.info("=" * 60)
    logger.info(f"✓ Training successful!")
    logger.info(f"  Samples: {results['n_samples']}")
    logger.info(f"  Primary model accuracy: {results['primary_accuracy']:.3f}")
    logger.info(f"  Meta model accuracy: {results['meta_accuracy']:.3f}")
    logger.info(f"  Top features: {results['top_features'][:5]}")
    logger.info("")

    # Step 7: Test predictions on recent data
    logger.info("=" * 60)
    logger.info("STEP 7: Testing predictions on recent data")
    logger.info("=" * 60)

    # Test on last 100 bars
    test_bars = bars_df.tail(200)  # Need extra for feature calculation

    predictions = []
    bet_sizes = []

    for i in range(100, len(test_bars)):
        # Get bars up to this point
        current_bars = test_bars.iloc[:i]

        # Get prediction
        signal, bet_size = strategy.predict(
            current_bars,
            prob_threshold=0.015,  # Your optimized threshold
            meta_threshold=0.001   # Your optimized threshold
        )

        predictions.append(signal)
        bet_sizes.append(bet_size)

    # Count predictions
    import numpy as np
    pred_counts = np.bincount(np.array(predictions) + 1, minlength=3)
    total_preds = len(predictions)

    short_count = pred_counts[0]  # -1 + 1 = 0
    neutral_count = pred_counts[1]  # 0 + 1 = 1
    long_count = pred_counts[2]  # 1 + 1 = 2

    logger.info(f"Prediction distribution on {total_preds} test samples:")
    logger.info(f"  Short (-1):   {short_count:4d} ({short_count/total_preds*100:5.2f}%)")
    logger.info(f"  Neutral (0):  {neutral_count:4d} ({neutral_count/total_preds*100:5.2f}%)")
    logger.info(f"  Long (1):     {long_count:4d} ({long_count/total_preds*100:5.2f}%)")
    logger.info("")

    if neutral_count / total_preds > 0.80:
        logger.warning("⚠️  Still predicting >80% neutral!")
        logger.warning("   Consider even tighter barriers or lower neutral_threshold")
    elif neutral_count / total_preds > 0.50:
        logger.warning("⚠️  Still predicting >50% neutral")
        logger.warning("   Better than 97%, but could be more aggressive")
    else:
        logger.info("✓ Good balance! Neutral predictions under 50%")

    logger.info("")

    # Step 8: Save models
    logger.info("=" * 60)
    logger.info("STEP 8: Saving models")
    logger.info("=" * 60)

    # Create models directory if it doesn't exist
    models_dir = Path(MODEL_PATH).parent
    models_dir.mkdir(exist_ok=True)

    strategy.save_models(MODEL_PATH)
    logger.info(f"✓ Models saved to {MODEL_PATH}")
    logger.info("")

    # Final summary
    logger.info("=" * 80)
    logger.info("AGGRESSIVE RETRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Training samples: {results['n_samples']}")
    logger.info(f"Primary accuracy: {results['primary_accuracy']:.3f}")
    logger.info(f"Meta accuracy: {results['meta_accuracy']:.3f}")
    logger.info(f"Prediction balance: {short_count/total_preds*100:.1f}% / {neutral_count/total_preds*100:.1f}% / {long_count/total_preds*100:.1f}%")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Update run_live_trading.py to use the new model path:")
    logger.info(f"   model_path='{MODEL_PATH}'")
    logger.info("2. Run the bot and verify it generates more trading signals")
    logger.info("3. Monitor terminal output to see balanced probability distributions")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
