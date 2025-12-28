#!/usr/bin/env python3
"""
Debug Model Predictions

Check what signals the model is generating on historical data.
"""

import sys
import logging
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy
from config.tick_config import TICK_DB_PATH, INITIAL_IMBALANCE_THRESHOLD
from data.tick_storage import TickStorage
from data.tick_to_bars import generate_bars_from_ticks

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Debug predictions."""
    # Load model
    logger.info("Loading model...")
    strategy = RiskLabAIStrategy()
    strategy.load_models('models/risklabai_tick_models_365days.pkl')

    # Load tick data
    logger.info("Loading tick data...")
    storage = TickStorage(TICK_DB_PATH)
    ticks = storage.load_ticks('SPY')
    storage.close()

    logger.info(f"Loaded {len(ticks):,} ticks")

    # Generate bars
    logger.info("Generating bars...")
    bars_list = generate_bars_from_ticks(ticks, threshold=INITIAL_IMBALANCE_THRESHOLD)
    logger.info(f"Generated {len(bars_list)} bars")

    # Convert to DataFrame
    bars_df = pd.DataFrame(bars_list)
    bars_df['bar_end'] = pd.to_datetime(bars_df['bar_end'])
    if bars_df['bar_end'].dt.tz is not None:
        bars_df['bar_end'] = bars_df['bar_end'].dt.tz_localize(None)
    bars_df.set_index('bar_end', inplace=True)

    logger.info(f"DataFrame shape: {bars_df.shape}")
    logger.info(f"Columns: {bars_df.columns.tolist()}")

    # Test predictions on recent data
    logger.info("\nTesting predictions on recent 10 bars:")
    logger.info("=" * 80)

    signals_count = {-1: 0, 0: 0, 1: 0}
    meta_probs = []

    for i in range(max(100, len(bars_df)-100), len(bars_df)):
        historical_bars = bars_df.iloc[:i].copy()

        try:
            signal, bet_size = strategy.predict(historical_bars)

            # Get meta probability
            features = strategy.prepare_features(historical_bars)
            if len(features) > 0 and strategy.meta_model is not None:
                X = features.iloc[[-1]]
                meta_prob = strategy.meta_model.predict_proba(X)[0, 1]
                meta_probs.append(meta_prob)
            else:
                meta_prob = 0.5

            signals_count[signal] += 1

            if i >= len(bars_df) - 10:
                logger.info(f"Bar {i}: signal={signal}, bet_size={bet_size:.3f}, meta_prob={meta_prob:.3f}")

        except Exception as e:
            logger.error(f"Prediction failed at bar {i}: {e}")
            import traceback
            traceback.print_exc()

    logger.info("\n" + "=" * 80)
    logger.info("SIGNAL DISTRIBUTION:")
    logger.info(f"  Short (-1): {signals_count[-1]}")
    logger.info(f"  No Trade (0): {signals_count[0]}")
    logger.info(f"  Long (1): {signals_count[1]}")

    if meta_probs:
        logger.info(f"\nMETA PROBABILITIES:")
        logger.info(f"  Min: {min(meta_probs):.3f}")
        logger.info(f"  Max: {max(meta_probs):.3f}")
        logger.info(f"  Mean: {sum(meta_probs)/len(meta_probs):.3f}")
        logger.info(f"  Above 0.5: {sum(1 for p in meta_probs if p > 0.5)}/{len(meta_probs)}")

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
