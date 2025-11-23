#!/Users/henry/miniconda3/envs/trading/bin/python
"""
Automated Strategy Retraining Script
====================================
Runs daily to incrementally retrain all strategies with new data

Usage:
    python retrain_all_strategies.py [--force-full-retrain]

Options:
    --force-full-retrain    Force complete retrain instead of incremental update
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add strategies to path
sys.path.insert(0, str(Path(__file__).parent))

from sentiment_trading import SentimentTradingStrategy
from volatility_trading import VolatilityTradingStrategy
from incremental_trainer import IncrementalTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('strategies/logs/retraining.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Retrain all strategies"""
    parser = argparse.ArgumentParser(description='Retrain trading strategies')
    parser.add_argument('--force-full-retrain', action='store_true',
                       help='Force complete retrain instead of incremental')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info(f"STRATEGY RETRAINING STARTED: {datetime.now()}")
    logger.info("=" * 80)

    if args.force_full_retrain:
        logger.info("FORCING FULL RETRAIN (all historical data)")
    else:
        logger.info("INCREMENTAL UPDATE MODE (new data only)")

    results = {}

    # 1. Sentiment Strategy
    logger.info("\n" + "=" * 80)
    logger.info("RETRAINING: Sentiment Trading Strategy")
    logger.info("=" * 80)

    try:
        sentiment_strategy = SentimentTradingStrategy()
        success = sentiment_strategy.train_model(force_full_retrain=args.force_full_retrain)

        if success:
            logger.info("✅ Sentiment Strategy: Training successful")
            results['SentimentTradingStrategy'] = 'SUCCESS'

            # Generate signals to verify
            signals = sentiment_strategy.generate_signals()
            logger.info(f"   Generated {len(signals)} signals")

            if len(signals) > 0:
                logger.info(f"   Signal breakdown:")
                for signal_type in signals['signal_type'].unique():
                    count = len(signals[signals['signal_type'] == signal_type])
                    logger.info(f"     {signal_type}: {count}")
        else:
            logger.error("❌ Sentiment Strategy: Training failed")
            results['SentimentTradingStrategy'] = 'FAILED'

    except Exception as e:
        logger.error(f"❌ Sentiment Strategy: Exception - {e}", exc_info=True)
        results['SentimentTradingStrategy'] = f'ERROR: {e}'

    # 2. TODO: Pairs Strategy (when converted to incremental)
    logger.info("\n" + "=" * 80)
    logger.info("RETRAINING: Pairs Trading Strategy")
    logger.info("=" * 80)
    logger.info("⏭️  Pairs Strategy: Not yet converted to incremental learning")
    logger.info("   Using existing implementation (no ML model)")
    results['PairsTradingStrategy'] = 'SKIPPED (no ML model)'

    # 3. Volatility Strategy (incremental learning enabled)
    logger.info("\n" + "=" * 80)
    logger.info("RETRAINING: Volatility Trading Strategy")
    logger.info("=" * 80)

    try:
        volatility_strategy = VolatilityTradingStrategy()
        result = volatility_strategy.incremental_train(force_full=args.force_full_retrain)

        if result.get('success'):
            logger.info("✅ Volatility Strategy: Training successful")
            results['VolatilityTradingStrategy'] = 'SUCCESS'

            # Log training details
            logger.info(f"   Training type: {'FULL' if result.get('full_retrain') else 'INCREMENTAL'}")
            logger.info(f"   Train accuracy: {result.get('train_accuracy', 0):.2%}")
            logger.info(f"   Test accuracy: {result.get('test_accuracy', 0):.2%}")
            logger.info(f"   Samples trained: {result.get('samples_trained', 0)}")

            # Generate signals to verify
            signals = volatility_strategy.generate_signals()
            logger.info(f"   Generated {len(signals)} signals")

            if len(signals) > 0:
                logger.info(f"   Signal breakdown:")
                for signal_type in signals['signal_type'].unique():
                    count = len(signals[signals['signal_type'] == signal_type])
                    logger.info(f"     {signal_type}: {count}")
        else:
            logger.error(f"❌ Volatility Strategy: Training failed - {result.get('error', 'Unknown')}")
            results['VolatilityTradingStrategy'] = 'FAILED'

    except Exception as e:
        logger.error(f"❌ Volatility Strategy: Exception - {e}", exc_info=True)
        results['VolatilityTradingStrategy'] = f'ERROR: {e}'

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("RETRAINING SUMMARY")
    logger.info("=" * 80)

    for strategy, status in results.items():
        icon = "✅" if status == "SUCCESS" else "⏭️" if "SKIPPED" in status else "❌"
        logger.info(f"{icon} {strategy}: {status}")

    logger.info("\n" + "=" * 80)
    logger.info(f"RETRAINING COMPLETED: {datetime.now()}")
    logger.info("=" * 80)

    # Check training history
    logger.info("\n" + "=" * 80)
    logger.info("MODEL VERSION HISTORY")
    logger.info("=" * 80)

    trainer = IncrementalTrainer()

    for strategy_name in ['SentimentTradingStrategy', 'VolatilityTradingStrategy']:
        logger.info(f"\n{strategy_name}:")
        logger.info("-" * 80)

        history = trainer.get_training_summary(strategy_name, limit=5)

        if not history.empty:
            for _, row in history.iterrows():
                retrain_type = "FULL" if row['is_full_retrain'] else "INCREMENTAL"
                logger.info(f"  v{row['model_version']} ({retrain_type})")
                logger.info(f"    Date: {row['trained_date']}")
                logger.info(f"    Data until: {row['training_end_date']}")
                logger.info(f"    Samples: {row['num_training_samples']} total, {row['num_new_samples']} new")
                logger.info(f"    Accuracy: Train={row['train_accuracy']:.2%}, Test={row['test_accuracy']:.2%}")
                logger.info("")
        else:
            logger.info("  No training history found")

    return 0 if all(v in ['SUCCESS', 'SKIPPED (no ML model)']
                    for v in results.values()) else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
