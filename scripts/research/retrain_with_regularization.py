#!/usr/bin/env python3
"""
Retrain Models with Better Generalization

Tests multiple regularization configurations and selects the one with best
out-of-sample performance (not just training performance).

Usage:
    python scripts/research/retrain_with_regularization.py --symbol AAPL --test-all
    python scripts/research/retrain_with_regularization.py --tier tier_1
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Regularization configurations to test
REGULARIZATION_CONFIGS = {
    'baseline': {
        'name': 'Baseline (Current)',
        'primary_model': RandomForestClassifier(
            n_estimators=100,
            max_depth=None,  # No limit
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        ),
        'meta_model': RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42
        )
    },

    'light_regularization': {
        'name': 'Light Regularization',
        'primary_model': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,  # Limit depth
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'meta_model': LogisticRegression(
            C=1.0,  # Default regularization
            random_state=42,
            max_iter=1000
        )
    },

    'medium_regularization': {
        'name': 'Medium Regularization (RECOMMENDED)',
        'primary_model': RandomForestClassifier(
            n_estimators=50,  # Fewer trees
            max_depth=5,  # Shallower
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ),
        'meta_model': LogisticRegression(
            C=0.1,  # Stronger regularization
            random_state=42,
            max_iter=1000
        )
    },

    'heavy_regularization': {
        'name': 'Heavy Regularization',
        'primary_model': LogisticRegression(
            C=0.01,  # Very strong regularization
            random_state=42,
            max_iter=1000
        ),
        'meta_model': LogisticRegression(
            C=0.01,
            random_state=42,
            max_iter=1000
        )
    },

    'simple_tree': {
        'name': 'Simple Decision Tree',
        'primary_model': DecisionTreeClassifier(
            max_depth=3,  # Very simple
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        ),
        'meta_model': LogisticRegression(
            C=0.1,
            random_state=42,
            max_iter=1000
        )
    }
}


class GeneralizationValidator:
    """Train and validate models with focus on out-of-sample performance."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.results = {}

    def load_data(self) -> bool:
        """Load and prepare data for training."""
        # Use RiskLabAIStrategy to load and process data
        self.strategy = RiskLabAIStrategy()

        # Load tick data and generate bars (same as original training)
        try:
            from config.tick_config import TICK_DB_PATH
            import sqlite3

            conn = sqlite3.connect(str(TICK_DB_PATH))
            query = f"""
                SELECT timestamp, price, volume
                FROM ticks
                WHERE symbol = '{self.symbol}'
                ORDER BY timestamp
            """
            ticks_df = pd.read_sql_query(query, conn)
            conn.close()

            if len(ticks_df) == 0:
                logger.error(f"{self.symbol}: No tick data")
                return False

            # Generate bars
            ticks_df['timestamp'] = pd.to_datetime(ticks_df['timestamp'])
            bars_df = self.strategy.generate_imbalance_bars(ticks_df)

            # Split into train/validation/test (60/20/20)
            n = len(bars_df)
            train_end = int(n * 0.6)
            val_end = int(n * 0.8)

            self.train_bars = bars_df.iloc[:train_end].copy()
            self.val_bars = bars_df.iloc[train_end:val_end].copy()
            self.test_bars = bars_df.iloc[val_end:].copy()

            logger.info(f"{self.symbol}: Loaded {len(self.train_bars)} train, "
                       f"{len(self.val_bars)} validation, {len(self.test_bars)} test bars")

            return True

        except Exception as e:
            logger.error(f"{self.symbol}: Error loading data - {e}")
            return False

    def train_and_evaluate(self, config_name: str, config: Dict) -> Dict:
        """Train with given configuration and evaluate on all sets."""

        logger.info(f"\n{self.symbol}: Testing {config['name']}...")

        # Create fresh strategy instance
        strategy = RiskLabAIStrategy()
        strategy.primary_model = config['primary_model']
        strategy.meta_model = config['meta_model']

        # Prepare features and labels for all sets
        train_features = strategy.generate_features(self.train_bars)
        val_features = strategy.generate_features(self.val_bars)
        test_features = strategy.generate_features(self.test_bars)

        train_labels_primary = strategy.triple_barrier_label(self.train_bars)
        val_labels_primary = strategy.triple_barrier_label(self.val_bars)
        test_labels_primary = strategy.triple_barrier_label(self.test_bars)

        # Train primary model
        strategy.primary_model.fit(train_features, train_labels_primary)

        # Evaluate primary model
        train_pred_primary = strategy.primary_model.predict(train_features)
        val_pred_primary = strategy.primary_model.predict(val_features)
        test_pred_primary = strategy.primary_model.predict(test_features)

        train_acc_primary = (train_pred_primary == train_labels_primary).mean()
        val_acc_primary = (val_pred_primary == val_labels_primary).mean()
        test_acc_primary = (test_pred_primary == test_labels_primary).mean()

        # Train meta model (on bars where primary predicted trade)
        train_trade_mask = train_pred_primary != 0
        if train_trade_mask.sum() > 10:  # Need minimum samples
            train_labels_meta = (train_labels_primary[train_trade_mask] == train_pred_primary[train_trade_mask]).astype(int)
            strategy.meta_model.fit(train_features[train_trade_mask], train_labels_meta)

            # Evaluate meta model
            val_trade_mask = val_pred_primary != 0
            test_trade_mask = test_pred_primary != 0

            val_acc_meta = 0.0
            test_acc_meta = 0.0

            if val_trade_mask.sum() > 0:
                val_labels_meta = (val_labels_primary[val_trade_mask] == val_pred_primary[val_trade_mask]).astype(int)
                val_pred_meta = strategy.meta_model.predict(val_features[val_trade_mask])
                val_acc_meta = (val_pred_meta == val_labels_meta).mean()

            if test_trade_mask.sum() > 0:
                test_labels_meta = (test_labels_primary[test_trade_mask] == test_pred_primary[test_trade_mask]).astype(int)
                test_pred_meta = strategy.meta_model.predict(test_features[test_trade_mask])
                test_acc_meta = (test_pred_meta == test_labels_meta).mean()
        else:
            val_acc_meta = 0.0
            test_acc_meta = 0.0

        results = {
            'config_name': config_name,
            'train_acc_primary': train_acc_primary,
            'val_acc_primary': val_acc_primary,
            'test_acc_primary': test_acc_primary,
            'train_acc_meta': train_acc_meta if 'train_acc_meta' in locals() else 0.0,
            'val_acc_meta': val_acc_meta,
            'test_acc_meta': test_acc_meta,
            'val_gap_primary': train_acc_primary - val_acc_primary,
            'test_gap_primary': train_acc_primary - test_acc_primary,
            'strategy': strategy  # Save for later use
        }

        # Print results
        logger.info(f"  Primary Model: Train={train_acc_primary:.1%}, Val={val_acc_primary:.1%}, "
                   f"Test={test_acc_primary:.1%}, Val Gap={results['val_gap_primary']:.1%}")
        logger.info(f"  Meta Model:    Train={results['train_acc_meta']:.1%}, Val={val_acc_meta:.1%}, "
                   f"Test={test_acc_meta:.1%}")

        return results

    def select_best_configuration(self, results: List[Dict]) -> Dict:
        """Select configuration with best validation performance (not training!)."""

        # Sort by validation accuracy (not training accuracy)
        # This ensures we pick the model that generalizes best
        sorted_results = sorted(results, key=lambda x: x['val_acc_primary'], reverse=True)

        best = sorted_results[0]

        logger.info(f"\n{'='*80}")
        logger.info(f"BEST CONFIGURATION: {best['config_name']}")
        logger.info(f"{'='*80}")
        logger.info(f"Validation Accuracy: {best['val_acc_primary']:.1%}")
        logger.info(f"Test Accuracy:       {best['test_acc_primary']:.1%}")
        logger.info(f"Validation Gap:      {best['val_gap_primary']:.1%}")
        logger.info(f"Test Gap:            {best['test_gap_primary']:.1%}")

        return best

    def save_best_model(self, best_result: Dict):
        """Save the best performing model."""

        models_dir = project_root / "models"
        models_dir.mkdir(exist_ok=True)

        model_path = models_dir / f"risklabai_{self.symbol}_models.pkl"

        # Save strategy with best models
        with open(model_path, 'wb') as f:
            pickle.dump({
                'primary_model': best_result['strategy'].primary_model,
                'meta_model': best_result['strategy'].meta_model,
                'config_name': best_result['config_name'],
                'val_acc_primary': best_result['val_acc_primary'],
                'test_acc_primary': best_result['test_acc_primary']
            }, f)

        logger.info(f"✓ Saved best model to {model_path}")
        logger.info(f"  Configuration: {best_result['config_name']}")
        logger.info(f"  Test Accuracy: {best_result['test_acc_primary']:.1%}")


def main():
    parser = argparse.ArgumentParser(description='Retrain models with better generalization')
    parser.add_argument('--symbol', type=str, help='Symbol to retrain')
    parser.add_argument('--tier', type=str, help='Tier to retrain (tier_1, tier_2, etc.)')
    parser.add_argument('--test-all', action='store_true', help='Test all regularization configs')
    parser.add_argument('--config', type=str, default='medium_regularization',
                       choices=list(REGULARIZATION_CONFIGS.keys()),
                       help='Which regularization config to use (if not testing all)')

    args = parser.parse_args()

    # Get symbols
    symbols = []
    if args.symbol:
        symbols = [args.symbol]
    elif args.tier:
        from config.all_symbols import get_symbols_by_tier
        symbols = get_symbols_by_tier(args.tier)
    else:
        logger.error("Must specify --symbol or --tier")
        return 1

    logger.info(f"Retraining {len(symbols)} symbols...")
    logger.info(f"Testing all configs: {args.test_all}")

    # Process each symbol
    for symbol in symbols:
        validator = GeneralizationValidator(symbol)

        if not validator.load_data():
            continue

        if args.test_all:
            # Test all configurations
            results = []
            for config_name, config in REGULARIZATION_CONFIGS.items():
                result = validator.train_and_evaluate(config_name, config)
                results.append(result)

            # Select and save best
            best = validator.select_best_configuration(results)
            validator.save_best_model(best)

            # Print comparison table
            print(f"\n{symbol}: CONFIGURATION COMPARISON")
            print("="*100)
            print(f"{'Configuration':<30} {'Train':<10} {'Val':<10} {'Test':<10} {'Val Gap':<10} {'Test Gap':<10}")
            print("-"*100)
            for r in sorted(results, key=lambda x: x['val_acc_primary'], reverse=True):
                print(f"{r['config_name']:<30} "
                      f"{r['train_acc_primary']:>8.1%}  "
                      f"{r['val_acc_primary']:>8.1%}  "
                      f"{r['test_acc_primary']:>8.1%}  "
                      f"{r['val_gap_primary']:>8.1%}  "
                      f"{r['test_gap_primary']:>8.1%}")
            print("="*100)

        else:
            # Use specified configuration
            config = REGULARIZATION_CONFIGS[args.config]
            result = validator.train_and_evaluate(args.config, config)
            validator.save_best_model(result)

    logger.info("\n✓ Retraining complete!")
    logger.info("\nNext steps:")
    logger.info("1. Run realistic backtest to see if performance improved:")
    logger.info("   python test_suite/realistic_backtest.py --tier tier_1 --days 252")
    logger.info("\n2. Compare to baseline with diagnostic tool:")
    logger.info("   python scripts/research/diagnose_overfitting.py --tier tier_1")


if __name__ == "__main__":
    main()
