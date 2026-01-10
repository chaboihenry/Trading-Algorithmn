#!/usr/bin/env python3
"""
Overfitting Diagnostic Tool

Analyzes trained models to measure overfitting severity and identify causes.
Provides actionable recommendations for improving generalization.

Usage:
    python scripts/research/diagnose_overfitting.py --symbols AAPL MSFT GOOGL
    python scripts/research/diagnose_overfitting.py --tier tier_1
"""

import sys
import sqlite3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.tick_config import TICK_DB_PATH
from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OverfittingDiagnostic:
    """Diagnose overfitting in trained RiskLabAI models."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.strategy = None
        self.train_bars = None
        self.test_bars = None

    def load_model_and_data(self) -> bool:
        """Load trained model and historical data."""
        model_path = project_root / f"models/risklabai_{self.symbol}_models.pkl"

        if not model_path.exists():
            logger.error(f"{self.symbol}: No trained model found at {model_path}")
            return False

        # Load model
        self.strategy = RiskLabAIStrategy()
        self.strategy.load_models(str(model_path))

        # Load tick data
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
            logger.error(f"{self.symbol}: No tick data found")
            return False

        # Generate bars
        ticks_df['timestamp'] = pd.to_datetime(ticks_df['timestamp'])
        bars_df = self.strategy.generate_imbalance_bars(ticks_df)

        # Split train/test (70/30 like training)
        split_idx = int(len(bars_df) * 0.7)
        self.train_bars = bars_df.iloc[:split_idx].copy()
        self.test_bars = bars_df.iloc[split_idx:].copy()

        logger.info(f"{self.symbol}: Loaded {len(self.train_bars)} train bars, {len(self.test_bars)} test bars")
        return True

    def measure_performance_gap(self) -> Dict[str, float]:
        """Measure train vs test performance gap (key overfitting indicator)."""

        # Prepare features for both sets
        train_features = self.strategy.generate_features(self.train_bars)
        test_features = self.strategy.generate_features(self.test_bars)

        # Get labels
        train_labels_primary = self.strategy.triple_barrier_label(self.train_bars)
        test_labels_primary = self.strategy.triple_barrier_label(self.test_bars)

        # Primary model predictions
        train_pred_primary = self.strategy.primary_model.predict(train_features)
        test_pred_primary = self.strategy.primary_model.predict(test_features)

        # Calculate accuracies
        train_acc_primary = (train_pred_primary == train_labels_primary).mean()
        test_acc_primary = (test_pred_primary == test_labels_primary).mean()

        # Meta model (only on bars where primary predicted trade)
        train_trade_mask = train_pred_primary != 0
        test_trade_mask = test_pred_primary != 0

        train_acc_meta = 0.0
        test_acc_meta = 0.0

        if train_trade_mask.sum() > 0:
            train_labels_meta = (train_labels_primary[train_trade_mask] == train_pred_primary[train_trade_mask]).astype(int)
            train_pred_meta = self.strategy.meta_model.predict(train_features[train_trade_mask])
            train_acc_meta = (train_pred_meta == train_labels_meta).mean()

        if test_trade_mask.sum() > 0:
            test_labels_meta = (test_labels_primary[test_trade_mask] == test_pred_primary[test_trade_mask]).astype(int)
            test_pred_meta = self.strategy.meta_model.predict(test_features[test_trade_mask])
            test_acc_meta = (test_pred_meta == test_labels_meta).mean()

        return {
            'train_acc_primary': train_acc_primary,
            'test_acc_primary': test_acc_primary,
            'train_acc_meta': train_acc_meta,
            'test_acc_meta': test_acc_meta,
            'gap_primary': train_acc_primary - test_acc_primary,
            'gap_meta': train_acc_meta - test_acc_meta
        }

    def analyze_feature_importance_stability(self) -> Dict[str, any]:
        """Check if feature importance is consistent across CV folds."""

        # Get feature importances from CV
        if not hasattr(self.strategy, 'cv_feature_importances'):
            return {'stable': None, 'message': 'No CV feature importance data available'}

        importances = np.array(self.strategy.cv_feature_importances)

        # Calculate coefficient of variation for each feature
        mean_importance = importances.mean(axis=0)
        std_importance = importances.std(axis=0)
        cv = std_importance / (mean_importance + 1e-10)

        # High CV = unstable feature importance = overfitting
        avg_cv = cv.mean()
        max_cv = cv.max()

        return {
            'avg_cv': avg_cv,
            'max_cv': max_cv,
            'stable': avg_cv < 0.5,  # Threshold for stability
            'message': 'Stable' if avg_cv < 0.5 else 'Unstable - features changing importance across folds'
        }

    def check_label_distribution(self) -> Dict[str, float]:
        """Verify labels aren't too imbalanced (can cause overfitting to majority class)."""

        train_labels = self.strategy.triple_barrier_label(self.train_bars)
        test_labels = self.strategy.triple_barrier_label(self.test_bars)

        # Count each class (-1, 0, 1)
        train_dist = np.bincount(train_labels + 1, minlength=3) / len(train_labels)
        test_dist = np.bincount(test_labels + 1, minlength=3) / len(test_labels)

        # Check if distributions match (KL divergence would be better but simpler check here)
        dist_shift = np.abs(train_dist - test_dist).max()

        return {
            'train_short_pct': train_dist[0],
            'train_neutral_pct': train_dist[1],
            'train_long_pct': train_dist[2],
            'test_short_pct': test_dist[0],
            'test_neutral_pct': test_dist[1],
            'test_long_pct': test_dist[2],
            'max_shift': dist_shift,
            'distributions_match': dist_shift < 0.1
        }

    def estimate_effective_samples(self) -> int:
        """Estimate effective number of independent samples (accounting for autocorrelation)."""

        # Use close prices as proxy for returns autocorrelation
        returns = self.train_bars['close'].pct_change().dropna()

        # Calculate autocorrelation at lag 1
        autocorr = returns.autocorr(lag=1)

        # Effective N = N * (1 - autocorr) / (1 + autocorr)
        # Higher autocorr = fewer independent samples = easier to overfit
        n_bars = len(self.train_bars)
        effective_n = n_bars * (1 - autocorr) / (1 + autocorr)

        return {
            'n_bars': n_bars,
            'autocorr_lag1': autocorr,
            'effective_n': int(effective_n),
            'sufficient_data': effective_n > 500  # Rule of thumb
        }

    def run_full_diagnostic(self) -> Dict[str, any]:
        """Run all diagnostic checks."""

        logger.info(f"\n{'='*80}")
        logger.info(f"OVERFITTING DIAGNOSTIC: {self.symbol}")
        logger.info(f"{'='*80}\n")

        if not self.load_model_and_data():
            return None

        results = {
            'symbol': self.symbol,
            'performance_gap': self.measure_performance_gap(),
            'feature_stability': self.analyze_feature_importance_stability(),
            'label_distribution': self.check_label_distribution(),
            'sample_independence': self.estimate_effective_samples()
        }

        # Print results
        self._print_diagnostic_results(results)

        # Generate recommendations
        recommendations = self._generate_recommendations(results)
        results['recommendations'] = recommendations

        return results

    def _print_diagnostic_results(self, results: Dict):
        """Print formatted diagnostic results."""

        perf = results['performance_gap']
        feat = results['feature_stability']
        label = results['label_distribution']
        sample = results['sample_independence']

        print("\n1. TRAIN/TEST PERFORMANCE GAP (Key Overfitting Indicator)")
        print("   " + "-" * 60)
        print(f"   Primary Model:")
        print(f"      Train Accuracy: {perf['train_acc_primary']:.1%}")
        print(f"      Test Accuracy:  {perf['test_acc_primary']:.1%}")
        print(f"      GAP:            {perf['gap_primary']:.1%} {'❌ OVERFITTING!' if perf['gap_primary'] > 0.10 else '✓ OK'}")
        print(f"\n   Meta Model:")
        print(f"      Train Accuracy: {perf['train_acc_meta']:.1%}")
        print(f"      Test Accuracy:  {perf['test_acc_meta']:.1%}")
        print(f"      GAP:            {perf['gap_meta']:.1%} {'❌ OVERFITTING!' if perf['gap_meta'] > 0.10 else '✓ OK'}")

        print("\n2. FEATURE IMPORTANCE STABILITY")
        print("   " + "-" * 60)
        if feat['stable'] is not None:
            print(f"   Average CV:     {feat['avg_cv']:.2f}")
            print(f"   Max CV:         {feat['max_cv']:.2f}")
            print(f"   Status:         {feat['message']}")
        else:
            print(f"   {feat['message']}")

        print("\n3. LABEL DISTRIBUTION")
        print("   " + "-" * 60)
        print(f"   Train: Short={label['train_short_pct']:.1%}, Neutral={label['train_neutral_pct']:.1%}, Long={label['train_long_pct']:.1%}")
        print(f"   Test:  Short={label['test_short_pct']:.1%}, Neutral={label['test_neutral_pct']:.1%}, Long={label['test_long_pct']:.1%}")
        print(f"   Max Shift: {label['max_shift']:.1%} {'✓ Distributions match' if label['distributions_match'] else '❌ Distribution shift!'}")

        print("\n4. SAMPLE INDEPENDENCE")
        print("   " + "-" * 60)
        print(f"   Raw bars:       {sample['n_bars']}")
        print(f"   Autocorr (lag1): {sample['autocorr_lag1']:.3f}")
        print(f"   Effective N:    {sample['effective_n']} {'✓ Sufficient' if sample['sufficient_data'] else '❌ Insufficient!'}")

    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate actionable recommendations based on diagnostic results."""

        recommendations = []

        perf = results['performance_gap']
        feat = results['feature_stability']
        label = results['label_distribution']
        sample = results['sample_independence']

        # Performance gap recommendations
        if perf['gap_primary'] > 0.10:
            recommendations.append("🔴 PRIMARY MODEL OVERFITTING - Gap > 10%")
            recommendations.append("   → Increase regularization (C=0.1 or C=0.01 for LogisticRegression)")
            recommendations.append("   → Reduce max_depth for RandomForest (try max_depth=5)")
            recommendations.append("   → Add dropout if using neural networks")

        if perf['gap_meta'] > 0.10:
            recommendations.append("🔴 META MODEL OVERFITTING - Gap > 10%")
            recommendations.append("   → Meta model may be too complex for available data")
            recommendations.append("   → Consider simpler model (LogisticRegression instead of RandomForest)")

        # Feature stability recommendations
        if feat['stable'] is False:
            recommendations.append("🔴 UNSTABLE FEATURE IMPORTANCE")
            recommendations.append("   → Features changing importance across folds indicates overfitting")
            recommendations.append("   → Remove low-importance features")
            recommendations.append("   → Increase regularization to enforce feature selection")

        # Label distribution recommendations
        if not label['distributions_match']:
            recommendations.append("🔴 TRAIN/TEST DISTRIBUTION MISMATCH")
            recommendations.append("   → Market regime may have changed between train/test periods")
            recommendations.append("   → Consider walk-forward validation with periodic retraining")
            recommendations.append("   → Use stratified sampling to ensure balanced splits")

        # Sample size recommendations
        if not sample['sufficient_data']:
            recommendations.append("🔴 INSUFFICIENT EFFECTIVE SAMPLES")
            recommendations.append(f"   → Only {sample['effective_n']} independent samples (need >500)")
            recommendations.append("   → High autocorrelation means bars aren't independent")
            recommendations.append("   → Collect more historical data OR use simpler models")

        # General recommendations if no specific issues
        if len(recommendations) == 0:
            recommendations.append("✅ NO MAJOR OVERFITTING DETECTED")
            recommendations.append("   → Model appears reasonably generalized")
            recommendations.append("   → Still recommend walk-forward validation for production")

        return recommendations


def main():
    parser = argparse.ArgumentParser(description='Diagnose model overfitting')
    parser.add_argument('--symbols', nargs='+', help='Symbols to analyze')
    parser.add_argument('--tier', type=str, help='Tier to analyze (tier_1, tier_2, etc.)')

    args = parser.parse_args()

    # Get symbols
    symbols = []
    if args.symbols:
        symbols = args.symbols
    elif args.tier:
        from config.all_symbols import get_symbols_by_tier
        symbols = get_symbols_by_tier(args.tier)
    else:
        # Default to a few common symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']

    logger.info(f"Analyzing {len(symbols)} symbols...")

    # Run diagnostics
    all_results = []
    for symbol in symbols:
        diagnostic = OverfittingDiagnostic(symbol)
        results = diagnostic.run_full_diagnostic()

        if results:
            all_results.append(results)

            print("\nRECOMMENDATIONS:")
            print("=" * 80)
            for rec in results['recommendations']:
                print(rec)
            print("\n" + "=" * 80 + "\n")

    # Summary across all symbols
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("SUMMARY ACROSS ALL SYMBOLS")
        print("=" * 80 + "\n")

        avg_gap_primary = np.mean([r['performance_gap']['gap_primary'] for r in all_results])
        avg_gap_meta = np.mean([r['performance_gap']['gap_meta'] for r in all_results])

        print(f"Average Primary Model Gap: {avg_gap_primary:.1%}")
        print(f"Average Meta Model Gap:    {avg_gap_meta:.1%}")

        overfitted_count = sum(1 for r in all_results if r['performance_gap']['gap_primary'] > 0.10)
        print(f"\nSymbols with >10% overfitting: {overfitted_count}/{len(all_results)}")

        if overfitted_count > len(all_results) * 0.5:
            print("\n🔴 CRITICAL: Majority of models are overfitted!")
            print("   Recommended action: Retrain all models with increased regularization")


if __name__ == "__main__":
    main()
