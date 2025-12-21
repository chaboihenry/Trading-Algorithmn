"""
Test RiskLabAI Components

This script tests individual RiskLabAI components to ensure they work correctly.
Run this before attempting live/paper trading.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all RiskLabAI modules can be imported."""
    logger.info("Testing imports...")

    try:
        from risklabai.data_structures.bars import BarGenerator
        from risklabai.labeling.triple_barrier import TripleBarrierLabeler
        from risklabai.labeling.meta_labeling import MetaLabeler
        from risklabai.features.fractional_diff import FractionalDifferentiator
        from risklabai.sampling.cusum_filter import CUSUMEventFilter
        from risklabai.cross_validation.purged_kfold import PurgedCrossValidator
        from risklabai.features.feature_importance import FeatureImportanceAnalyzer
        from risklabai.portfolio.hrp import HRPPortfolio
        from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy

        logger.info("✓ All imports successful")
        return True

    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_cusum_filter():
    """Test CUSUM event filtering."""
    logger.info("\nTesting CUSUM filter...")

    try:
        from risklabai.sampling.cusum_filter import CUSUMEventFilter

        # Generate sample price data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='D')
        prices = pd.Series(100 * np.exp(np.random.randn(1000).cumsum() * 0.01), index=dates)

        # Test CUSUM filter
        cusum = CUSUMEventFilter()
        events = cusum.get_events(prices)

        logger.info(f"✓ CUSUM filter found {len(events)} events from {len(prices)} prices")
        return len(events) > 0

    except Exception as e:
        logger.error(f"✗ CUSUM filter test failed: {e}")
        return False


def test_fractional_diff():
    """Test fractional differentiation."""
    logger.info("\nTesting fractional differentiation...")

    try:
        from risklabai.features.fractional_diff import FractionalDifferentiator

        # Generate sample price data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=500, freq='D')
        prices = pd.Series(100 * np.exp(np.random.randn(500).cumsum() * 0.01), index=dates)

        # Test fractional differentiation
        frac_diff = FractionalDifferentiator(d=0.5)
        transformed = frac_diff.transform(prices)

        logger.info(f"✓ Fractional diff transformed {len(prices)} prices to {len(transformed)} values")
        return len(transformed) > 0

    except Exception as e:
        logger.error(f"✗ Fractional diff test failed: {e}")
        return False


def test_triple_barrier():
    """Test triple-barrier labeling."""
    logger.info("\nTesting triple-barrier labeling...")

    try:
        from risklabai.labeling.triple_barrier import TripleBarrierLabeler

        # Generate sample price data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
        prices = pd.Series(100 * np.exp(np.random.randn(300).cumsum() * 0.01), index=dates)

        # Test triple-barrier labeling
        labeler = TripleBarrierLabeler()
        labels = labeler.label(prices)

        logger.info(f"✓ Triple-barrier created {len(labels)} labels")
        logger.info(f"  Label distribution: {labels['bin'].value_counts().to_dict()}")
        return len(labels) > 0

    except Exception as e:
        logger.error(f"✗ Triple-barrier test failed: {e}")
        return False


def test_hrp_portfolio():
    """Test HRP portfolio optimization."""
    logger.info("\nTesting HRP portfolio optimization...")

    try:
        from risklabai.portfolio.hrp import HRPPortfolio

        # Generate sample returns for 5 assets
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        returns = pd.DataFrame(
            np.random.randn(252, 5) * 0.01,
            index=dates,
            columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        )

        # Test HRP
        hrp = HRPPortfolio()
        weights = hrp.optimize(returns)

        logger.info(f"✓ HRP optimized portfolio for {len(weights)} assets")
        logger.info(f"  Weights: {weights.to_dict()}")
        logger.info(f"  Sum of weights: {weights.sum():.4f}")

        return abs(weights.sum() - 1.0) < 0.01  # Should sum to 1

    except Exception as e:
        logger.error(f"✗ HRP test failed: {e}")
        return False


def test_full_strategy():
    """Test the complete RiskLabAI strategy."""
    logger.info("\nTesting full RiskLabAI strategy...")

    try:
        from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy

        # Download real data
        logger.info("Downloading SPY data...")
        spy = yf.download('SPY', start='2022-01-01', end='2024-01-01', progress=False)

        if spy.empty:
            logger.warning("Could not download data, using synthetic data")
            # Generate synthetic data
            np.random.seed(42)
            dates = pd.date_range(start='2022-01-01', periods=500, freq='D')
            spy = pd.DataFrame({
                'open': 400 + np.random.randn(500).cumsum(),
                'high': 405 + np.random.randn(500).cumsum(),
                'low': 395 + np.random.randn(500).cumsum(),
                'close': 400 + np.random.randn(500).cumsum(),
                'volume': np.random.randint(50000000, 100000000, 500)
            }, index=dates)
        else:
            # Handle MultiIndex columns from yfinance
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)

        # Lowercase column names
        spy.columns = spy.columns.str.lower()

        logger.info(f"Testing with {len(spy)} bars of data")

        # Initialize strategy
        strategy = RiskLabAIStrategy(
            profit_taking=2.0,
            stop_loss=2.0,
            max_holding=10
        )

        # Train
        logger.info("Training models...")
        results = strategy.train(spy, min_samples=50)

        if not results['success']:
            logger.warning(f"Training failed: {results.get('reason')}")
            return False

        logger.info(f"✓ Training successful!")
        logger.info(f"  Primary accuracy: {results['primary_accuracy']:.3f}")
        logger.info(f"  Meta accuracy: {results['meta_accuracy']:.3f}")

        # Test prediction
        logger.info("Testing prediction...")
        signal, bet_size = strategy.predict(spy.tail(100))

        logger.info(f"✓ Prediction: signal={signal}, bet_size={bet_size:.3f}")

        return True

    except Exception as e:
        logger.error(f"✗ Full strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("=" * 70)
    logger.info("RISKLABAI COMPONENT TESTS")
    logger.info("=" * 70)

    tests = [
        ("Imports", test_imports),
        ("CUSUM Filter", test_cusum_filter),
        ("Fractional Differentiation", test_fractional_diff),
        ("Triple-Barrier Labeling", test_triple_barrier),
        ("HRP Portfolio", test_hrp_portfolio),
        ("Full Strategy", test_full_strategy)
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            logger.error(f"Test {name} crashed: {e}")
            results[name] = False

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)

    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{name:.<50} {status}")

    passed = sum(results.values())
    total = len(results)

    logger.info("=" * 70)
    logger.info(f"TOTAL: {passed}/{total} tests passed")
    logger.info("=" * 70)

    if passed == total:
        logger.info("\n🎉 All tests passed! RiskLabAI integration is ready.")
        return 0
    else:
        logger.warning(f"\n⚠️  {total - passed} test(s) failed. Review errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
