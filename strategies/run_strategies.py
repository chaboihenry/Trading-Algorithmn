"""
Run All Trading Strategies
===========================
Execute all three strategies and display results
"""

import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent))

from pairs_trading import PairsTradingStrategy
from sentiment_trading import SentimentTradingStrategy
from volatility_trading import VolatilityTradingStrategy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run all strategies"""

    strategies = [
        ("Pairs Trading", PairsTradingStrategy()),
        ("Sentiment Trading", SentimentTradingStrategy()),
        ("Volatility Trading", VolatilityTradingStrategy())
    ]

    logger.info("="*60)
    logger.info("STRATEGY EXECUTION")
    logger.info("="*60)

    results = {}

    for name, strategy in strategies:
        try:
            logger.info(f"\n▶ {name}")
            signals = strategy.run()
            results[name] = len(signals)
            logger.info(f"✓ {name}: {len(signals)} signals")
        except Exception as e:
            logger.error(f"✗ {name}: {str(e)}")
            results[name] = 0

    logger.info("\n" + "="*60)
    logger.info("RESULTS")
    logger.info("="*60)

    for name, count in results.items():
        logger.info(f"{name}: {count} signals")

    total = sum(results.values())
    logger.info(f"\nTotal: {total} signals")
    logger.info("="*60)


if __name__ == "__main__":
    main()
