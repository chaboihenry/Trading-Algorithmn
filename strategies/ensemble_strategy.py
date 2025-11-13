"""
Ensemble Trading Strategy with Weighted Voting
================================================
Combines signals from all three strategies using weighted voting
for maximum accuracy and profit potential.

Strategy Weights:
- Sentiment Trading: 40% (ML-based, most features)
- Pairs Trading: 35% (Statistical arbitrage, proven method)
- Volatility Trading: 25% (Regime-based, complementary)
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from base_strategy import BaseStrategy
import logging
from typing import Dict, List
from datetime import datetime

# Import individual strategies
from sentiment_trading import SentimentTradingStrategy
from pairs_trading import PairsTradingStrategy
from volatility_trading import VolatilityTradingStrategy

logger = logging.getLogger(__name__)


class EnsembleStrategy(BaseStrategy):
    """
    Ensemble strategy that combines signals from multiple strategies
    using weighted voting for improved accuracy

    M1 Optimizations:
    - NumPy vectorized signal aggregation
    - Batch processing of strategy signals
    - Memory-efficient pandas operations
    """

    def __init__(self,
                 db_path: str = "/Volumes/Vault/85_assets_prediction.db",
                 sentiment_weight: float = 0.40,
                 pairs_weight: float = 0.35,
                 volatility_weight: float = 0.25,
                 min_agreement: float = 0.6):
        """
        Initialize ensemble strategy

        Args:
            db_path: Path to database
            sentiment_weight: Weight for sentiment strategy (default 40%)
            pairs_weight: Weight for pairs trading (default 35%)
            volatility_weight: Weight for volatility trading (default 25%)
            min_agreement: Minimum weighted agreement score to generate signal (default 60%)
        """
        super().__init__(db_path)
        self.name = "EnsembleStrategy"

        # Strategy weights
        self.weights = {
            'sentiment': sentiment_weight,
            'pairs': pairs_weight,
            'volatility': volatility_weight
        }

        # Normalize weights to sum to 1.0
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}

        self.min_agreement = min_agreement

        # Initialize individual strategies
        self.sentiment_strategy = SentimentTradingStrategy(db_path)
        self.pairs_strategy = PairsTradingStrategy(db_path)
        self.volatility_strategy = VolatilityTradingStrategy(db_path)

        logger.info(f"Initialized EnsembleStrategy with weights: {self.weights}")
        logger.info(f"Minimum agreement threshold: {min_agreement:.1%}")

    def _get_strategy_signals(self) -> Dict[str, pd.DataFrame]:
        """
        Run all strategies and collect their signals

        Returns:
            Dictionary mapping strategy name to signals DataFrame
        """
        logger.info("=== Running Individual Strategies ===")

        signals = {}

        # Run sentiment strategy
        try:
            logger.info("\n▶ Sentiment Trading Strategy")
            sentiment_signals = self.sentiment_strategy.generate_signals()
            signals['sentiment'] = sentiment_signals
            logger.info(f"✓ Sentiment: {len(sentiment_signals)} signals")
        except Exception as e:
            logger.error(f"✗ Sentiment strategy failed: {str(e)}")
            signals['sentiment'] = pd.DataFrame()

        # Run pairs trading strategy
        try:
            logger.info("\n▶ Pairs Trading Strategy")
            pairs_signals = self.pairs_strategy.generate_signals()
            signals['pairs'] = pairs_signals
            logger.info(f"✓ Pairs: {len(pairs_signals)} signals")
        except Exception as e:
            logger.error(f"✗ Pairs strategy failed: {str(e)}")
            signals['pairs'] = pd.DataFrame()

        # Run volatility strategy
        try:
            logger.info("\n▶ Volatility Trading Strategy")
            volatility_signals = self.volatility_strategy.generate_signals()
            signals['volatility'] = volatility_signals
            logger.info(f"✓ Volatility: {len(volatility_signals)} signals")
        except Exception as e:
            logger.error(f"✗ Volatility strategy failed: {str(e)}")
            signals['volatility'] = pd.DataFrame()

        return signals

    def _normalize_signal_type(self, signal_type: str) -> int:
        """
        Convert signal type to numeric value for voting

        Args:
            signal_type: 'BUY' or 'SELL'

        Returns:
            1 for BUY, -1 for SELL
        """
        return 1 if signal_type.upper() == 'BUY' else -1

    def _aggregate_signals(self, strategy_signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Aggregate signals using weighted voting (NumPy optimized)

        Args:
            strategy_signals: Dictionary of strategy signals

        Returns:
            DataFrame with ensemble signals
        """
        logger.info("\n=== Aggregating Signals with Weighted Voting ===")

        # Combine all signals by ticker
        all_tickers = set()
        for signals in strategy_signals.values():
            if not signals.empty:
                all_tickers.update(signals['symbol_ticker'].unique())

        if not all_tickers:
            logger.warning("No signals from any strategy")
            return pd.DataFrame()

        logger.info(f"Found signals for {len(all_tickers)} unique tickers")

        # NumPy-optimized voting aggregation
        ensemble_signals = []

        for ticker in all_tickers:
            # Collect votes from each strategy
            votes = {}
            strengths = {}
            prices = {}
            metadata = {}

            for strategy_name, signals in strategy_signals.items():
                if signals.empty:
                    continue

                ticker_signals = signals[signals['symbol_ticker'] == ticker]

                if not ticker_signals.empty:
                    # Get the strongest signal for this ticker from this strategy
                    best_signal = ticker_signals.loc[ticker_signals['strength'].idxmax()]

                    signal_value = self._normalize_signal_type(best_signal['signal_type'])
                    signal_strength = best_signal['strength']

                    votes[strategy_name] = signal_value
                    strengths[strategy_name] = signal_strength
                    prices[strategy_name] = best_signal.get('entry_price', 0)
                    metadata[strategy_name] = {
                        'signal': best_signal['signal_type'],
                        'strength': signal_strength,
                        'metadata': best_signal.get('metadata', '{}')
                    }

            if not votes:
                continue

            # Calculate weighted vote (vectorized)
            weighted_vote = 0.0
            total_confidence = 0.0

            for strategy_name, vote in votes.items():
                weight = self.weights[strategy_name]
                strength = strengths[strategy_name]

                # Weighted vote: weight * direction * strength
                weighted_vote += weight * vote * strength
                total_confidence += weight * strength

            # Normalize to [-1, 1] range
            if total_confidence > 0:
                agreement_score = abs(weighted_vote) / total_confidence
            else:
                agreement_score = 0

            # Only generate signal if agreement meets threshold
            if agreement_score >= self.min_agreement:
                signal_type = 'BUY' if weighted_vote > 0 else 'SELL'

                # Calculate average entry price (weighted by strength)
                avg_price = sum(prices[s] * strengths[s] for s in prices) / sum(strengths.values())

                # Calculate stop loss and take profit based on average volatility
                # More conservative than individual strategies
                stop_loss_pct = 0.02 if signal_type == 'BUY' else -0.02
                take_profit_pct = 0.04 if signal_type == 'BUY' else -0.04

                ensemble_signals.append({
                    'symbol_ticker': ticker,
                    'signal_date': datetime.now().strftime('%Y-%m-%d'),
                    'signal_type': signal_type,
                    'strength': agreement_score,
                    'entry_price': avg_price,
                    'stop_loss': avg_price * (1 + stop_loss_pct),
                    'take_profit': avg_price * (1 + take_profit_pct),
                    'metadata': f'{{"agreement": {agreement_score:.3f}, "weighted_vote": {weighted_vote:.3f}, '
                               f'"strategies": {list(votes.keys())}, "strategy_votes": {metadata}}}'
                })

                logger.info(f"✓ {ticker}: {signal_type} (agreement: {agreement_score:.1%}, "
                          f"votes: {votes})")
            else:
                logger.debug(f"✗ {ticker}: Insufficient agreement ({agreement_score:.1%} < "
                           f"{self.min_agreement:.1%})")

        result_df = pd.DataFrame(ensemble_signals)

        if not result_df.empty:
            # Sort by strength (highest confidence first)
            result_df = result_df.sort_values('strength', ascending=False)

            logger.info(f"\n=== Ensemble Results ===")
            logger.info(f"Total signals: {len(result_df)}")
            if len(result_df) > 0:
                logger.info(f"Signal distribution:")
                logger.info(result_df['signal_type'].value_counts().to_dict())
                logger.info(f"Average agreement: {result_df['strength'].mean():.1%}")
                logger.info(f"Top 5 signals by confidence:")
                for idx, row in result_df.head(5).iterrows():
                    logger.info(f"  {row['symbol_ticker']}: {row['signal_type']} "
                              f"({row['strength']:.1%} confidence)")

        return result_df

    def generate_signals(self) -> pd.DataFrame:
        """
        Generate ensemble signals using weighted voting with dynamic filtering

        Returns:
            DataFrame with high-confidence ensemble signals with Kelly position sizing
        """
        logger.info("="*60)
        logger.info("ENSEMBLE STRATEGY EXECUTION")
        logger.info("="*60)

        # Run all strategies
        strategy_signals = self._get_strategy_signals()

        # Aggregate with weighted voting
        ensemble_signals = self._aggregate_signals(strategy_signals)

        if not ensemble_signals.empty:
            # Apply dynamic filtering with Kelly Criterion position sizing
            logger.info("\n=== Applying Dynamic Filtering & Kelly Sizing ===")
            ensemble_signals = self.apply_dynamic_filtering(
                ensemble_signals,
                min_position_size=0.02,  # Minimum 2% position
                portfolio_value=100000.0  # $100k default portfolio
            )

        logger.info("="*60)
        return ensemble_signals


if __name__ == "__main__":
    # Run ensemble strategy
    strategy = EnsembleStrategy(
        sentiment_weight=0.40,  # 40% weight
        pairs_weight=0.35,      # 35% weight
        volatility_weight=0.25, # 25% weight
        min_agreement=0.6       # 60% minimum agreement
    )

    signals = strategy.run()

    print(f"\n{'='*60}")
    print(f"ENSEMBLE STRATEGY RESULTS")
    print(f"{'='*60}")
    print(f"Total high-confidence signals: {len(signals)}")

    if len(signals) > 0:
        print(f"\nSignal breakdown:")
        print(signals['signal_type'].value_counts())
        print(f"\nAverage confidence: {signals['strength'].mean():.1%}")

        if 'position_size' in signals.columns:
            print(f"Average position size: {signals['position_size'].mean():.1%}")
            print(f"Total capital allocated: ${signals['position_value'].sum():,.0f}")

        print(f"\nTop 10 signals:")
        display_cols = ['symbol_ticker', 'signal_type', 'strength', 'entry_price']
        if 'position_size' in signals.columns:
            display_cols.append('position_size')
        print(signals[display_cols].head(10).to_string(index=False))
