"""
Meta-Labeling for Bet Sizing

Once you have a primary model predicting direction (buy/sell),
meta-labeling builds a SECONDARY model that predicts:
"Should I actually take this trade, and if so, how much?"

This separates the problem into:
1. Direction model: Long or short? (your primary strategy)
2. Size model: What's the probability this bet will succeed?

The size model's output probability becomes the bet size.
"""

from RiskLabAI.labeling import meta_labeling
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MetaLabeler:
    """
    Creates meta-labels for bet sizing.

    The genius of meta-labeling:
    - Primary model determines SIDE (long/short)
    - Meta-labeling determines SIZE (how much to bet)
    - Size = probability of primary model being correct

    This reduces overfitting because:
    - Primary model doesn't need to learn everything
    - Meta-model specializes in filtering false positives

    Attributes:
        primary_model: Model that predicts trade direction
    """

    def __init__(self, primary_model=None):
        """
        Initialize meta-labeler.

        Args:
            primary_model: Trained model that outputs +1 (long) or -1 (short)
        """
        self.primary_model = primary_model
        logger.info("MetaLabeler initialized")

    def create_meta_labels(
        self,
        triple_barrier_events: pd.DataFrame,
        primary_predictions: pd.Series
    ) -> pd.DataFrame:
        """
        Create meta-labels from triple-barrier events.

        Args:
            triple_barrier_events: Output from TripleBarrierLabeler.label()
                                  Must have 'bin' column with labels
            primary_predictions: +1/-1 predictions from primary model
                                Aligned with triple_barrier_events index

        Returns:
            DataFrame with meta-labels:
            - 1: Primary model prediction was profitable
            - 0: Primary model prediction was unprofitable
        """
        logger.info(f"Creating meta-labels for {len(triple_barrier_events)} events")

        # Align predictions with events
        predictions_aligned = primary_predictions.reindex(triple_barrier_events.index)

        # Create meta-labels using RiskLabAI
        try:
            meta_labels = meta_labeling(
                events=triple_barrier_events,
                side=predictions_aligned
            )

            logger.info(f"Meta-labels created: {len(meta_labels)}")

            # Log distribution
            if 'bin' in meta_labels.columns:
                distribution = meta_labels['bin'].value_counts().to_dict()
                logger.info(f"Meta-label distribution: {distribution}")

            return meta_labels

        except Exception as e:
            logger.error(f"Error creating meta-labels: {e}")
            raise

    def get_bet_size(
        self,
        meta_model,
        features: pd.DataFrame,
        min_probability: float = 0.5,
        scale_factor: float = 1.0
    ) -> pd.Series:
        """
        Get bet sizes from meta-model probabilities.

        Args:
            meta_model: Trained classifier (RF, XGBoost, etc.)
            features: Feature matrix for prediction
            min_probability: Minimum probability to take a bet (default 0.5)
            scale_factor: Multiplier for bet sizes (default 1.0)

        Returns:
            Series of bet sizes (0 to scale_factor)
        """
        # Get probability of class 1 (profitable trade)
        try:
            probabilities = meta_model.predict_proba(features)[:, 1]
        except AttributeError:
            # Model doesn't support predict_proba
            logger.warning("Model doesn't support predict_proba, using predictions as probabilities")
            probabilities = meta_model.predict(features)

        # Convert to Series
        bet_sizes = pd.Series(probabilities, index=features.index)

        # Clip low probabilities to 0
        bet_sizes[bet_sizes < min_probability] = 0

        # Scale bet sizes
        bet_sizes = bet_sizes * scale_factor

        # Ensure within [0, scale_factor]
        bet_sizes = bet_sizes.clip(0, scale_factor)

        logger.debug(f"Bet sizes: mean={bet_sizes.mean():.3f}, "
                    f"non-zero={(bet_sizes > 0).sum()}/{len(bet_sizes)}")

        return bet_sizes

    def evaluate_meta_model(
        self,
        meta_model,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> dict:
        """
        Evaluate meta-model performance.

        Args:
            meta_model: Trained meta-model
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        predictions = meta_model.predict(X_test)
        probabilities = meta_model.predict_proba(X_test)[:, 1] if hasattr(meta_model, 'predict_proba') else predictions

        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, zero_division=0),
            'recall': recall_score(y_test, predictions, zero_division=0),
            'f1': f1_score(y_test, predictions, zero_division=0),
            'mean_probability': probabilities.mean(),
            'std_probability': probabilities.std()
        }

        logger.info(f"Meta-model evaluation: {metrics}")
        return metrics

    def adaptive_bet_sizing(
        self,
        probabilities: pd.Series,
        strategy: str = 'linear'
    ) -> pd.Series:
        """
        Apply adaptive bet sizing strategy.

        Args:
            probabilities: Model probabilities
            strategy: Sizing strategy:
                     - 'linear': Direct probability
                     - 'quadratic': Square of probability (more aggressive)
                     - 'kelly': Kelly criterion approximation

        Returns:
            Series of bet sizes
        """
        if strategy == 'linear':
            return probabilities

        elif strategy == 'quadratic':
            # More aggressive: square the probability
            return probabilities ** 2

        elif strategy == 'kelly':
            # Kelly criterion: f = (p * b - q) / b
            # Simplified for binary outcomes where b = 1
            # f = 2p - 1 (only bet when p > 0.5)
            kelly_sizes = 2 * probabilities - 1
            kelly_sizes = kelly_sizes.clip(lower=0)
            return kelly_sizes

        else:
            raise ValueError(f"Unknown strategy: {strategy}")
