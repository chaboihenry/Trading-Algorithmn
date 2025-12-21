"""
Meta-Labeling for Bet Sizing

Uses RiskLabAI's meta-labeling implementation to create labels for bet sizing.

Separates:
1. Direction model: Long or short? (primary strategy)
2. Size model: What's the probability this bet will succeed?
"""

from RiskLabAI.data.labeling import meta_labeling
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MetaLabeler:
    """
    Creates meta-labels for bet sizing using RiskLabAI.

    The genius of meta-labeling:
    - Primary model determines SIDE (long/short)
    - Meta-labeling determines SIZE (how much to bet)
    - Size = probability of primary model being correct

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
        events: pd.DataFrame,
        close: pd.Series
    ) -> pd.DataFrame:
        """
        Create meta-labels from triple-barrier events.

        Args:
            events: Output from TripleBarrierLabeler.label()
                   Must have barrier touch information
            close: Closing price series

        Returns:
            DataFrame with meta-labels
        """
        logger.info(f"Creating meta-labels for {len(events)} events")

        try:
            # Use RiskLabAI's meta_labeling function
            meta_labels = meta_labeling(events=events, close=close)

            # RiskLabAI returns 'Label' column, rename to 'bin' for consistency
            if 'Label' in meta_labels.columns:
                meta_labels['bin'] = meta_labels['Label']

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
        try:
            probabilities = meta_model.predict_proba(features)[:, 1]
        except AttributeError:
            logger.warning("Model doesn't support predict_proba, using predictions")
            probabilities = meta_model.predict(features)

        bet_sizes = pd.Series(probabilities, index=features.index)
        bet_sizes[bet_sizes < min_probability] = 0
        bet_sizes = bet_sizes * scale_factor
        bet_sizes = bet_sizes.clip(0, scale_factor)

        logger.debug(f"Bet sizes: mean={bet_sizes.mean():.3f}, "
                    f"non-zero={(bet_sizes > 0).sum()}/{len(bet_sizes)}")

        return bet_sizes
