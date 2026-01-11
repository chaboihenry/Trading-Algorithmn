"""
Meta-Labeling for Bet Sizing

Meta-labeling is a two-stage prediction approach that separates:
1. Primary Model (Direction): What direction should we trade? (Long/Short/Neutral)
2. Meta Model (Size/Confidence): Will the primary model be correct?

================================================================================
CRITICAL CONCEPT: Meta-Labeling vs. Traditional Ensembles
================================================================================

Traditional Ensemble:
- Multiple models predict the SAME thing (e.g., price direction)
- Combine predictions via voting or averaging
- Example: 5 models all predict Long/Short

Meta-Labeling:
- Primary model predicts DIRECTION (Long/Short)
- Meta model predicts CORRECTNESS (Will primary be right?)
- Meta model learns WHEN to trust primary model

Key Difference:
- Ensemble: "What will happen?"
- Meta-labeling: "When should I trust my prediction?"

Benefits of Meta-Labeling:
1. Better bet sizing: Size bets based on confidence
2. Risk management: Skip trades when primary likely wrong
3. Improved Sharpe: Fewer bad trades, similar good trades
4. Interpretability: See when model is confident

Example:
    Primary Model predicts: +1 (Long)
    Meta Model predicts: P(correct) = 0.35 (low confidence)
    Action: Skip trade (bet size = 0)

    Primary Model predicts: +1 (Long)
    Meta Model predicts: P(correct) = 0.75 (high confidence)
    Action: Take trade with larger bet size

Expected Meta Model Performance:
- Accuracy: 45-60% (modestly better than random)
- NOT 90%+ (would indicate data leakage)
- NOT 4% (indicates wrong label creation)

If meta model achieves 55% accuracy at predicting when primary is correct,
it can significantly improve portfolio performance by:
- Avoiding 45% of bad trades
- Keeping 55% of good trades
- Improving overall Sharpe ratio
================================================================================
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

    def create_meta_labels_from_predictions(
        self,
        primary_predictions: np.ndarray,
        actual_labels: np.ndarray
    ) -> np.ndarray:
        """
        Create meta labels from primary model predictions.

        CRITICAL CONCEPT: Meta-labeling predicts WHEN primary model is correct, not WHAT to trade.

        The meta model learns:
        - "Given these features, will the primary model be right?"
        - NOT "What direction should we trade?"

        This is the key difference from traditional ensemble methods.

        Args:
            primary_predictions: What primary model predicted (e.g., -1, 0, 1)
            actual_labels: What actually happened (e.g., -1, 0, 1)

        Returns:
            Binary array: 1 = primary was correct, 0 = primary was wrong

        Examples:
            primary_predictions = [1, -1, 0, 1, -1]
            actual_labels       = [1,  1, 0, -1, -1]
            meta_labels         = [1,  0, 1,  0,  1]  # Matches where predictions were right
        """
        # CRITICAL: Meta label is 1 when prediction MATCHES actual
        meta_labels = (primary_predictions == actual_labels).astype(int)

        # Log for sanity check
        accuracy = meta_labels.mean()
        n_correct = meta_labels.sum()
        n_wrong = len(meta_labels) - n_correct

        logger.info("")
        logger.info("=" * 60)
        logger.info("META-LABELING: Primary Model Performance")
        logger.info("=" * 60)
        logger.info(f"Primary model accuracy: {accuracy*100:.1f}%")
        logger.info(f"Meta labels created:")
        logger.info(f"  Correct predictions (1): {n_correct} ({n_correct/len(meta_labels)*100:.1f}%)")
        logger.info(f"  Wrong predictions (0):   {n_wrong} ({n_wrong/len(meta_labels)*100:.1f}%)")
        logger.info("=" * 60)

        # Sanity checks
        if accuracy < 0.30:
            logger.warning("")
            logger.warning("⚠️  LOW PRIMARY MODEL ACCURACY")
            logger.warning(f"    Current: {accuracy:.1%}")
            logger.warning(f"    Expected: >40% for useful meta-labeling")
            logger.warning("")
            logger.warning("    ISSUE: Primary model is performing poorly")
            logger.warning("    ACTION: Check primary model training, features, or labels")
            logger.warning("")
        elif accuracy > 0.70:
            logger.warning("")
            logger.warning("⚠️  VERY HIGH PRIMARY MODEL ACCURACY")
            logger.warning(f"    Current: {accuracy:.1%}")
            logger.warning(f"    Expected: 40-65% (challenging but learnable)")
            logger.warning("")
            logger.warning("    POSSIBLE ISSUE: Data leakage")
            logger.warning("    ACTION: Check for look-ahead bias or feature leakage")
            logger.warning("")
        else:
            logger.info(f"✓ Good primary accuracy: {accuracy:.1%} (target: 40-65%)")

        # Balance check
        balance = abs(0.5 - accuracy)
        if balance < 0.05:
            logger.info(f"✓ Good meta-label balance: {accuracy:.1%} vs {1-accuracy:.1%}")
        elif accuracy > 0.55 or accuracy < 0.45:
            logger.warning(f"⚠️ Imbalanced meta-labels: {accuracy:.1%} correct vs {1-accuracy:.1%} wrong")
            logger.warning(f"   Meta model may struggle to learn meaningful patterns")

        return meta_labels

    def create_meta_labels(
        self,
        events: pd.DataFrame,
        close: pd.Series
    ) -> pd.DataFrame:
        """
        Create meta-labels from triple-barrier events (LEGACY METHOD).

        NOTE: This uses RiskLabAI's original meta_labeling approach.
        For modern meta-labeling with XGBoost, use create_meta_labels_from_predictions().

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
