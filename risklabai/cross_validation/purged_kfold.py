"""
Purged K-Fold Cross-Validation using RiskLabAI

Prevents information leakage by purging overlapping samples.
"""

from RiskLabAI.backtest.validation.purged_kfold import PurgedKFold as RiskLabPurgedKFold
import pandas as pd
import numpy as np
from typing import Optional, Union, Dict
import logging

logger = logging.getLogger(__name__)


class PurgedCrossValidator:
    """
    Cross-validation with purging and embargo using RiskLabAI.

    Why standard CV fails:
    - Training folds leak information to test folds
    - Labels overlap in time

    Purged CV solution:
    - Removes training samples that overlap with test samples
    - Adds embargo period after test to prevent leakage

    Attributes:
        n_splits: Number of cross-validation folds
        embargo_pct: Percentage of data to embargo after test
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01
    ):
        """
        Initialize purged cross-validator.

        Args:
            n_splits: Number of folds
            embargo_pct: Fraction of samples to embargo (0.01 = 1%)
        """
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

        logger.info(f"PurgedCrossValidator initialized: n_splits={n_splits}, "
                   f"embargo_pct={embargo_pct}")

    def get_cv(
        self,
        samples_info: Union[pd.DataFrame, pd.Series]
    ):
        """
        Get purged K-fold cross-validator.

        Args:
            samples_info: DataFrame with 't1' column (label end times)
                         OR Series of label end times

        Returns:
            Configured PurgedKFold object
        """
        logger.info(f"Creating PurgedKFold for {len(samples_info)} samples")

        # Convert samples_info to the format RiskLabAI expects
        if isinstance(samples_info, pd.DataFrame):
            if 't1' in samples_info.columns:
                times = samples_info['t1']
            else:
                raise ValueError("DataFrame must have 't1' column")
        else:
            times = samples_info

        cv = RiskLabPurgedKFold(
            n_splits=self.n_splits,
            times=times,
            embargo=self.embargo_pct
        )

        return cv

    def cross_val_score_purged(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        samples_info: Union[pd.DataFrame, pd.Series],
        scoring: str = 'accuracy'
    ) -> np.ndarray:
        """
        Perform cross-validation with purged K-fold.

        Args:
            model: Scikit-learn compatible model
            X: Feature matrix
            y: Labels
            samples_info: DataFrame or Series with label end times
            scoring: Scoring metric

        Returns:
            Array of scores for each fold
        """
        from sklearn.metrics import get_scorer

        cv = self.get_cv(samples_info)
        scorer = get_scorer(scoring)

        scores = []

        for fold, (train_idx, test_idx) in enumerate(cv.split(X), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            score = scorer(model, X_test, y_test)
            scores.append(score)

            logger.info(f"Fold {fold}/{self.n_splits}: {scoring}={score:.4f}")

        scores = np.array(scores)
        logger.info(f"CV {scoring}: {scores.mean():.4f} ± {scores.std():.4f}")

        return scores
