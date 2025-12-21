"""
Purged K-Fold Cross-Validation

Standard K-Fold fails for time series because:
- Training folds leak information to test folds
- Labels overlap in time (triple-barrier spans multiple bars)

Purged K-Fold:
1. PURGES training samples that overlap with test samples
2. EMBARGOES additional samples after test period

This prevents lookahead bias in backtests.
"""

from RiskLabAI.cross_validation import (
    PurgedKFold,
    CombinatorialPurgedKFold
)
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PurgedCrossValidator:
    """
    Cross-validation with purging and embargo.

    Why standard CV fails:
    - Fold 1 train: [1, 2, 3], test: [4, 5]
    - But label at t=3 might depend on price at t=5!
    - Information leakage → overly optimistic results

    Purged CV solution:
    - Remove training samples that overlap with test times
    - Add embargo period after test to prevent leakage

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
        samples_info: pd.DataFrame
    ) -> PurgedKFold:
        """
        Get purged K-fold cross-validator.

        Args:
            samples_info: DataFrame with 't1' column (label end times)
                         Index should be label start times

        Returns:
            Configured PurgedKFold object
        """
        if 't1' not in samples_info.columns:
            raise ValueError("samples_info must have 't1' column with label end times")

        logger.info(f"Creating PurgedKFold with {len(samples_info)} samples")

        cv = PurgedKFold(
            n_splits=self.n_splits,
            samples_info_sets=samples_info['t1'],
            pct_embargo=self.embargo_pct
        )

        return cv

    def get_combinatorial_cv(
        self,
        samples_info: pd.DataFrame,
        n_test_splits: int = 2
    ):
        """
        Get combinatorial purged K-fold for more robust evaluation.

        Creates all possible train/test combinations from groups.
        More paths = more confidence in results.

        Args:
            samples_info: DataFrame with 't1' column
            n_test_splits: Number of groups in each test set

        Returns:
            CombinatorialPurgedKFold object
        """
        if 't1' not in samples_info.columns:
            raise ValueError("samples_info must have 't1' column with label end times")

        logger.info(f"Creating CombinatorialPurgedKFold with {len(samples_info)} samples")

        cv = CombinatorialPurgedKFold(
            n_splits=self.n_splits,
            n_test_splits=n_test_splits,
            samples_info_sets=samples_info['t1'],
            pct_embargo=self.embargo_pct
        )

        return cv

    def validate_no_leakage(
        self,
        cv,
        samples_info: pd.DataFrame,
        X: pd.DataFrame
    ) -> dict:
        """
        Validate that there's no information leakage in CV splits.

        Args:
            cv: Cross-validator object
            samples_info: DataFrame with 't1' column
            X: Feature matrix (for indexing)

        Returns:
            Dictionary with validation results
        """
        leakage_count = 0
        total_splits = 0

        for train_idx, test_idx in cv.split(X):
            total_splits += 1

            # Get train and test times
            train_times = samples_info.iloc[train_idx]
            test_times = samples_info.iloc[test_idx]

            # Check if any training labels overlap with test period
            test_start = test_times.index.min()
            test_end = test_times['t1'].max()

            # Training samples that start before test_end and end after test_start
            overlapping = train_times[
                (train_times.index < test_end) &
                (train_times['t1'] > test_start)
            ]

            if len(overlapping) > 0:
                leakage_count += 1
                logger.warning(f"Split {total_splits}: {len(overlapping)} overlapping samples found!")

        results = {
            'total_splits': total_splits,
            'splits_with_leakage': leakage_count,
            'leakage_free': leakage_count == 0
        }

        if results['leakage_free']:
            logger.info("✓ No information leakage detected in CV splits")
        else:
            logger.error(f"✗ Leakage detected in {leakage_count}/{total_splits} splits!")

        return results

    def cross_val_score_purged(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        samples_info: pd.DataFrame,
        scoring: str = 'accuracy',
        sample_weight: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        Perform cross-validation with purged K-fold.

        Args:
            model: Scikit-learn compatible model
            X: Feature matrix
            y: Labels
            samples_info: DataFrame with 't1' column
            scoring: Scoring metric
            sample_weight: Optional sample weights

        Returns:
            Array of scores for each fold
        """
        from sklearn.metrics import get_scorer

        cv = self.get_cv(samples_info)
        scorer = get_scorer(scoring)

        scores = []

        for fold, (train_idx, test_idx) in enumerate(cv.split(X), 1):
            # Get train/test data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Get sample weights if provided
            if sample_weight is not None:
                sw_train = sample_weight.iloc[train_idx]
                model.fit(X_train, y_train, sample_weight=sw_train)
            else:
                model.fit(X_train, y_train)

            # Score on test set
            score = scorer(model, X_test, y_test)
            scores.append(score)

            logger.info(f"Fold {fold}/{self.n_splits}: {scoring}={score:.4f}")

        scores = np.array(scores)
        logger.info(f"CV {scoring}: {scores.mean():.4f} ± {scores.std():.4f}")

        return scores

    def get_split_info(
        self,
        cv,
        X: pd.DataFrame,
        samples_info: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get detailed information about CV splits.

        Args:
            cv: Cross-validator
            X: Feature matrix
            samples_info: DataFrame with 't1' column

        Returns:
            DataFrame with split statistics
        """
        split_info = []

        for fold, (train_idx, test_idx) in enumerate(cv.split(X), 1):
            train_times = samples_info.iloc[train_idx]
            test_times = samples_info.iloc[test_idx]

            info = {
                'fold': fold,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'train_start': train_times.index.min(),
                'train_end': train_times.index.max(),
                'test_start': test_times.index.min(),
                'test_end': test_times.index.max()
            }

            split_info.append(info)

        df = pd.DataFrame(split_info)
        logger.info(f"\nCV Split Information:\n{df}")

        return df
