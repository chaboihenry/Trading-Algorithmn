"""
Purged K-Fold Cross-Validation using RiskLabAI

Prevents information leakage by purging overlapping samples.
"""

from RiskLabAI.backtest.validation.purged_kfold import PurgedKFold as RiskLabPurgedKFold
import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, Iterator, Tuple
import logging
from sklearn.metrics import get_scorer

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

    def _time_block_purged_splits(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        samples_info: Union[pd.DataFrame, pd.Series]
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate time-contiguous splits and purge overlaps to reduce leakage.

        This avoids stratified shuffling, which can make test windows span
        most of the timeline and purge nearly all training samples.
        """
        if isinstance(samples_info, pd.DataFrame):
            if 't1' in samples_info.columns:
                times = samples_info['t1']
            else:
                raise ValueError("DataFrame must have 't1' column")
        else:
            times = samples_info
        indices = np.arange(len(X))
        start_times = X.index

        # Build contiguous blocks
        block_sizes = np.full(self.n_splits, len(X) // self.n_splits, dtype=int)
        block_sizes[:len(X) % self.n_splits] += 1
        blocks = []
        start = 0
        for size in block_sizes:
            end = start + size
            blocks.append(np.arange(start, end))
            start = end

        # Merge blocks until each has all classes (or only one block remains)
        merged_blocks = []
        i = 0
        all_classes = set(np.unique(y))
        while i < len(blocks):
            current = blocks[i]
            current_classes = set(np.unique(y.iloc[current]))
            while current_classes != all_classes and i + 1 < len(blocks):
                i += 1
                current = np.concatenate([current, blocks[i]])
                current_classes = set(np.unique(y.iloc[current]))
            merged_blocks.append(current)
            i += 1

        if len(merged_blocks) < self.n_splits:
            logger.info(
                f"Reduced folds from {self.n_splits} to {len(merged_blocks)} to preserve classes in time blocks"
            )

        for test_idx in merged_blocks:
            test_idx = np.array(test_idx)
            test_start = start_times[test_idx].min()
            test_end = times.iloc[test_idx].max()

            train_mask = np.ones(len(X), dtype=bool)
            train_mask[test_idx] = False

            overlap = (start_times <= test_end) & (times >= test_start)
            train_mask &= ~overlap

            embargo = int(len(X) * self.embargo_pct)
            if embargo > 0:
                test_end_pos = test_idx.max()
                embargo_end = min(len(X), test_end_pos + embargo + 1)
                train_mask[test_end_pos:embargo_end] = False

            train_idx = indices[train_mask]
            yield train_idx, test_idx

    def iter_time_block_splits(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        samples_info: Union[pd.DataFrame, pd.Series]
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Public iterator for time-blocked purged splits.
        """
        return self._time_block_purged_splits(X, y, samples_info)

    def cross_val_score_purged_stratified(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        samples_info: Union[pd.DataFrame, pd.Series],
        scoring: str = 'balanced_accuracy'
    ) -> np.ndarray:
        """
        Time-blocked K-fold with purging + embargo to reduce leakage.
        """
        def _is_zero_based_contiguous(class_set: set) -> bool:
            if not class_set:
                return False
            class_list = sorted(class_set)
            return class_list[0] == 0 and class_list[-1] == len(class_list) - 1

        full_classes = set(np.unique(y))
        expected_classes = set(range(len(full_classes)))
        require_full = full_classes == expected_classes

        scorer = get_scorer(scoring)
        scores = []
        skipped = 0

        for fold, (train_idx, test_idx) in enumerate(
            self._time_block_purged_splits(X, y, samples_info), 1
        ):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            train_classes = set(np.unique(y_train))
            test_classes = set(np.unique(y_test))
            if require_full:
                if train_classes != expected_classes or test_classes != expected_classes:
                    skipped += 1
                    logger.info(
                        f"Fold {fold}/{self.n_splits}: skipped (classes train={sorted(train_classes)} "
                        f"test={sorted(test_classes)})"
                    )
                    continue
                if (not _is_zero_based_contiguous(train_classes) or
                        not _is_zero_based_contiguous(test_classes)):
                    skipped += 1
                    logger.info(
                        f"Fold {fold}/{self.n_splits}: skipped (non-contiguous classes train={sorted(train_classes)} "
                        f"test={sorted(test_classes)})"
                    )
                    continue
            elif len(train_classes) < 2 or len(test_classes) < 2:
                skipped += 1
                logger.info(f"Fold {fold}/{self.n_splits}: skipped (missing classes)")
                continue

            model.fit(X_train, y_train)
            score = scorer(model, X_test, y_test)
            scores.append(score)

            logger.info(f"Fold {fold}/{self.n_splits}: {scoring}={score:.4f}")

        scores = np.array(scores)
        if len(scores) == 0:
            raise ValueError("All stratified purged folds were skipped due to missing classes.")

        logger.info(f"CV {scoring}: {scores.mean():.4f} ± {scores.std():.4f} (skipped: {skipped})")
        return scores
