import unittest
import pandas as pd
import numpy as np

from risklabai.cross_validation.purged_kfold import PurgedCrossValidator


class LeakageSplitTests(unittest.TestCase):
    def test_purged_splits_remove_overlap(self):
        index = pd.date_range("2024-01-01", periods=12, freq="min")
        X = pd.DataFrame({"feature": np.arange(len(index))}, index=index)
        y = pd.Series([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], index=index)

        # Each label event ends two bars later
        t1 = pd.Series(index + pd.Timedelta(minutes=2), index=index)

        cv = PurgedCrossValidator(n_splits=3, embargo_pct=0.0)

        for train_idx, test_idx in cv.iter_time_block_splits(X, y, t1):
            test_start = X.index[test_idx].min()
            test_end = t1.iloc[test_idx].max()
            train_start = X.index[train_idx]
            train_end = t1.iloc[train_idx]

            overlap = (train_start <= test_end) & (train_end >= test_start)
            self.assertFalse(overlap.any(), "Purged split contains overlapping train samples")


if __name__ == "__main__":
    unittest.main()
