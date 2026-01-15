import unittest
import pandas as pd

from risklabai.labeling.triple_barrier import TripleBarrierLabeler


class TripleBarrierTests(unittest.TestCase):
    def _labeler_with_constant_vol(self, vol: float) -> TripleBarrierLabeler:
        labeler = TripleBarrierLabeler(
            profit_taking_mult=1.0,
            stop_loss_mult=1.0,
            max_holding_period=3,
            volatility_lookback=2,
        )
        labeler.get_volatility = lambda close, span=None: pd.Series(vol, index=close.index)
        return labeler

    def test_profit_first_touch(self):
        index = pd.date_range("2024-01-01", periods=4, freq="min")
        close = pd.Series([100.0, 101.0, 100.0, 100.0], index=index)
        events = pd.DataFrame(index=index[:1])
        events["t1"] = index[3]

        labeler = self._labeler_with_constant_vol(0.01)
        labels = labeler.label(close, events=events)

        self.assertEqual(labels.iloc[0]["bin"], 1)
        self.assertEqual(labels.iloc[0]["touched"], index[1])

    def test_stop_first_touch(self):
        index = pd.date_range("2024-01-01", periods=4, freq="min")
        close = pd.Series([100.0, 99.0, 102.0, 102.0], index=index)
        events = pd.DataFrame(index=index[:1])
        events["t1"] = index[3]

        labeler = self._labeler_with_constant_vol(0.01)
        labels = labeler.label(close, events=events)

        self.assertEqual(labels.iloc[0]["bin"], -1)
        self.assertEqual(labels.iloc[0]["touched"], index[1])

    def test_timeout_label(self):
        index = pd.date_range("2024-01-01", periods=4, freq="min")
        close = pd.Series([100.0, 100.2, 100.1, 100.15], index=index)
        events = pd.DataFrame(index=index[:1])
        events["t1"] = index[3]

        labeler = self._labeler_with_constant_vol(0.01)
        labels = labeler.label(close, events=events)

        self.assertEqual(labels.iloc[0]["bin"], 0)
        self.assertEqual(labels.iloc[0]["touched"], index[3])


if __name__ == "__main__":
    unittest.main()
