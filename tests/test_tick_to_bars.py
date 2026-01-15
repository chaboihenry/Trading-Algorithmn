import unittest

from data.tick_to_bars import generate_bars_from_ticks


class TickToBarsTests(unittest.TestCase):
    def test_bar_boundaries_with_unit_sizes(self):
        ticks = [
            (0, 100.0, 1),
            (1, 101.0, 1),
            (2, 102.0, 1),
            (3, 101.0, 1),
            (4, 100.0, 1),
            (5, 99.0, 1),
        ]

        bars = generate_bars_from_ticks(ticks, threshold=3, ewma_alpha=0.5)

        self.assertEqual(len(bars), 2)
        self.assertEqual(bars[0]["bar_start"], 0)
        self.assertEqual(bars[0]["bar_end"], 2)
        self.assertAlmostEqual(bars[0]["imbalance"], 1.0)

        self.assertEqual(bars[1]["bar_start"], 3)
        self.assertEqual(bars[1]["bar_end"], 5)
        self.assertAlmostEqual(bars[1]["imbalance"], -1.0)

    def test_size_weighted_imbalance(self):
        ticks = [
            (0, 100.0, 2),
            (1, 101.0, 2),
        ]

        bars = generate_bars_from_ticks(ticks, threshold=3, ewma_alpha=0.5)

        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0]["tick_count"], 2)
        self.assertAlmostEqual(bars[0]["imbalance"], 1.0)


if __name__ == "__main__":
    unittest.main()
