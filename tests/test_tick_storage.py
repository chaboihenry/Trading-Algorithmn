import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from data.tick_storage import TickStorage, to_epoch_ms
from scripts.init_tick_tables import init_tables


class TickStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "ticks.db"
        init_tables(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_idempotent_save_without_trade_id(self):
        storage = TickStorage(str(self.db_path))
        ticks = [
            {"timestamp": "2024-01-01T09:30:00Z", "price": 100.0, "size": 10},
            {"timestamp": "2024-01-01T09:30:01Z", "price": 100.5, "size": 20},
        ]

        saved_first = storage.save_ticks("SPY", ticks)
        saved_second = storage.save_ticks("SPY", ticks)

        self.assertEqual(saved_first, 2)
        self.assertEqual(saved_second, 0)
        self.assertEqual(storage.get_tick_count("SPY"), 2)

        storage.close()

    def test_idempotent_save_with_trade_id(self):
        storage = TickStorage(str(self.db_path))
        ticks = [
            {"timestamp": "2024-01-01T09:30:00Z", "price": 100.0, "size": 10, "trade_id": "t1"},
            {"timestamp": "2024-01-01T09:30:01Z", "price": 100.5, "size": 20, "trade_id": "t2"},
        ]

        saved_first = storage.save_ticks("SPY", ticks)
        saved_second = storage.save_ticks("SPY", ticks)

        self.assertEqual(saved_first, 2)
        self.assertEqual(saved_second, 0)
        self.assertEqual(storage.get_tick_count("SPY"), 2)

        storage.close()

    def test_timestamps_stored_as_epoch_ms(self):
        storage = TickStorage(str(self.db_path))
        timestamp = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)
        storage.save_ticks(
            "SPY",
            [{"timestamp": timestamp, "price": 100.0, "size": 10}],
        )
        storage.close()

        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp FROM ticks WHERE symbol = ?", ("SPY",))
            value = cursor.fetchone()[0]
        finally:
            conn.close()

        self.assertIsInstance(value, int)
        self.assertEqual(value, to_epoch_ms(timestamp))


if __name__ == "__main__":
    unittest.main()
