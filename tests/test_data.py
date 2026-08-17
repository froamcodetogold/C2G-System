from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from c2g.data import load_ohlcv


class DataQualityTests(unittest.TestCase):
    def test_utc_duplicates_and_gaps_are_audited(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            pd.DataFrame(
                {
                    "timestamp": [
                        "2025-01-01 00:00:00",
                        "2025-01-01 01:00:00",
                        "2025-01-01 01:00:00",
                        "2025-01-01 03:00:00",
                    ],
                    "open": [100, 101, 101, 103],
                    "high": [102, 103, 103, 105],
                    "low": [99, 100, 100, 102],
                    "close": [101, 102, 102, 104],
                }
            ).to_csv(path, index=False)
            frame, quality = load_ohlcv(path)

        self.assertEqual(len(frame), 3)
        self.assertEqual(quality.duplicate_rows_removed, 1)
        self.assertEqual(quality.non_hourly_gaps, 1)
        self.assertEqual(quality.estimated_missing_candles, 1)
        self.assertEqual(str(frame.index.tz), "UTC")


if __name__ == "__main__":
    unittest.main()
