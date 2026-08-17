from __future__ import annotations

import unittest
from pathlib import Path

from c2g.suite import run_canonical_suite


class HistoricalRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.results = run_canonical_suite(cls.root, bootstrap_samples=250)

    def test_cross_asset_common_period_matches_frozen_reference(self) -> None:
        summary = self.results["summary"]
        selected = summary[
            (summary["scope"] == "ASSET_COMMON_PERIOD")
            & (summary["scenario"] == "ASSET_COMMON_COST")
        ].set_index("asset")
        expected = {
            "BTC": (20, 1.4123501730),
            "ETH": (14, 1.9527100000),
            "SOL": (13, 1.5532000000),
            "XRP": (12, 0.9633000000),
            "BNB": (12, 0.3823066064),
        }
        for asset, (trades, factor) in expected.items():
            self.assertEqual(int(selected.loc[asset, "trades"]), trades)
            self.assertAlmostEqual(float(selected.loc[asset, "profit_factor"]), factor, places=3)

    def test_binance_and_bybit_same_period_reference(self) -> None:
        summary = self.results["summary"]
        selected = summary[
            (summary["scope"] == "BTC_COMMON_PERIOD") & (summary["scenario"] == "BTC_COMMON_COST")
        ].set_index("market")
        self.assertEqual(int(selected.loc["BINANCE_SPOT", "trades"]), 25)
        self.assertAlmostEqual(
            float(selected.loc["BINANCE_SPOT", "profit_factor"]), 1.707, places=3
        )
        self.assertEqual(int(selected.loc["BYBIT_PERPETUAL", "trades"]), 24)
        self.assertAlmostEqual(
            float(selected.loc["BYBIT_PERPETUAL", "profit_factor"]), 1.459, places=3
        )


if __name__ == "__main__":
    unittest.main()
