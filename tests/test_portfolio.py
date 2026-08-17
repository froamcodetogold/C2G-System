from __future__ import annotations

import unittest

import pandas as pd

from c2g.portfolio import equal_slot_portfolio


class PortfolioTests(unittest.TestCase):
    def test_equal_slots_scale_returns_and_measure_concurrency(self) -> None:
        trades = pd.DataFrame(
            {
                "asset": ["BTC", "ETH"],
                "entry_time": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T12:00:00Z"]),
                "exit_execution_time": pd.to_datetime(
                    ["2025-01-02T00:00:00Z", "2025-01-02T12:00:00Z"]
                ),
                "pnl_pct": [10.0, -5.0],
                "clock_hours": [24.0, 24.0],
            }
        )
        proxy, summary = equal_slot_portfolio(trades, allocation_per_asset=0.20)
        self.assertEqual(proxy["pnl_pct"].tolist(), [2.0, -1.0])
        self.assertEqual(summary["max_concurrent_positions"], 2)
        self.assertEqual(summary["maximum_nominal_exposure_pct"], 40.0)


if __name__ == "__main__":
    unittest.main()
