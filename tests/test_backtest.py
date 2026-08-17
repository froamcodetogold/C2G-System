from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from c2g.backtest import calculate_metrics, equity_curve, run_time_exit_backtest
from c2g.config import FrozenV1Config


class BacktestTests(unittest.TestCase):
    @staticmethod
    def prepared(rows: int = 40) -> pd.DataFrame:
        index = pd.date_range("2025-01-01", periods=rows, freq="1h", tz="UTC")
        open_ = 100.0 + np.arange(rows)
        frame = pd.DataFrame(
            {
                "open": open_,
                "high": open_ + 2.0,
                "low": open_ - 2.0,
                "close": open_ + 1.0,
                "adx": 30.0,
                "buy_signal": False,
            },
            index=index,
        )
        return frame

    def test_next_open_and_close_of_24th_held_candle(self) -> None:
        frame = self.prepared()
        frame.iloc[2, frame.columns.get_loc("buy_signal")] = True
        frame.iloc[10, frame.columns.get_loc("buy_signal")] = True
        trades = run_time_exit_backtest(
            frame,
            asset="BTC",
            market="TEST",
            config=FrozenV1Config(),
            include_costs=False,
        )
        self.assertEqual(len(trades), 1, "signals inside an open position must be ignored")
        trade = trades.iloc[0]
        self.assertEqual(trade["entry_time"], frame.index[3])
        self.assertEqual(trade["entry_price"], frame["open"].iloc[3])
        self.assertEqual(trade["exit_bar_time"], frame.index[26])
        self.assertEqual(trade["exit_execution_time"], frame.index[27])
        self.assertEqual(trade["bars_held"], 24)
        self.assertEqual(trade["clock_hours"], 24.0)

    def test_round_trip_cost_is_exactly_point_fifteen_percent(self) -> None:
        frame = self.prepared()
        frame.iloc[2, frame.columns.get_loc("buy_signal")] = True
        gross = run_time_exit_backtest(frame, asset="BTC", market="TEST", include_costs=False).iloc[
            0
        ]
        cost = run_time_exit_backtest(frame, asset="BTC", market="TEST", include_costs=True).iloc[0]
        self.assertAlmostEqual(gross["pnl_pct"] - cost["pnl_pct"], 0.15, places=12)

    def test_drawdown_includes_initial_equity(self) -> None:
        trades = pd.DataFrame(
            {
                "entry_time": pd.to_datetime(["2025-01-01T00:00:00Z"]),
                "exit_execution_time": pd.to_datetime(["2025-01-02T00:00:00Z"]),
                "pnl_pct": [-10.0],
                "clock_hours": [24.0],
            }
        )
        curve = equity_curve(trades)
        self.assertEqual(curve["equity"].tolist(), [100.0, 90.0])
        self.assertAlmostEqual(calculate_metrics(trades).max_drawdown_pct, -10.0)


if __name__ == "__main__":
    unittest.main()
