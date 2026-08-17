from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from c2g.forward import LedgerIntegrityError, build_forward_snapshot, merge_forward_ledger


class ForwardLedgerTests(unittest.TestCase):
    @staticmethod
    def prepared(rows: int) -> pd.DataFrame:
        index = pd.date_range("2026-08-15T20:00:00Z", periods=rows, freq="1h")
        price = 100.0 + np.arange(rows)
        frame = pd.DataFrame(
            {
                "open": price,
                "high": price + 2,
                "low": price - 2,
                "close": price + 1,
                "adx": 30.0,
                "buy_signal": False,
            },
            index=index,
        )
        frame.iloc[0, frame.columns.get_loc("buy_signal")] = True
        return frame

    def test_open_event_advances_to_closed_and_then_becomes_immutable(self) -> None:
        first_seen = pd.Timestamp("2026-08-16T12:00:00Z")
        open_snapshot = build_forward_snapshot(
            self.prepared(10),
            asset="BTC",
            market="BYBIT_PERPETUAL",
            freeze_time="2026-08-15T20:00:00Z",
            observed_at=first_seen,
        )
        self.assertEqual(open_snapshot.iloc[0]["status"], "OPEN")

        closed_snapshot = build_forward_snapshot(
            self.prepared(30),
            asset="BTC",
            market="BYBIT_PERPETUAL",
            freeze_time="2026-08-15T20:00:00Z",
            observed_at=pd.Timestamp("2026-08-17T12:00:00Z"),
        )
        ledger = merge_forward_ledger(open_snapshot, closed_snapshot)
        self.assertEqual(ledger.iloc[0]["status"], "CLOSED")
        self.assertEqual(pd.Timestamp(ledger.iloc[0]["first_seen_at"]), first_seen)

        changed = closed_snapshot.copy()
        changed.loc[0, "exit_price"] += 1.0
        with self.assertRaises(LedgerIntegrityError):
            merge_forward_ledger(ledger, changed)


if __name__ == "__main__":
    unittest.main()
