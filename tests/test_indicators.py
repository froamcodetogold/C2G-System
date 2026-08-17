from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from c2g.indicators import adx, atr, ema, supertrend


class IndicatorParityTests(unittest.TestCase):
    @staticmethod
    def fixture() -> pd.DataFrame:
        x = np.arange(40, dtype=float)
        close = 100 + 0.55 * x + 2.3 * np.sin(x / 2.7)
        open_ = close + 0.4 * np.cos(x / 3.1)
        high = np.maximum(open_, close) + 1.1 + 0.1 * np.sin(x)
        low = np.minimum(open_, close) - 0.9 - 0.1 * np.cos(x)
        return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})

    def test_frozen_indicator_reference_values(self) -> None:
        frame = self.fixture()
        trend = supertrend(frame.high, frame.low, frame.close, length=10, multiplier=3.0)
        adx_frame = adx(frame.high, frame.low, frame.close, length=14)

        expected_trend = np.array(
            [
                107.437663838014,
                107.883132602749,
                109.038339274982,
                110.397479780023,
                111.809191985904,
                113.151872941436,
                114.370068351620,
                115.429646038803,
                116.222347203706,
                116.712565682035,
            ]
        )
        expected_adx = np.array(
            [
                84.214551990688,
                83.644435694744,
                83.326504805496,
                83.227075195164,
                83.296483173238,
                83.500377181938,
                83.799818341712,
                84.162336295499,
                84.558282005336,
                84.958969564601,
            ]
        )
        np.testing.assert_allclose(trend["SUPERT_10_3.0"].tail(10), expected_trend, rtol=1e-11)
        np.testing.assert_allclose(adx_frame["ADX_14"].tail(10), expected_adx, rtol=1e-11)
        np.testing.assert_allclose(
            ema(frame.close, length=20).tail(5),
            [
                113.692204916992,
                114.425908118186,
                115.188896690883,
                115.951423072896,
                116.683884617335,
            ],
            rtol=1e-11,
        )
        np.testing.assert_allclose(
            atr(frame.high, frame.low, frame.close, length=14).tail(5),
            [2.382919204290, 2.390568990946, 2.391231755652, 2.399449758775, 2.408274996007],
            rtol=1e-11,
        )


if __name__ == "__main__":
    unittest.main()
