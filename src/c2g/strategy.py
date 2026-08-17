from __future__ import annotations

import pandas as pd

from .config import FrozenV1Config
from .indicators import adx, ema, supertrend


def prepare_signals(
    frame: pd.DataFrame,
    config: FrozenV1Config | None = None,
) -> pd.DataFrame:
    """Compute the frozen V1 signal using only information known at candle close."""

    settings = config or FrozenV1Config()
    prepared = frame.copy()

    trend = supertrend(
        prepared["high"],
        prepared["low"],
        prepared["close"],
        length=settings.supertrend_length,
        multiplier=settings.supertrend_multiplier,
    )
    direction_column = next(column for column in trend if column.startswith("SUPERTd_"))
    line_column = next(column for column in trend if column.startswith("SUPERT_"))
    prepared["trend_direction"] = trend[direction_column]
    prepared["trend_line"] = trend[line_column]
    prepared["supertrend_flip"] = prepared["trend_direction"].diff()
    prepared["st_buy_flip"] = prepared["supertrend_flip"].eq(2.0)

    adx_frame = adx(
        prepared["high"],
        prepared["low"],
        prepared["close"],
        length=settings.adx_length,
    )
    prepared["adx"] = adx_frame[f"ADX_{settings.adx_length}"]
    prepared["adx_rising"] = prepared["adx"].gt(prepared["adx"].shift(1))

    prepared["ema200"] = ema(prepared["close"], length=settings.ema_length)
    prepared["ema200_lag"] = prepared["ema200"].shift(settings.ema_slope_lookback)
    prepared["bull_regime"] = prepared["close"].gt(prepared["ema200"]) & prepared["ema200"].gt(
        prepared["ema200_lag"]
    )

    prepared["buy_signal"] = (
        prepared["st_buy_flip"]
        & prepared["adx"].gt(settings.adx_min)
        & prepared["adx_rising"]
        & prepared["bull_regime"]
    )
    return prepared


def signal_snapshot(prepared: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "open",
        "high",
        "low",
        "close",
        "trend_direction",
        "trend_line",
        "adx",
        "ema200",
        "bull_regime",
        "buy_signal",
    ]
    available = [column for column in columns if column in prepared.columns]
    return prepared.loc[prepared["buy_signal"].fillna(False), available].copy()
