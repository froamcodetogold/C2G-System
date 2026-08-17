from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    prenan: bool = False,
) -> pd.Series:
    previous_close = close.shift(1)
    ranges = pd.concat(
        [high - low, high - previous_close, previous_close - low],
        axis=1,
    )
    result = ranges.abs().max(axis=1)
    if prenan and len(result):
        result.iloc[0] = np.nan
    result.name = "true_range"
    return result


def rma(values: pd.Series, length: int) -> pd.Series:
    return values.ewm(alpha=1.0 / length, adjust=False).mean()


def _presma_rma(values: pd.Series, length: int) -> pd.Series:
    prepared = values.copy()
    seed = prepared.iloc[:length].mean()
    prepared.iloc[: length - 1] = np.nan
    prepared.iloc[length - 1] = seed
    return rma(prepared, length)


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    length: int = 14,
    prenan: bool = False,
) -> pd.Series:
    """ATR compatible with pandas-ta 0.4.71b0's default RMA path."""

    result = _presma_rma(true_range(high, low, close, prenan=prenan), length)
    result.name = f"ATRr_{length}"
    return result


def ema(close: pd.Series, *, length: int = 10) -> pd.Series:
    """EMA with the SMA seed used by pandas-ta's default implementation."""

    prepared = close.copy()
    seed = prepared.iloc[:length].mean()
    prepared.iloc[: length - 1] = np.nan
    prepared.iloc[length - 1] = seed
    result = prepared.ewm(span=length, adjust=False).mean()
    result.name = f"EMA_{length}"
    return result


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    length: int = 14,
) -> pd.DataFrame:
    """ADX/DMI compatible with the frozen research implementation."""

    atr_values = atr(high, low, close, length=length, prenan=True)
    scale = 100.0 / atr_values

    up = high - high.shift(1)
    down = low.shift(1) - low
    positive = (((up > down) & (up > 0)) * up).where(lambda item: item != 0, 0.0)
    negative = (((down > up) & (down > 0)) * down).where(lambda item: item != 0, 0.0)

    dmp = scale * rma(positive, length)
    dmn = scale * rma(negative, length)
    denominator = dmp + dmn
    dx = 100.0 * (dmp - dmn).abs() / denominator.replace(0.0, np.nan)
    adx_values = rma(dx, length)
    adxr = 0.5 * (adx_values + adx_values.shift(2))

    return pd.DataFrame(
        {
            f"ADX_{length}": adx_values,
            f"ADXR_{length}_2": adxr,
            f"DMP_{length}": dmp,
            f"DMN_{length}": dmn,
        },
        index=close.index,
    )


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    length: int = 7,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """Supertrend compatible with pandas-ta 0.4.71b0's default path."""

    count = len(close)
    direction = np.ones(count, dtype=float)
    trend = np.zeros(count, dtype=float)
    long_band = np.full(count, np.nan, dtype=float)
    short_band = np.full(count, np.nan, dtype=float)

    midpoint = 0.5 * (high.to_numpy(dtype=float) + low.to_numpy(dtype=float))
    atr_values = atr(high, low, close, length=length).to_numpy(dtype=float)
    lower = midpoint - multiplier * atr_values
    upper = midpoint + multiplier * atr_values
    close_values = close.to_numpy(dtype=float)

    for index in range(1, count):
        if close_values[index] > upper[index - 1]:
            direction[index] = 1.0
        elif close_values[index] < lower[index - 1]:
            direction[index] = -1.0
        else:
            direction[index] = direction[index - 1]
            if direction[index] > 0 and lower[index] < lower[index - 1]:
                lower[index] = lower[index - 1]
            if direction[index] < 0 and upper[index] > upper[index - 1]:
                upper[index] = upper[index - 1]

        if direction[index] > 0:
            trend[index] = lower[index]
            long_band[index] = lower[index]
        else:
            trend[index] = upper[index]
            short_band[index] = upper[index]

    if count:
        trend[0] = np.nan
        direction[:length] = np.nan

    suffix = f"_{length}_{multiplier}"
    return pd.DataFrame(
        {
            f"SUPERT{suffix}": trend,
            f"SUPERTd{suffix}": direction,
            f"SUPERTl{suffix}": long_band,
            f"SUPERTs{suffix}": short_band,
        },
        index=close.index,
    )
