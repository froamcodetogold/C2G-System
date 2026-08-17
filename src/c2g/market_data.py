from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .data import load_ohlcv, merge_ohlcv

BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"
HOUR_MS = 60 * 60 * 1000


def _get_json(url: str, parameters: dict[str, object], *, timeout: int = 30) -> dict:
    request = Request(
        f"{url}?{urlencode(parameters)}",
        headers={"User-Agent": "C2G-System/1.18 research-paper-logger"},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_bybit_linear_1h(
    symbol: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch closed Bybit linear-perpetual one-hour candles in bounded windows."""

    start_utc = pd.Timestamp(start)
    start_utc = (
        start_utc.tz_localize("UTC") if start_utc.tzinfo is None else start_utc.tz_convert("UTC")
    )
    if end is None:
        now = pd.Timestamp(datetime.now(UTC))
        end_utc = now.floor("h") - pd.Timedelta(hours=1)
    else:
        end_utc = pd.Timestamp(end)
        end_utc = (
            end_utc.tz_localize("UTC") if end_utc.tzinfo is None else end_utc.tz_convert("UTC")
        )

    cursor_ms = int(start_utc.timestamp() * 1000)
    end_ms = int(end_utc.timestamp() * 1000)
    rows: list[list[object]] = []

    while cursor_ms <= end_ms:
        window_end = min(cursor_ms + (limit - 1) * HOUR_MS, end_ms)
        payload = _get_json(
            BYBIT_KLINE_URL,
            {
                "category": "linear",
                "symbol": symbol.upper(),
                "interval": "60",
                "start": cursor_ms,
                "end": window_end,
                "limit": limit,
            },
        )
        if payload.get("retCode") != 0:
            raise RuntimeError(f"Bybit API error for {symbol}: {payload}")
        page = payload.get("result", {}).get("list", [])
        if not page:
            cursor_ms = window_end + HOUR_MS
            continue
        page = sorted(page, key=lambda item: int(item[0]))
        rows.extend(page)
        last_ms = int(page[-1][0])
        cursor_ms = max(last_ms + HOUR_MS, window_end + HOUR_MS)
        sleep(0.05)

    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    frame = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"].astype("int64"), unit="ms", utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.drop(columns="turnover").drop_duplicates("timestamp", keep="last")
    frame = frame.set_index("timestamp").sort_index()
    return frame.loc[:end_utc]


def _atomic_write_csv(frame: pd.DataFrame, destination: Path) -> None:
    export = frame.reset_index()
    export["timestamp"] = pd.to_datetime(export["timestamp"], utc=True).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", suffix=".csv", delete=False, dir=destination.parent
    ) as temporary:
        temporary_path = Path(temporary.name)
        export.to_csv(temporary, index=False)
    os.replace(temporary_path, destination)


def update_bybit_file(
    path: str | Path,
    *,
    symbol: str,
    overlap_hours: int = 48,
) -> dict[str, object]:
    destination = Path(path)
    existing, before = load_ohlcv(destination)
    fetch_start = existing.index.max() - pd.Timedelta(hours=overlap_hours)
    incoming = fetch_bybit_linear_1h(symbol, start=fetch_start)
    merged = merge_ohlcv(existing, incoming)
    _atomic_write_csv(merged, destination)
    _, after = load_ohlcv(destination)
    return {
        "symbol": symbol,
        "path": str(destination),
        "before_rows": before.rows,
        "after_rows": after.rows,
        "new_rows": after.rows - before.rows,
        "last_candle_utc": after.end,
        "gaps": after.non_hourly_gaps,
    }
