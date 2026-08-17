from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True, slots=True)
class DataQuality:
    path: str
    rows: int
    start: str
    end: str
    duplicate_rows_removed: int
    non_hourly_gaps: int
    estimated_missing_candles: int
    invalid_rows_removed: int
    source_sha256: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _estimated_missing(index: pd.DatetimeIndex, timeframe: pd.Timedelta) -> tuple[int, int]:
    if len(index) < 2:
        return 0, 0
    deltas = index.to_series().diff().dropna()
    gaps = deltas[deltas != timeframe]
    missing = 0
    for delta in gaps:
        if delta > timeframe:
            missing += max(round(delta / timeframe) - 1, 0)
    return len(gaps), missing


def load_ohlcv(
    path: str | Path,
    *,
    timeframe: str = "1h",
    reject_invalid_prices: bool = True,
) -> tuple[pd.DataFrame, DataQuality]:
    """Load an OHLCV CSV, normalize timestamps to UTC and audit its quality."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Market data file not found: {source}")

    frame = pd.read_csv(source)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if "timestamp" not in frame.columns:
        raise ValueError(f"{source}: required column 'timestamp' is missing")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"{source}: required columns missing: {missing_columns}")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    for column in (*REQUIRED_COLUMNS, "volume"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    before_clean = len(frame)
    frame = frame.dropna(subset=["timestamp", *REQUIRED_COLUMNS]).copy()

    if reject_invalid_prices:
        valid = (
            (frame[list(REQUIRED_COLUMNS)] > 0).all(axis=1)
            & (frame["high"] >= frame[["open", "close", "low"]].max(axis=1))
            & (frame["low"] <= frame[["open", "close", "high"]].min(axis=1))
        )
        frame = frame.loc[valid].copy()

    invalid_rows_removed = before_clean - len(frame)
    frame = frame.sort_values("timestamp")
    duplicate_rows_removed = int(frame["timestamp"].duplicated(keep="last").sum())
    frame = frame.drop_duplicates(subset="timestamp", keep="last")
    frame = frame.set_index("timestamp")
    frame.index = pd.DatetimeIndex(frame.index, name="timestamp")

    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"{source}: timestamps are not monotonic after normalization")

    expected_delta = pd.Timedelta(timeframe)
    non_hourly_gaps, estimated_missing = _estimated_missing(frame.index, expected_delta)

    quality = DataQuality(
        path=str(source),
        rows=len(frame),
        start=frame.index.min().isoformat() if len(frame) else "",
        end=frame.index.max().isoformat() if len(frame) else "",
        duplicate_rows_removed=duplicate_rows_removed,
        non_hourly_gaps=non_hourly_gaps,
        estimated_missing_candles=estimated_missing,
        invalid_rows_removed=invalid_rows_removed,
        source_sha256=file_sha256(source),
    )
    return frame, quality


def merge_ohlcv(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Merge data refreshes deterministically, keeping the newest duplicate candle."""

    combined = pd.concat([existing, incoming]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


def dataframe_sha256(frame: pd.DataFrame) -> str:
    """Stable hash used in forward manifests and audit logs."""

    normalized = frame.sort_index().copy()
    payload = pd.util.hash_pandas_object(normalized, index=True).to_numpy(dtype=np.uint64)
    return sha256(payload.tobytes()).hexdigest()
