from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import calculate_metrics
from .config import FrozenV1Config
from .data import dataframe_sha256

STATUS_ORDER = {"WAITING_ENTRY": 0, "OPEN": 1, "CLOSED": 2}
FORWARD_COLUMNS = [
    "event_id",
    "asset",
    "market",
    "signal_time",
    "status",
    "entry_time",
    "entry_price",
    "planned_exit_candles",
    "exit_bar_time",
    "exit_execution_time",
    "exit_price",
    "clock_hours",
    "gross_return_pct",
    "fees_pct",
    "slippage_pct",
    "funding_pct",
    "pnl_pct",
    "mfe_pct",
    "mae_pct",
    "signal_adx",
    "config_hash",
    "source_hash",
    "source_last_candle",
    "first_seen_at",
    "last_observed_at",
]
IMMUTABLE_CLOSED_FIELDS = (
    "asset",
    "market",
    "signal_time",
    "entry_time",
    "entry_price",
    "exit_bar_time",
    "exit_execution_time",
    "exit_price",
    "gross_return_pct",
    "pnl_pct",
    "config_hash",
)


class LedgerIntegrityError(RuntimeError):
    pass


def _event_id(
    asset: str,
    market: str,
    signal_time: pd.Timestamp,
    config_hash: str,
) -> str:
    payload = f"{asset}|{market}|{signal_time.isoformat()}|{config_hash}"
    return sha256(payload.encode("utf-8")).hexdigest()[:20]


def build_forward_snapshot(
    prepared: pd.DataFrame,
    *,
    asset: str,
    market: str,
    freeze_time: pd.Timestamp | str,
    config: FrozenV1Config | None = None,
    observed_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Reconstruct current event state while enforcing the historical no-overlap rule."""

    settings = config or FrozenV1Config()
    freeze = pd.Timestamp(freeze_time)
    freeze = freeze.tz_localize("UTC") if freeze.tzinfo is None else freeze.tz_convert("UTC")
    observed = observed_at or pd.Timestamp(datetime.now(UTC))
    observed = (
        observed.tz_localize("UTC") if observed.tzinfo is None else observed.tz_convert("UTC")
    )
    source_hash = dataframe_sha256(prepared[["open", "high", "low", "close"]])
    source_last_candle = prepared.index.max() if len(prepared) else pd.NaT

    rows: list[dict[str, object]] = []
    blocked_through = -1
    signal_positions = np.flatnonzero(prepared["buy_signal"].fillna(False).to_numpy())

    for signal_index in signal_positions:
        signal_time = pd.Timestamp(prepared.index[signal_index])
        if signal_time < freeze or signal_index <= blocked_through:
            continue

        entry_index = signal_index + 1
        base = {
            "event_id": _event_id(asset, market, signal_time, settings.fingerprint),
            "asset": asset,
            "market": market,
            "signal_time": signal_time,
            "entry_time": pd.NaT,
            "entry_price": np.nan,
            "planned_exit_candles": settings.time_exit_bars,
            "exit_bar_time": pd.NaT,
            "exit_execution_time": pd.NaT,
            "exit_price": np.nan,
            "clock_hours": np.nan,
            "gross_return_pct": np.nan,
            "fees_pct": 2.0 * settings.fee_pct_per_side,
            "slippage_pct": 2.0 * settings.slippage_pct_per_side,
            "funding_pct": settings.funding_pct_per_trade,
            "pnl_pct": np.nan,
            "mfe_pct": np.nan,
            "mae_pct": np.nan,
            "signal_adx": float(prepared["adx"].iloc[signal_index]),
            "config_hash": settings.fingerprint,
            "source_hash": source_hash,
            "source_last_candle": source_last_candle,
            "first_seen_at": observed,
            "last_observed_at": observed,
        }

        if entry_index >= len(prepared):
            rows.append({**base, "status": "WAITING_ENTRY"})
            break

        entry_time = pd.Timestamp(prepared.index[entry_index])
        entry_price = float(prepared["open"].iloc[entry_index])
        exit_index = entry_index + settings.time_exit_bars - 1
        blocked_through = exit_index
        available_end = min(exit_index, len(prepared) - 1)
        window = prepared.iloc[entry_index : available_end + 1]
        mfe = (float(window["high"].max()) - entry_price) / entry_price * 100.0
        mae = (entry_price - float(window["low"].min())) / entry_price * 100.0
        base.update(
            {
                "entry_time": entry_time,
                "entry_price": entry_price,
                "mfe_pct": mfe,
                "mae_pct": mae,
            }
        )

        if exit_index >= len(prepared):
            rows.append({**base, "status": "OPEN"})
            break

        exit_bar_time = pd.Timestamp(prepared.index[exit_index])
        exit_execution_time = exit_bar_time + pd.Timedelta(hours=settings.timeframe_hours)
        exit_price = float(prepared["close"].iloc[exit_index])
        gross = (exit_price - entry_price) / entry_price * 100.0
        pnl = gross - settings.round_trip_cost_pct
        rows.append(
            {
                **base,
                "status": "CLOSED",
                "exit_bar_time": exit_bar_time,
                "exit_execution_time": exit_execution_time,
                "exit_price": exit_price,
                "clock_hours": (exit_execution_time - entry_time).total_seconds() / 3600.0,
                "gross_return_pct": gross,
                "pnl_pct": pnl,
            }
        )

    snapshot = pd.DataFrame(rows, columns=FORWARD_COLUMNS)
    for column in (
        "signal_time",
        "entry_time",
        "exit_bar_time",
        "exit_execution_time",
        "source_last_candle",
        "first_seen_at",
        "last_observed_at",
    ):
        if column in snapshot:
            snapshot[column] = pd.to_datetime(snapshot[column], utc=True, errors="coerce")
    return snapshot


def _same_value(left: object, right: object) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    try:
        return bool(np.isclose(float(left), float(right), equal_nan=True, rtol=1e-10, atol=1e-10))
    except (TypeError, ValueError):
        return str(left) == str(right)


def merge_forward_ledger(existing: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    """Append new events and permit only forward status transitions."""

    if snapshot.empty:
        return existing.copy()
    if existing.empty:
        return snapshot.sort_values(["signal_time", "asset", "market"]).reset_index(drop=True)

    current = existing.copy().set_index("event_id", drop=False)
    for incoming in snapshot.to_dict("records"):
        event_id = incoming["event_id"]
        if event_id not in current.index:
            current.loc[event_id] = incoming
            continue

        previous = current.loc[event_id].to_dict()
        old_status = previous["status"]
        new_status = incoming["status"]
        if STATUS_ORDER[new_status] < STATUS_ORDER[old_status]:
            raise LedgerIntegrityError(
                f"Forward event {event_id} regressed from {old_status} to {new_status}"
            )

        if old_status == "CLOSED":
            for field in IMMUTABLE_CLOSED_FIELDS:
                if not _same_value(previous.get(field), incoming.get(field)):
                    raise LedgerIntegrityError(
                        f"Closed event {event_id} changed field {field}: "
                        f"{previous.get(field)!r} -> {incoming.get(field)!r}"
                    )
            continue

        incoming["first_seen_at"] = previous.get("first_seen_at", incoming["first_seen_at"])
        current.loc[event_id] = incoming

    merged = current.reset_index(drop=True)
    return merged.sort_values(["signal_time", "asset", "market"]).reset_index(drop=True)


def forward_summary(
    ledger: pd.DataFrame,
    *,
    expected_pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    summary_columns = [
        "asset",
        "market",
        "signals_after_freeze",
        "waiting_entry",
        "open_trades",
        "closed_trades",
        *calculate_metrics(pd.DataFrame()).as_dict().keys(),
    ]
    if ledger.empty:
        rows = []
        for asset, market in expected_pairs or []:
            rows.append(
                {
                    "asset": asset,
                    "market": market,
                    "signals_after_freeze": 0,
                    "waiting_entry": 0,
                    "open_trades": 0,
                    "closed_trades": 0,
                    **calculate_metrics(pd.DataFrame()).as_dict(),
                }
            )
        return pd.DataFrame(rows, columns=summary_columns)
    rows: list[dict[str, object]] = []
    for (asset, market), group in ledger.groupby(["asset", "market"], dropna=False):
        closed = group[group["status"] == "CLOSED"].copy()
        metrics = (
            calculate_metrics(closed) if not closed.empty else calculate_metrics(pd.DataFrame())
        )
        rows.append(
            {
                "asset": asset,
                "market": market,
                "signals_after_freeze": len(group),
                "waiting_entry": int((group["status"] == "WAITING_ENTRY").sum()),
                "open_trades": int((group["status"] == "OPEN").sum()),
                "closed_trades": len(closed),
                **metrics.as_dict(),
            }
        )
    return pd.DataFrame(rows, columns=summary_columns)


def load_ledger(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists() or source.stat().st_size <= 1:
        return pd.DataFrame(columns=FORWARD_COLUMNS)
    frame = pd.read_csv(source)
    for column in (
        "signal_time",
        "entry_time",
        "exit_bar_time",
        "exit_execution_time",
        "source_last_candle",
        "first_seen_at",
        "last_observed_at",
    ):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return frame


def save_forward_state(
    ledger: pd.DataFrame,
    *,
    ledger_path: str | Path,
    summary_path: str | Path,
    manifest_path: str | Path,
    manifest: dict[str, object],
    expected_pairs: list[tuple[str, str]] | None = None,
) -> None:
    ledger.to_csv(ledger_path, index=False)
    forward_summary(ledger, expected_pairs=expected_pairs).to_csv(summary_path, index=False)
    Path(manifest_path).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
