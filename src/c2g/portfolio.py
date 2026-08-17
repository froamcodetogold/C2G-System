from __future__ import annotations

import pandas as pd

from .backtest import calculate_metrics, equity_curve


def equal_slot_portfolio(
    trades: pd.DataFrame,
    *,
    allocation_per_asset: float = 0.20,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build an unlevered realized-equity proxy with one fixed capital slot per asset."""

    if not 0 < allocation_per_asset <= 1:
        raise ValueError("allocation_per_asset must be within (0, 1]")
    if trades.empty:
        return pd.DataFrame(), {
            **calculate_metrics(pd.DataFrame()).as_dict(),
            "allocation_per_asset": allocation_per_asset,
            "max_concurrent_positions": 0,
            "maximum_nominal_exposure_pct": 0.0,
        }

    proxy = trades.sort_values("exit_execution_time").copy()
    proxy["asset_pnl_pct"] = proxy["pnl_pct"]
    proxy["pnl_pct"] = proxy["asset_pnl_pct"] * allocation_per_asset
    metrics = calculate_metrics(proxy).as_dict()
    maximum_concurrent = max_concurrent_positions(trades)
    summary = {
        **metrics,
        "allocation_per_asset": allocation_per_asset,
        "max_concurrent_positions": maximum_concurrent,
        "maximum_nominal_exposure_pct": maximum_concurrent * allocation_per_asset * 100.0,
    }
    return proxy, summary


def max_concurrent_positions(trades: pd.DataFrame) -> int:
    events: list[tuple[pd.Timestamp, int]] = []
    for row in trades.to_dict("records"):
        events.append((pd.Timestamp(row["entry_time"]), 1))
        events.append((pd.Timestamp(row["exit_execution_time"]), -1))
    # Exits free capital before entries at the exact same timestamp.
    events.sort(key=lambda item: (item[0], item[1]))
    active = 0
    maximum = 0
    for _, delta in events:
        active += delta
        maximum = max(maximum, active)
    return maximum


def monthly_asset_correlation(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    monthly = trades.copy()
    monthly["month"] = (
        pd.to_datetime(monthly["exit_execution_time"], utc=True)
        .dt.tz_localize(None)
        .dt.to_period("M")
    )
    matrix = monthly.pivot_table(
        index="month",
        columns="asset",
        values="pnl_pct",
        aggfunc="sum",
        fill_value=0.0,
    )
    return matrix.corr(min_periods=6)


def portfolio_equity(proxy_trades: pd.DataFrame) -> pd.DataFrame:
    return equity_curve(proxy_trades)
