from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from . import __version__
from .backtest import (
    bootstrap_metrics,
    calculate_metrics,
    run_time_exit_backtest,
    top_winner_dependency,
    yearly_metrics,
)
from .config import BTC_MARKETS, BYBIT_ASSETS, FrozenV1Config
from .data import load_ohlcv
from .portfolio import equal_slot_portfolio, monthly_asset_correlation, portfolio_equity
from .strategy import prepare_signals


def _classification(profit_factor: float, expectancy: float) -> str:
    if profit_factor >= 1.20 and expectancy > 0:
        return "HISTORICALLY_POSITIVE"
    if profit_factor > 1.0 and expectancy > 0:
        return "MARGINAL"
    return "FAILED"


def _scenario_result(
    prepared: pd.DataFrame,
    *,
    asset: str,
    market: str,
    scope: str,
    scenario: str,
    include_costs: bool,
    config: FrozenV1Config,
) -> tuple[dict[str, object], pd.DataFrame]:
    trades = run_time_exit_backtest(
        prepared,
        asset=asset,
        market=market,
        config=config,
        include_costs=include_costs,
    )
    metrics = calculate_metrics(trades)
    row = {
        "scope": scope,
        "scenario": scenario,
        "asset": asset,
        "market": market,
        "period_start": prepared.index.min(),
        "period_end": prepared.index.max(),
        "candles": len(prepared),
        **metrics.as_dict(),
        "classification": _classification(metrics.profit_factor, metrics.expectancy_pct),
    }
    trades = trades.assign(scope=scope, scenario=scenario)
    return row, trades


def _append_scenario(
    summaries: list[dict[str, object]],
    trade_frames: list[pd.DataFrame],
    prepared: pd.DataFrame,
    *,
    asset: str,
    market: str,
    scope: str,
    scenario: str,
    include_costs: bool,
    config: FrozenV1Config,
) -> None:
    row, trades = _scenario_result(
        prepared,
        asset=asset,
        market=market,
        scope=scope,
        scenario=scenario,
        include_costs=include_costs,
        config=config,
    )
    summaries.append(row)
    trade_frames.append(trades)


def run_canonical_suite(
    project_root: str | Path,
    *,
    config: FrozenV1Config | None = None,
    bootstrap_samples: int = 20_000,
) -> dict[str, object]:
    """Run the final frozen strategy across venues, assets and common periods."""

    root = Path(project_root)
    settings = config or FrozenV1Config()
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    prepared: dict[tuple[str, str], pd.DataFrame] = {}
    quality_rows: list[dict[str, object]] = []

    sources: dict[tuple[str, str], str] = {
        ("BTC", market): filename for market, filename in BTC_MARKETS.items()
    }
    sources.update(
        {
            (asset, "BYBIT_PERPETUAL"): filename
            for asset, filename in BYBIT_ASSETS.items()
            if asset != "BTC"
        }
    )

    for key, filename in sources.items():
        frame, quality = load_ohlcv(root / filename)
        frames[key] = frame
        prepared[key] = prepare_signals(frame, settings)
        quality_rows.append(
            {"asset": key[0], "market": key[1], **quality.as_dict(), "path": filename}
        )

    summary_rows: list[dict[str, object]] = []
    trade_frames: list[pd.DataFrame] = []

    for (asset, market), signal_frame in prepared.items():
        for suffix, include_costs in (("GROSS", False), ("COST", True)):
            _append_scenario(
                summary_rows,
                trade_frames,
                signal_frame,
                asset=asset,
                market=market,
                scope="FULL_AVAILABLE",
                scenario=f"FULL_{suffix}",
                include_costs=include_costs,
                config=settings,
            )

    btc_keys = [("BTC", market) for market in BTC_MARKETS]
    btc_common_start = max(frames[key].index.min() for key in btc_keys)
    btc_common_end = min(frames[key].index.max() for key in btc_keys)
    for key in btc_keys:
        signal_frame = prepared[key].loc[btc_common_start:btc_common_end].copy()
        for suffix, include_costs in (("GROSS", False), ("COST", True)):
            _append_scenario(
                summary_rows,
                trade_frames,
                signal_frame,
                asset="BTC",
                market=key[1],
                scope="BTC_COMMON_PERIOD",
                scenario=f"BTC_COMMON_{suffix}",
                include_costs=include_costs,
                config=settings,
            )

    asset_keys = [(asset, "BYBIT_PERPETUAL") for asset in BYBIT_ASSETS]
    asset_common_start = max(frames[key].index.min() for key in asset_keys)
    asset_common_end = min(frames[key].index.max() for key in asset_keys)
    for key in asset_keys:
        signal_frame = prepared[key].loc[asset_common_start:asset_common_end].copy()
        for suffix, include_costs in (("GROSS", False), ("COST", True)):
            _append_scenario(
                summary_rows,
                trade_frames,
                signal_frame,
                asset=key[0],
                market="BYBIT_PERPETUAL",
                scope="ASSET_COMMON_PERIOD",
                scenario=f"ASSET_COMMON_{suffix}",
                include_costs=include_costs,
                config=settings,
            )

    summary = pd.DataFrame(summary_rows)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    quality = pd.DataFrame(quality_rows)

    yearly_frames: list[pd.DataFrame] = []
    outlier_frames: list[pd.DataFrame] = []
    bootstrap_rows: list[dict[str, object]] = []
    cost_summary = summary[summary["scenario"].str.endswith("_COST")]
    for row in cost_summary.to_dict("records"):
        subset = trades[
            (trades["scope"] == row["scope"])
            & (trades["scenario"] == row["scenario"])
            & (trades["asset"] == row["asset"])
            & (trades["market"] == row["market"])
        ].copy()
        identity = {
            "scope": row["scope"],
            "scenario": row["scenario"],
            "asset": row["asset"],
            "market": row["market"],
        }
        annual = yearly_metrics(subset)
        if not annual.empty:
            yearly_frames.append(annual.assign(**identity))
        dependency = top_winner_dependency(subset)
        if not dependency.empty:
            outlier_frames.append(dependency.assign(**identity))
        bootstrap_rows.append(
            {**identity, **bootstrap_metrics(subset, samples=bootstrap_samples, seed=42)}
        )

    yearly = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    outliers = pd.concat(outlier_frames, ignore_index=True) if outlier_frames else pd.DataFrame()
    bootstrap = pd.DataFrame(bootstrap_rows)

    portfolio_source = trades[
        (trades["scope"] == "ASSET_COMMON_PERIOD") & (trades["scenario"] == "ASSET_COMMON_COST")
    ].copy()
    portfolio_trades, portfolio_row = equal_slot_portfolio(
        portfolio_source,
        allocation_per_asset=0.20,
    )
    portfolio_summary = pd.DataFrame([portfolio_row])
    portfolio_curve = portfolio_equity(portfolio_trades)
    asset_correlation = monthly_asset_correlation(portfolio_source)

    manifest = {
        "engine_version": __version__,
        "run_utc": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": asdict(settings),
        "config_hash": settings.fingerprint,
        "bootstrap_samples": bootstrap_samples,
        "btc_common_period": {
            "start": btc_common_start.isoformat(),
            "end": btc_common_end.isoformat(),
        },
        "asset_common_period": {
            "start": asset_common_start.isoformat(),
            "end": asset_common_end.isoformat(),
        },
        "funding_model": "not_included; configure funding_pct_per_trade when validated data exists",
        "execution_model": "next candle open; close of the 24th held 1H candle",
        "portfolio_proxy": {
            "allocation_per_asset": 0.20,
            "leverage": 1.0,
            "description": "fixed equal capital slots; realized trade-level equity, not mark-to-market",
        },
        "limitations": [
            "OHLC execution is deterministic and does not model order-book depth or partial fills.",
            "Cross-venue BTC feeds are correlated observations, not independent markets.",
            "Forward and historical samples remain small.",
        ],
        "data": quality.to_dict("records"),
    }
    return {
        "summary": summary,
        "trades": trades,
        "yearly": yearly,
        "outliers": outliers,
        "bootstrap": bootstrap,
        "data_quality": quality,
        "portfolio_summary": portfolio_summary,
        "portfolio_trades": portfolio_trades,
        "portfolio_equity": portfolio_curve,
        "asset_correlation": asset_correlation,
        "manifest": manifest,
        "prepared": prepared,
    }


def save_suite_results(results: dict[str, object], output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "summary",
        "trades",
        "yearly",
        "outliers",
        "bootstrap",
        "data_quality",
        "portfolio_summary",
        "portfolio_trades",
    ):
        frame = results[name]
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(destination / f"{name}.csv", index=False)
    portfolio_curve = results.get("portfolio_equity")
    if isinstance(portfolio_curve, pd.DataFrame):
        portfolio_curve.to_csv(destination / "portfolio_equity.csv", index=True)
    correlation = results.get("asset_correlation")
    if isinstance(correlation, pd.DataFrame):
        correlation.to_csv(destination / "asset_correlation.csv", index=True)
    (destination / "run_manifest.json").write_text(
        json.dumps(results["manifest"], indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
