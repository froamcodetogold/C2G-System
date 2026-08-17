from __future__ import annotations

from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import equity_curve

BACKGROUND = "#0b1120"
PANEL = "#111827"
FOREGROUND = "#e5e7eb"
MUTED = "#94a3b8"
GRID = "#334155"
GREEN = "#22c55e"
RED = "#ef4444"
AMBER = "#f59e0b"
SERIES = ["#38bdf8", "#a78bfa", "#22c55e", "#f59e0b", "#f472b6", "#fb7185"]


def _style_axis(axis: plt.Axes, *, grid: bool = True) -> None:
    axis.set_facecolor(PANEL)
    axis.tick_params(colors=MUTED, labelsize=8)
    axis.xaxis.label.set_color(MUTED)
    axis.yaxis.label.set_color(MUTED)
    axis.title.set_color(FOREGROUND)
    for spine in axis.spines.values():
        spine.set_color(GRID)
    if grid:
        axis.grid(True, color=GRID, alpha=0.35, linewidth=0.7)


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def _asset_cost_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[
        (summary["scope"] == "ASSET_COMMON_PERIOD") & (summary["scenario"] == "ASSET_COMMON_COST")
    ].sort_values("asset")


def _asset_cost_trades(trades: pd.DataFrame) -> pd.DataFrame:
    return trades[
        (trades["scope"] == "ASSET_COMMON_PERIOD") & (trades["scenario"] == "ASSET_COMMON_COST")
    ].copy()


def create_dashboard(results: dict[str, object], output_path: str | Path) -> Path:
    summary = results["summary"]
    trades = results["trades"]
    if not isinstance(summary, pd.DataFrame) or not isinstance(trades, pd.DataFrame):
        raise TypeError("results must include pandas DataFrames")

    asset_summary = _asset_cost_summary(summary)
    asset_trades = _asset_cost_trades(trades)
    path = Path(output_path)

    figure, axes = plt.subplots(2, 2, figsize=(16, 9), facecolor=BACKGROUND)
    figure.suptitle(
        "C2G System Pro · Frozen BUY 24H · Common-period cost stress",
        color=FOREGROUND,
        fontsize=18,
        fontweight="bold",
        x=0.06,
        ha="left",
    )
    figure.text(
        0.06,
        0.925,
        "Same rules on every asset · fees + slippage = 0.15% round trip · funding excluded",
        color=MUTED,
        fontsize=9,
    )

    equity_axis, drawdown_axis, factor_axis, risk_axis = axes.flat
    for asset_index, asset in enumerate(asset_summary["asset"]):
        subset = asset_trades[asset_trades["asset"] == asset].copy()
        curve = equity_curve(subset)
        color = SERIES[asset_index % len(SERIES)]
        equity_axis.plot(curve.index, curve["equity"], label=asset, color=color, linewidth=2)
        drawdown_axis.plot(
            curve.index, curve["drawdown_pct"], label=asset, color=color, linewidth=1.7
        )
    portfolio_curve = results.get("portfolio_equity")
    if isinstance(portfolio_curve, pd.DataFrame) and not portfolio_curve.empty:
        equity_axis.plot(
            portfolio_curve.index,
            portfolio_curve["equity"],
            label="Portfolio 20% slots",
            color=FOREGROUND,
            linewidth=2.4,
            linestyle="--",
        )

    _style_axis(equity_axis)
    equity_axis.set_title("Compounded equity", loc="left", fontsize=11, color=FOREGROUND)
    equity_axis.set_ylabel("Equity (start = 100)")
    equity_axis.legend(frameon=False, labelcolor=FOREGROUND, fontsize=8, ncol=3)
    equity_axis.xaxis.set_major_locator(mdates.YearLocator())
    equity_axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    _style_axis(drawdown_axis)
    drawdown_axis.set_title("Drawdown path", loc="left", fontsize=11, color=FOREGROUND)
    drawdown_axis.set_ylabel("Drawdown (%)")
    drawdown_axis.axhline(0, color=GRID, linewidth=0.8)
    drawdown_axis.xaxis.set_major_locator(mdates.YearLocator())
    drawdown_axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    factors = asset_summary["profit_factor"].to_numpy(dtype=float)
    labels = asset_summary["asset"].tolist()
    colors = [GREEN if factor > 1 else RED for factor in factors]
    bars = factor_axis.bar(labels, factors, color=colors, alpha=0.9)
    _style_axis(factor_axis, grid=False)
    factor_axis.set_title("Profit factor by asset", loc="left", fontsize=11, color=FOREGROUND)
    factor_axis.set_ylabel("Profit factor")
    factor_axis.axhline(1.0, color=AMBER, linewidth=1.2, linestyle="--")
    for bar, value in zip(bars, factors):
        factor_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.04,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            color=FOREGROUND,
            fontsize=8,
        )
    factor_axis.set_ylim(0, max(2.3, float(np.nanmax(factors)) + 0.25))

    _style_axis(risk_axis)
    risk_axis.set_title("Edge versus historical risk", loc="left", fontsize=11, color=FOREGROUND)
    risk_axis.set_xlabel("Maximum drawdown (%)")
    risk_axis.set_ylabel("Expectancy per trade (%)")
    risk_axis.axhline(0.0, color=AMBER, linewidth=1.0, linestyle="--")
    for asset_index, row in enumerate(asset_summary.to_dict("records")):
        color = SERIES[asset_index % len(SERIES)]
        risk_axis.scatter(
            row["max_drawdown_pct"], row["expectancy_pct"], s=80, color=color, zorder=3
        )
        risk_axis.annotate(
            row["asset"],
            (row["max_drawdown_pct"], row["expectancy_pct"]),
            xytext=(6, 5),
            textcoords="offset points",
            color=FOREGROUND,
            fontsize=8,
        )

    figure.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.07, hspace=0.34, wspace=0.20)
    _save_figure(figure, path)
    return path


def create_yearly_heatmap(results: dict[str, object], output_path: str | Path) -> Path:
    yearly = results["yearly"]
    if not isinstance(yearly, pd.DataFrame):
        raise TypeError("results must include yearly DataFrame")
    selected = yearly[
        (yearly["scope"] == "ASSET_COMMON_PERIOD") & (yearly["scenario"] == "ASSET_COMMON_COST")
    ]
    matrix = selected.pivot(index="asset", columns="year", values="simple_return_pct").sort_index()

    figure, axis = plt.subplots(figsize=(12, 4.8), facecolor=BACKGROUND)
    _style_axis(axis, grid=False)
    values = matrix.to_numpy(dtype=float)
    limit = max(abs(np.nanmin(values)), abs(np.nanmax(values)), 1.0)
    image = axis.imshow(values, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(matrix.columns)), labels=[str(year) for year in matrix.columns])
    axis.set_yticks(range(len(matrix.index)), labels=matrix.index)
    axis.set_title(
        "Yearly simple return after costs", loc="left", fontsize=12, pad=12, color=FOREGROUND
    )
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            text = "—" if np.isnan(value) else f"{value:.1f}%"
            axis.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                color="#08111f" if not np.isnan(value) and abs(value) < limit * 0.35 else "white",
                fontsize=8,
                fontweight="bold",
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.ax.tick_params(colors=MUTED, labelsize=8)
    colorbar.set_label("Return (%)", color=MUTED)
    _save_figure(figure, Path(output_path))
    return Path(output_path)


def create_trade_distribution(results: dict[str, object], output_path: str | Path) -> Path:
    trades = _asset_cost_trades(results["trades"])
    assets = sorted(trades["asset"].unique())
    values = [trades.loc[trades["asset"] == asset, "pnl_pct"].to_numpy(float) for asset in assets]

    figure, axis = plt.subplots(figsize=(12, 5), facecolor=BACKGROUND)
    _style_axis(axis)
    box = axis.boxplot(
        values,
        tick_labels=assets,
        patch_artist=True,
        showmeans=True,
        flierprops={
            "marker": "o",
            "markerfacecolor": "none",
            "markeredgecolor": FOREGROUND,
            "markersize": 6,
        },
    )
    for index, patch in enumerate(box["boxes"]):
        patch.set_facecolor(SERIES[index % len(SERIES)])
        patch.set_alpha(0.55)
    for collection in (box["whiskers"], box["caps"], box["medians"]):
        for item in collection:
            item.set_color(FOREGROUND)
    for mean in box["means"]:
        mean.set_markerfacecolor(AMBER)
        mean.set_markeredgecolor(AMBER)
    axis.axhline(0.0, color=AMBER, linestyle="--", linewidth=1.0)
    axis.set_title(
        "Trade-return distribution after costs", loc="left", fontsize=12, color=FOREGROUND
    )
    axis.set_ylabel("Return per trade (%)")
    _save_figure(figure, Path(output_path))
    return Path(output_path)


def create_signal_map(results: dict[str, object], output_path: str | Path) -> Path:
    prepared = results["prepared"]
    frame = prepared[("BTC", "BYBIT_PERPETUAL")]
    signals = frame[frame["buy_signal"]]
    figure, axis = plt.subplots(figsize=(14, 5.2), facecolor=BACKGROUND)
    _style_axis(axis)
    axis.plot(frame.index, frame["close"], color=SERIES[0], linewidth=1.0, label="BTC close")
    axis.plot(frame.index, frame["ema200"], color=AMBER, linewidth=1.0, label="EMA 200")
    axis.scatter(
        signals.index,
        signals["close"],
        marker="^",
        s=55,
        color=GREEN,
        edgecolor=BACKGROUND,
        linewidth=0.7,
        label="Frozen BUY signal",
        zorder=4,
    )
    axis.set_yscale("log")
    axis.set_title(
        "BTC Bybit · frozen signals in market context", loc="left", fontsize=12, color=FOREGROUND
    )
    axis.set_ylabel("USDT · logarithmic scale")
    axis.legend(frameon=False, labelcolor=FOREGROUND, fontsize=8, ncol=3)
    axis.xaxis.set_major_locator(mdates.YearLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _save_figure(figure, Path(output_path))
    return Path(output_path)


def _format_float(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if np.isnan(number):
        return "—"
    if np.isinf(number):
        return "∞"
    return f"{number:.{digits}f}"


def _summary_table(frame: pd.DataFrame) -> str:
    headers = ["Asset", "Trades", "Win rate", "PF", "Expectancy", "Return", "Max DD", "Result"]
    body: list[str] = []
    for row in frame.to_dict("records"):
        result_class = "positive" if row["classification"] == "HISTORICALLY_POSITIVE" else "failed"
        body.append(
            "<tr>"
            f"<td><strong>{escape(str(row['asset']))}</strong></td>"
            f"<td>{int(row['trades'])}</td>"
            f"<td>{_format_float(row['win_rate_pct'])}%</td>"
            f"<td>{_format_float(row['profit_factor'], 3)}</td>"
            f"<td>{_format_float(row['expectancy_pct'], 4)}%</td>"
            f"<td>{_format_float(row['simple_return_pct'])}%</td>"
            f"<td>{_format_float(row['max_drawdown_pct'])}%</td>"
            f'<td><span class="tag {result_class}">{escape(str(row["classification"]))}</span></td>'
            "</tr>"
        )
    header_html = "".join(f"<th>{header}</th>" for header in headers)
    return f'<div class="table-wrap"><table><thead><tr>{header_html}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _robustness_table(results: dict[str, object]) -> str:
    bootstrap = results["bootstrap"]
    outliers = results["outliers"]
    selected_bootstrap = bootstrap[
        (bootstrap["scope"] == "ASSET_COMMON_PERIOD")
        & (bootstrap["scenario"] == "ASSET_COMMON_COST")
    ].set_index("asset")
    selected_outliers = outliers[
        (outliers["scope"] == "ASSET_COMMON_PERIOD")
        & (outliers["scenario"] == "ASSET_COMMON_COST")
        & (outliers["removed_top_winners"] == 1)
    ].set_index("asset")
    rows: list[str] = []
    for asset in sorted(selected_bootstrap.index):
        boot = selected_bootstrap.loc[asset]
        outlier = selected_outliers.loc[asset]
        rows.append(
            "<tr>"
            f"<td><strong>{escape(str(asset))}</strong></td>"
            f"<td>{_format_float(boot['probability_expectancy_positive_pct'])}%</td>"
            f"<td>{_format_float(boot['expectancy_ci_low_pct'], 3)}% to {_format_float(boot['expectancy_ci_high_pct'], 3)}%</td>"
            f"<td>{_format_float(boot['probability_pf_above_one_pct'])}%</td>"
            f"<td>{_format_float(outlier['profit_factor'], 3)}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Asset</th><th>P(Exp &gt; 0)</th><th>Expectancy 95% CI</th>"
        "<th>P(PF &gt; 1)</th><th>PF after top winner removed</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def create_html_report(results: dict[str, object], output_path: str | Path) -> Path:
    summary = _asset_cost_summary(results["summary"])
    quality = results["data_quality"]
    manifest = results["manifest"]
    portfolio = results["portfolio_summary"].iloc[0]
    strongest = summary.sort_values("profit_factor", ascending=False).iloc[0]
    positive = int((summary["classification"] == "HISTORICALLY_POSITIVE").sum())
    gaps = int(quality["non_hourly_gaps"].sum())
    robustness_table = _robustness_table(results)

    quality_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['asset']))}</td>"
        f"<td>{escape(str(row['market']))}</td>"
        f"<td>{int(row['rows']):,}</td>"
        f"<td>{int(row['non_hourly_gaps'])}</td>"
        f"<td>{escape(str(row['end']))}</td>"
        "</tr>"
        for row in quality.to_dict("records")
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>C2G Backtest Report</title>
<style>
:root {{ color-scheme: dark; --bg:#070b14; --panel:#111827; --text:#e5e7eb; --muted:#94a3b8; --border:#263244; --green:#22c55e; --red:#ef4444; --amber:#f59e0b; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.55 Inter,Segoe UI,Arial,sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:42px 24px 72px; }}
h1 {{ font-size:36px; margin:0 0 8px; letter-spacing:-.03em; }}
h2 {{ margin:38px 0 12px; font-size:21px; }}
p {{ color:var(--muted); margin:8px 0; }}
.eyebrow {{ color:var(--green); font-weight:700; letter-spacing:.12em; text-transform:uppercase; font-size:12px; }}
.metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:26px 0; }}
.metric {{ background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:16px; }}
.metric span {{ display:block; color:var(--muted); font-size:12px; }}
.metric strong {{ display:block; margin-top:4px; font-size:23px; }}
img {{ width:100%; display:block; border:1px solid var(--border); border-radius:14px; background:var(--panel); margin:14px 0; }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
.table-wrap {{ overflow-x:auto; border:1px solid var(--border); border-radius:14px; }}
table {{ width:100%; border-collapse:collapse; background:var(--panel); }}
th,td {{ padding:11px 12px; text-align:right; border-bottom:1px solid var(--border); white-space:nowrap; }}
th:first-child,td:first-child {{ text-align:left; }}
th {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
.tag {{ display:inline-block; padding:3px 7px; border-radius:999px; font-size:11px; font-weight:700; }}
.tag.positive {{ color:var(--green); background:rgba(34,197,94,.12); }}
.tag.failed {{ color:var(--red); background:rgba(239,68,68,.12); }}
.warning {{ border-left:3px solid var(--amber); padding:11px 14px; background:rgba(245,158,11,.08); color:var(--muted); }}
code {{ color:var(--text); }}
@media (max-width:760px) {{ .metrics,.grid {{ grid-template-columns:1fr; }} h1 {{ font-size:29px; }} main {{ padding:28px 14px 52px; }} }}
</style>
</head>
<body><main>
<div class="eyebrow">C2G System Pro · V1.18 research report</div>
<h1>Frozen BUY 24H backtest</h1>
<p>Generated {escape(str(manifest["run_utc"]))}. The same frozen parameters were applied without asset-specific retuning.</p>
<div class="metrics">
  <div class="metric"><span>Positive assets after costs</span><strong>{positive}/{len(summary)}</strong></div>
  <div class="metric"><span>Strongest historical PF</span><strong>{escape(str(strongest["asset"]))} · {_format_float(strongest["profit_factor"], 3)}</strong></div>
  <div class="metric"><span>Equal-slot portfolio return</span><strong>{_format_float(portfolio["compounded_return_pct"])}%</strong></div>
  <div class="metric"><span>Maximum nominal exposure</span><strong>{_format_float(portfolio["maximum_nominal_exposure_pct"], 0)}%</strong></div>
</div>
<img src="c2g-backtest-dashboard.png" alt="Equity, drawdown, profit factor and risk comparison dashboard">
<h2>Common-period cross-asset result</h2>
{_summary_table(summary)}
<h2>Uncertainty and top-winner dependency</h2>
<p>Bootstrap probabilities use 20,000 resamples. Wide intervals are expected because each asset has only 12–20 trades.</p>
{robustness_table}
<div class="grid">
  <img src="yearly-returns.png" alt="Yearly returns heatmap">
  <img src="trade-distribution.png" alt="Distribution of returns per trade">
</div>
<h2>Signal context</h2>
<img src="btc-signal-map.png" alt="BTC Bybit price, EMA 200 and frozen buy signals">
<h2>Data-quality audit</h2>
<p>{gaps} non-hourly gaps were detected across the source files. They remain visible in the audit instead of being silently filled.</p>
<div class="table-wrap"><table><thead><tr><th>Asset</th><th>Market</th><th>Rows</th><th>Gaps</th><th>Last candle UTC</th></tr></thead><tbody>{quality_rows}</tbody></table></div>
<h2>Methodology</h2>
<p>BUY only: Supertrend 10/3 flip, ADX(14) &gt; 25 and rising, close above a rising EMA 200. Entry occurs at the next candle open. Exit occurs at the close of the 24th held one-hour candle.</p>
<p>The portfolio diagnostic uses five fixed 20% capital slots with no leverage. It is a realized trade-level proxy, not a mark-to-market risk engine.</p>
<div class="warning"><strong>Research limitation.</strong> Funding, order-book depth, latency, partial fills and exchange outages are not modeled. Historical and forward samples are small, and correlated crypto assets are not independent evidence. This report is not financial advice.</div>
</main></body></html>
"""
    path = Path(output_path)
    path.write_text(html, encoding="utf-8")
    return path


def create_report_bundle(results: dict[str, object], output_dir: str | Path) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    files = {
        "dashboard": create_dashboard(results, destination / "c2g-backtest-dashboard.png"),
        "yearly": create_yearly_heatmap(results, destination / "yearly-returns.png"),
        "distribution": create_trade_distribution(results, destination / "trade-distribution.png"),
        "signals": create_signal_map(results, destination / "btc-signal-map.png"),
    }
    files["html"] = create_html_report(results, destination / "c2g-backtest-report.html")
    return files
