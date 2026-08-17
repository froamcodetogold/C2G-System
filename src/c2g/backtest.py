from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256

import numpy as np
import pandas as pd

from .config import FrozenV1Config

TRADE_COLUMNS = [
    "trade_id",
    "asset",
    "market",
    "signal_time",
    "entry_time",
    "entry_price",
    "exit_bar_time",
    "exit_execution_time",
    "exit_price",
    "bars_held",
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
]


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    trades: int
    wins: int
    losses: int
    win_rate_pct: float
    simple_return_pct: float
    compounded_return_pct: float
    ending_equity: float
    profit_factor: float
    expectancy_pct: float
    median_trade_pct: float
    max_drawdown_pct: float
    max_loss_streak: int
    payoff_ratio: float
    trade_sharpe: float
    recovery_factor: float
    average_clock_hours: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _trade_id(asset: str, market: str, signal_time: pd.Timestamp, config_hash: str) -> str:
    raw = f"{asset}|{market}|{signal_time.isoformat()}|{config_hash}"
    return sha256(raw.encode("utf-8")).hexdigest()[:20]


def run_time_exit_backtest(
    prepared: pd.DataFrame,
    *,
    asset: str,
    market: str,
    config: FrozenV1Config | None = None,
    include_costs: bool = True,
) -> pd.DataFrame:
    """Run the frozen next-open/24-candle-close model without overlapping positions."""

    settings = config or FrozenV1Config()
    rows: list[dict[str, object]] = []
    in_position = False
    entry_index = -1
    signal_index = -1

    for current_index in range(1, len(prepared)):
        if not in_position and bool(prepared["buy_signal"].iloc[current_index - 1]):
            in_position = True
            entry_index = current_index
            signal_index = current_index - 1

        if not in_position:
            continue

        bars_held = current_index - entry_index + 1
        if bars_held < settings.time_exit_bars:
            continue

        signal_time = pd.Timestamp(prepared.index[signal_index])
        entry_time = pd.Timestamp(prepared.index[entry_index])
        exit_bar_time = pd.Timestamp(prepared.index[current_index])
        exit_execution_time = exit_bar_time + pd.Timedelta(hours=settings.timeframe_hours)
        entry_price = float(prepared["open"].iloc[entry_index])
        exit_price = float(prepared["close"].iloc[current_index])
        gross_return = (exit_price - entry_price) / entry_price * 100.0
        fees = 2.0 * settings.fee_pct_per_side if include_costs else 0.0
        slippage = 2.0 * settings.slippage_pct_per_side if include_costs else 0.0
        funding = settings.funding_pct_per_trade if include_costs else 0.0
        pnl = gross_return - fees - slippage - funding

        window = prepared.iloc[entry_index : current_index + 1]
        mfe = (float(window["high"].max()) - entry_price) / entry_price * 100.0
        mae = (entry_price - float(window["low"].min())) / entry_price * 100.0

        rows.append(
            {
                "trade_id": _trade_id(asset, market, signal_time, settings.fingerprint),
                "asset": asset,
                "market": market,
                "signal_time": signal_time,
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_bar_time": exit_bar_time,
                "exit_execution_time": exit_execution_time,
                "exit_price": exit_price,
                "bars_held": bars_held,
                "clock_hours": (exit_execution_time - entry_time).total_seconds() / 3600.0,
                "gross_return_pct": gross_return,
                "fees_pct": fees,
                "slippage_pct": slippage,
                "funding_pct": funding,
                "pnl_pct": pnl,
                "mfe_pct": mfe,
                "mae_pct": mae,
                "signal_adx": float(prepared["adx"].iloc[signal_index]),
                "config_hash": settings.fingerprint,
            }
        )
        in_position = False

    trades = pd.DataFrame(rows, columns=TRADE_COLUMNS)
    for column in ("signal_time", "entry_time", "exit_bar_time", "exit_execution_time"):
        if column in trades:
            trades[column] = pd.to_datetime(trades[column], utc=True)
    return trades


def max_loss_streak(values: pd.Series | list[float]) -> int:
    longest = 0
    current = 0
    for value in values:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def equity_curve(trades: pd.DataFrame, *, initial_equity: float = 100.0) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            {"equity": [initial_equity], "peak": [initial_equity], "drawdown_pct": [0.0]}
        )

    ordered = trades.sort_values("exit_execution_time")
    compounded = initial_equity * (1.0 + ordered["pnl_pct"] / 100.0).cumprod()
    timestamps = [pd.Timestamp(ordered["entry_time"].iloc[0])]
    timestamps.extend(pd.to_datetime(ordered["exit_execution_time"], utc=True).tolist())
    equity_values = np.concatenate([[initial_equity], compounded.to_numpy(dtype=float)])
    curve = pd.DataFrame(
        {"equity": equity_values},
        index=pd.DatetimeIndex(timestamps, name="timestamp"),
    )
    curve["peak"] = curve["equity"].cummax()
    curve["drawdown_pct"] = (curve["equity"] / curve["peak"] - 1.0) * 100.0
    return curve


def calculate_metrics(trades: pd.DataFrame) -> BacktestMetrics:
    if trades.empty:
        return BacktestMetrics(
            trades=0,
            wins=0,
            losses=0,
            win_rate_pct=0.0,
            simple_return_pct=0.0,
            compounded_return_pct=0.0,
            ending_equity=100.0,
            profit_factor=0.0,
            expectancy_pct=0.0,
            median_trade_pct=0.0,
            max_drawdown_pct=0.0,
            max_loss_streak=0,
            payoff_ratio=0.0,
            trade_sharpe=0.0,
            recovery_factor=0.0,
            average_clock_hours=0.0,
        )

    ordered = trades.sort_values("entry_time")
    values = ordered["pnl_pct"].astype(float)
    winning = values[values > 0]
    losing = values[values < 0]
    gross_profit = float(winning.sum()) if len(winning) else 0.0
    gross_loss = abs(float(losing.sum())) if len(losing) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    payoff_ratio = (
        float(winning.mean()) / abs(float(losing.mean()))
        if len(winning) and len(losing) and float(losing.mean()) != 0
        else np.inf
        if len(winning)
        else 0.0
    )

    curve = equity_curve(ordered)
    ending_equity = float(curve["equity"].iloc[-1])
    compounded_return = ending_equity - 100.0
    max_drawdown = float(curve["drawdown_pct"].min())
    standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    trade_sharpe = (
        float(values.mean()) / standard_deviation * np.sqrt(len(values))
        if standard_deviation > 0
        else 0.0
    )
    recovery_factor = compounded_return / abs(max_drawdown) if max_drawdown < 0 else np.inf

    return BacktestMetrics(
        trades=len(ordered),
        wins=len(winning),
        losses=len(losing),
        win_rate_pct=len(winning) / len(ordered) * 100.0,
        simple_return_pct=float(values.sum()),
        compounded_return_pct=compounded_return,
        ending_equity=ending_equity,
        profit_factor=float(profit_factor),
        expectancy_pct=float(values.mean()),
        median_trade_pct=float(values.median()),
        max_drawdown_pct=max_drawdown,
        max_loss_streak=max_loss_streak(values.tolist()),
        payoff_ratio=float(payoff_ratio),
        trade_sharpe=float(trade_sharpe),
        recovery_factor=float(recovery_factor),
        average_clock_hours=float(ordered["clock_hours"].mean()),
    )


def yearly_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    years = pd.to_datetime(trades["entry_time"], utc=True).dt.year
    for year, group in trades.groupby(years):
        rows.append({"year": int(year), **calculate_metrics(group).as_dict()})
    return pd.DataFrame(rows)


def top_winner_dependency(trades: pd.DataFrame, *, maximum_removed: int = 3) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ordered = trades.sort_values("pnl_pct", ascending=False)
    for removed in range(maximum_removed + 1):
        subset = ordered.iloc[removed:].copy()
        rows.append({"removed_top_winners": removed, **calculate_metrics(subset).as_dict()})
    return pd.DataFrame(rows)


def bootstrap_metrics(
    trades: pd.DataFrame,
    *,
    samples: int = 20_000,
    seed: int = 42,
) -> dict[str, float | int]:
    values = trades["pnl_pct"].dropna().to_numpy(dtype=float)
    if not len(values):
        return {"sample_trades": 0, "bootstrap_samples": samples}

    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(samples, len(values)), replace=True)
    means = sampled.mean(axis=1)
    positive = np.where(sampled > 0, sampled, 0.0).sum(axis=1)
    negative = np.abs(np.where(sampled < 0, sampled, 0.0).sum(axis=1))
    factors = np.divide(positive, negative, out=np.full(samples, np.inf), where=negative > 0)
    terminal_equity = np.prod(1.0 + sampled / 100.0, axis=1) * 100.0
    finite_factors = factors[np.isfinite(factors)]

    return {
        "sample_trades": len(values),
        "bootstrap_samples": samples,
        "expectancy_ci_low_pct": float(np.quantile(means, 0.025)),
        "expectancy_ci_high_pct": float(np.quantile(means, 0.975)),
        "probability_expectancy_positive_pct": float((means > 0).mean() * 100.0),
        "probability_pf_above_one_pct": float((factors > 1.0).mean() * 100.0),
        "median_profit_factor": (
            float(np.median(finite_factors)) if len(finite_factors) else np.inf
        ),
        "terminal_equity_ci_low": float(np.quantile(terminal_equity, 0.025)),
        "terminal_equity_ci_high": float(np.quantile(terminal_equity, 0.975)),
    }
