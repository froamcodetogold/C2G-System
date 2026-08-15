from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pandas_ta as ta


# =============================================================================
# C2G V2 RESEARCH - BASELINE REPRODUCTION / DATA INTEGRITY
# =============================================================================
#
# PURPOSE
# -------
# Before changing ANY strategy parameter, reproduce the frozen V1 baseline
# inside C2GSystem_Clone and verify the datasets.
#
# This file DOES NOT optimize anything.
# This file DOES NOT modify the original C2G System folder.
#
# Frozen baseline:
# - 1H
# - BUY only
# - Supertrend 10 / 3.0 bullish flip
# - ADX(14) > 25
# - ADX current > ADX previous
# - Bull regime:
#       Close > EMA200
#       EMA200 current > EMA200 50 bars ago
# - Entry: next candle OPEN
# - Exit: close after 24 x 1H bars
# - Fee: 0.055% per side
# - Slippage: 0.020% per side
#
# The expected metrics below come from the historical V1.16 common-period
# robustness audit. They are used ONLY as a reproduction checksum.
# =============================================================================


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "research_v2" / "results"
REPORTS_DIR = ROOT / "research_v2" / "reports"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "BTC": ROOT / "btc_data_bybit_perp_1h.csv",
    "ETH": ROOT / "eth_data_bybit_perp_1h.csv",
    "SOL": ROOT / "sol_data_bybit_perp_1h.csv",
}

SUPERTREND_LENGTH = 10
SUPERTREND_MULTIPLIER = 3.0

ADX_LENGTH = 14
ADX_MIN = 25.0

EMA_LENGTH = 200
EMA_SLOPE_LOOKBACK = 50

TIME_EXIT_BARS = 24

FEE_PCT_PER_SIDE = 0.055
SLIPPAGE_PCT_PER_SIDE = 0.020

WARMUP_HOURS = 1000

EXPECTED = {
    "BTC": {
        "trades": 20,
        "wr": 50.00,
        "pnl": 6.73,
        "pf": 1.412,
        "exp": 0.3366,
        "median": -0.0157,
        "dd": -7.68,
    },
    "ETH": {
        "trades": 14,
        "wr": 57.14,
        "pnl": 17.59,
        "pf": 1.953,
        "exp": 1.2563,
        "median": 1.0924,
        "dd": -8.32,
    },
    "SOL": {
        "trades": 13,
        "wr": 46.15,
        "pnl": 8.18,
        "pf": 1.553,
        "exp": 0.6291,
        "median": -0.3793,
        "dd": -13.19,
    },
}

TOLERANCE = {
    "wr": 0.05,
    "pnl": 0.05,
    "pf": 0.015,
    "exp": 0.005,
    "median": 0.01,
    "dd": 0.05,
}


def heading(text: str, width: int = 150) -> None:
    print()
    print("=" * width)
    print(text)
    print("=" * width)


def find_column(columns, prefix: str) -> str:
    matches = [c for c in columns if str(c).startswith(prefix)]
    if not matches:
        raise KeyError(
            f"Nenhuma coluna começando com '{prefix}'. "
            f"Disponíveis: {list(columns)}"
        )
    return matches[0]


def load_and_audit(asset: str, path: Path) -> tuple[pd.DataFrame, dict]:
    if not path.exists():
        raise FileNotFoundError(f"{asset}: arquivo não encontrado: {path}")

    raw = pd.read_csv(path)

    if "timestamp" not in raw.columns:
        raise ValueError(f"{asset}: coluna 'timestamp' ausente.")

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in [str(x).strip().lower() for x in raw.columns]]
    if missing:
        raise ValueError(f"{asset}: colunas OHLC ausentes: {missing}")

    raw.columns = [str(c).strip().lower() for c in raw.columns]
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")

    invalid_timestamp_rows = int(raw["timestamp"].isna().sum())
    raw = raw.dropna(subset=["timestamp"]).copy()

    duplicate_timestamps = int(raw["timestamp"].duplicated(keep=False).sum())

    raw = raw.sort_values("timestamp")
    raw = raw.drop_duplicates(subset=["timestamp"], keep="last")
    raw = raw.set_index("timestamp")

    for col in required:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    missing_ohlc_rows = int(raw[required].isna().any(axis=1).sum())
    raw = raw.dropna(subset=required).copy()

    nonpositive_rows = int((raw[required] <= 0).any(axis=1).sum())

    high_invalid = int(
        (
            raw["high"]
            < raw[["open", "close", "low"]].max(axis=1)
        ).sum()
    )

    low_invalid = int(
        (
            raw["low"]
            > raw[["open", "close", "high"]].min(axis=1)
        ).sum()
    )

    deltas = raw.index.to_series().diff().dropna()

    one_hour = pd.Timedelta(hours=1)
    gap_rows = int((deltas > one_hour).sum())
    sub_hour_rows = int((deltas < one_hour).sum())

    if len(deltas):
        most_common_delta = deltas.value_counts().index[0]
        one_hour_ratio = float((deltas == one_hour).mean() * 100.0)
        max_gap = deltas.max()
    else:
        most_common_delta = pd.NaT
        one_hour_ratio = np.nan
        max_gap = pd.NaT

    audit = {
        "asset": asset,
        "file": path.name,
        "rows_clean": len(raw),
        "first_timestamp": raw.index.min(),
        "last_timestamp": raw.index.max(),
        "invalid_timestamp_rows": invalid_timestamp_rows,
        "duplicate_timestamp_rows_before_dedup": duplicate_timestamps,
        "missing_ohlc_rows_before_drop": missing_ohlc_rows,
        "nonpositive_ohlc_rows": nonpositive_rows,
        "high_invariant_violations": high_invalid,
        "low_invariant_violations": low_invalid,
        "gaps_gt_1h": gap_rows,
        "intervals_lt_1h": sub_hour_rows,
        "one_hour_interval_pct": one_hour_ratio,
        "most_common_interval": str(most_common_delta),
        "max_gap": str(max_gap),
    }

    return raw, audit


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    st = ta.supertrend(
        high=out["high"],
        low=out["low"],
        close=out["close"],
        length=SUPERTREND_LENGTH,
        multiplier=SUPERTREND_MULTIPLIER,
    )

    out["Trend_Direction"] = st[
        find_column(st.columns, "SUPERTd_")
    ]

    out["ST_Buy_Flip"] = (
        out["Trend_Direction"].diff() == 2
    )

    adx_df = ta.adx(
        high=out["high"],
        low=out["low"],
        close=out["close"],
        length=ADX_LENGTH,
    )

    out["ADX"] = adx_df[
        find_column(adx_df.columns, "ADX_")
    ]

    out["ADX_Rising"] = (
        out["ADX"] > out["ADX"].shift(1)
    )

    out["EMA200"] = ta.ema(
        out["close"],
        length=EMA_LENGTH,
    )

    out["EMA200_Lag"] = out["EMA200"].shift(
        EMA_SLOPE_LOOKBACK
    )

    out["Bull_Regime"] = (
        (out["close"] > out["EMA200"])
        & (out["EMA200"] > out["EMA200_Lag"])
    )

    out["Buy_Signal"] = (
        out["ST_Buy_Flip"]
        & (out["ADX"] > ADX_MIN)
        & out["ADX_Rising"]
        & out["Bull_Regime"]
    )

    return out


def calculate_return(entry_price: float, exit_price: float) -> float:
    gross = ((exit_price - entry_price) / entry_price) * 100.0

    costs = (
        FEE_PCT_PER_SIDE * 2.0
        + SLIPPAGE_PCT_PER_SIDE * 2.0
    )

    return gross - costs


def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduces the V1.16 bar-count convention exactly:
    the entry candle is counted as held bar #1.
    """
    trades = []

    position = False
    entry_price = None
    entry_time = None
    bars_held = 0

    for i in range(1, len(df)):
        previous = df.iloc[i - 1]
        current = df.iloc[i]

        if (
            not position
            and bool(previous["Buy_Signal"])
        ):
            position = True
            entry_price = float(current["open"])
            entry_time = df.index[i]
            bars_held = 0

        if not position:
            continue

        bars_held += 1

        if bars_held >= TIME_EXIT_BARS:
            exit_price = float(current["close"])

            pnl = calculate_return(
                entry_price,
                exit_price,
            )

            trades.append(
                {
                    "entry_time": entry_time,
                    "exit_time": df.index[i],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": pnl,
                }
            )

            position = False
            entry_price = None
            entry_time = None
            bars_held = 0

    trades = pd.DataFrame(trades)

    if not trades.empty:
        trades["entry_time"] = pd.to_datetime(
            trades["entry_time"]
        )
        trades["exit_time"] = pd.to_datetime(
            trades["exit_time"]
        )
        trades["year"] = trades["entry_time"].dt.year

    return trades


def metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "wr": 0.0,
            "pnl": 0.0,
            "pf": 0.0,
            "exp": 0.0,
            "median": 0.0,
            "dd": 0.0,
            "trade_sharpe_like": np.nan,
        }

    ordered = trades.sort_values("entry_time").copy()
    returns = ordered["pnl_pct"].astype(float)

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    gp = float(wins.sum()) if len(wins) else 0.0
    gl = abs(float(losses.sum())) if len(losses) else 0.0

    pf = gp / gl if gl > 0 else np.inf

    equity = [100.0]

    for pnl in returns:
        equity.append(
            equity[-1] * (1.0 + pnl / 100.0)
        )

    equity = pd.Series(equity, dtype=float)
    peak = equity.cummax()
    drawdown = ((equity / peak) - 1.0) * 100.0

    std = float(returns.std(ddof=1)) if len(returns) > 1 else np.nan

    # NOT annualized calendar Sharpe.
    # This is only a normalized trade-return diagnostic.
    trade_sharpe_like = (
        float(returns.mean() / std * np.sqrt(len(returns)))
        if len(returns) > 1 and std > 0
        else np.nan
    )

    return {
        "trades": int(len(ordered)),
        "wr": float((returns > 0).mean() * 100.0),
        "pnl": float(returns.sum()),
        "pf": float(pf),
        "exp": float(returns.mean()),
        "median": float(returns.median()),
        "dd": float(drawdown.min()),
        "trade_sharpe_like": trade_sharpe_like,
    }


def compare_to_expected(asset: str, actual: dict) -> tuple[bool, list[str]]:
    expected = EXPECTED[asset]
    issues = []

    if actual["trades"] != expected["trades"]:
        issues.append(
            f"trades esperado={expected['trades']} obtido={actual['trades']}"
        )

    for key in ["wr", "pnl", "pf", "exp", "median", "dd"]:
        diff = abs(actual[key] - expected[key])
        if diff > TOLERANCE[key]:
            issues.append(
                f"{key} esperado={expected[key]:.4f} "
                f"obtido={actual[key]:.4f} "
                f"diff={diff:.4f}"
            )

    return len(issues) == 0, issues


def main() -> None:
    heading("C2G V2 - BASELINE REPRODUCTION / DATA INTEGRITY")

    print(f"Research clone root: {ROOT}")
    print("Nenhum parâmetro será otimizado nesta execução.")

    raw_by_asset = {}
    audit_rows = []

    heading("1. DATA INTEGRITY")

    for asset, path in FILES.items():
        df, audit = load_and_audit(asset, path)
        raw_by_asset[asset] = df
        audit_rows.append(audit)

        print()
        print(
            f"{asset} | rows={len(df)} | "
            f"{df.index.min()} -> {df.index.max()}"
        )
        print(
            f"  duplicates(before dedup): "
            f"{audit['duplicate_timestamp_rows_before_dedup']}"
        )
        print(
            f"  OHLC invariant violations: "
            f"high={audit['high_invariant_violations']} "
            f"low={audit['low_invariant_violations']}"
        )
        print(
            f"  gaps > 1H: {audit['gaps_gt_1h']} | "
            f"1H intervals: {audit['one_hour_interval_pct']:.4f}% | "
            f"max gap: {audit['max_gap']}"
        )

    audit_df = pd.DataFrame(audit_rows)
    audit_path = RESULTS_DIR / "v2_baseline_data_integrity.csv"
    audit_df.to_csv(audit_path, index=False)

    overlap_start = max(
        df.index.min()
        for df in raw_by_asset.values()
    )
    overlap_end = min(
        df.index.max()
        for df in raw_by_asset.values()
    )

    warmup_start = (
        overlap_start
        - pd.Timedelta(hours=WARMUP_HOURS)
    )

    heading("2. COMMON PERIOD")

    print(f"Common period: {overlap_start} -> {overlap_end}")
    print(f"Warmup starts: {warmup_start}")

    prepared = {}

    for asset, raw in raw_by_asset.items():
        calc = prepare_indicators(
            raw.loc[warmup_start:overlap_end].copy()
        )

        prepared[asset] = calc.loc[
            overlap_start:overlap_end
        ].copy()

        print(
            f"{asset}: candles={len(prepared[asset])} | "
            f"BUY signals={int(prepared[asset]['Buy_Signal'].sum())}"
        )

    heading("3. FROZEN V1 BASELINE - COST STRESSED")

    summary_rows = []
    all_pass = True
    mismatch_messages = []

    for asset, df in prepared.items():
        trades = run_backtest(df)
        m = metrics(trades)

        passed, issues = compare_to_expected(asset, m)
        all_pass = all_pass and passed

        status = "PASS" if passed else "MISMATCH"

        pf_text = (
            "inf"
            if np.isinf(m["pf"])
            else f"{m['pf']:.3f}"
        )

        print(
            f"{asset:<4} | {status:<8} | "
            f"Trades {m['trades']:>3} | "
            f"WR {m['wr']:>6.2f}% | "
            f"PnL {m['pnl']:>8.2f}% | "
            f"PF {pf_text:>6} | "
            f"Exp {m['exp']:>8.4f}% | "
            f"Median {m['median']:>8.4f}% | "
            f"DD {m['dd']:>7.2f}% | "
            f"TradeSharpeLike {m['trade_sharpe_like']:.3f}"
        )

        summary_rows.append(
            {
                "asset": asset,
                "baseline_reproduction": status,
                **m,
            }
        )

        trades_path = (
            RESULTS_DIR
            / f"v2_baseline_{asset.lower()}_trades.csv"
        )
        trades.to_csv(trades_path, index=False)

        if issues:
            for issue in issues:
                mismatch_messages.append(
                    f"{asset}: {issue}"
                )

    summary_df = pd.DataFrame(summary_rows)

    summary_path = (
        RESULTS_DIR
        / "v2_baseline_validation_summary.csv"
    )
    summary_df.to_csv(summary_path, index=False)

    heading("4. BASELINE REPRODUCTION VERDICT")

    if all_pass:
        print("PASS - O clone reproduz a baseline histórica V1.16.")
        print("Podemos iniciar os experimentos C2G V2.")
    else:
        print("MISMATCH - NÃO iniciar otimização ainda.")
        print("Diferenças encontradas:")
        for message in mismatch_messages:
            print(f"  - {message}")

    report_path = (
        REPORTS_DIR
        / "v2_baseline_reproduction_report.txt"
    )

    lines = [
        "C2G V2 - BASELINE REPRODUCTION REPORT",
        "",
        f"Root: {ROOT}",
        f"Common period: {overlap_start} -> {overlap_end}",
        f"Verdict: {'PASS' if all_pass else 'MISMATCH'}",
        "",
        "Important:",
        "- This is historical reproduction, not new OOS evidence.",
        "- No parameter was optimized.",
        "- TradeSharpeLike is NOT an annualized calendar Sharpe.",
        "",
    ]

    if mismatch_messages:
        lines.append("Mismatches:")
        lines.extend(
            f"- {m}"
            for m in mismatch_messages
        )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    heading("5. OUTPUT FILES")

    print(audit_path)
    print(summary_path)
    print(report_path)

    for asset in FILES:
        print(
            RESULTS_DIR
            / f"v2_baseline_{asset.lower()}_trades.csv"
        )

    print()
    print(
        "NEXT STEP ONLY IF VERDICT = PASS:\n"
        "Experiment 001 - predeclared ADX threshold sensitivity."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 150)
        print("FATAL ERROR")
        print("=" * 150)
        print(type(exc).__name__, str(exc))
        sys.exit(1)
