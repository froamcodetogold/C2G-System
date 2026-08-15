import pandas as pd
import pandas_ta as ta
import numpy as np
from pathlib import Path


# ============================================================
# C2G SYSTEM PRO - V1.16 CROSS-ASSET ROBUSTNESS AUDIT
#
# Objetivo:
# Auditar a MESMA regra congelada nos 3 ativos que ficaram
# positivos após custos na V1.15:
#
# BTC, ETH, SOL
#
# IMPORTANTE:
# - NÃO altera parâmetro por ativo.
# - NÃO escolhe um "melhor" ativo.
# - NÃO transforma isso em portfolio live.
# - XRP e BNB continuam registrados como falhas da V1.15.
#
# REGRA CONGELADA:
# BUY ONLY
# Supertrend 10 / 3
# ADX(14) > 25
# ADX atual > ADX anterior
# Bull regime:
#   Close > EMA200
#   EMA200 atual > EMA200 de 50 candles atrás
# Entrada:
#   OPEN do próximo candle
# Saída:
#   CLOSE após 24 candles de 1H
#
# Stress de custos:
# 0.055% fee por lado
# 0.020% slippage por lado
#
# Auditorias:
# 1) mesmo período comum BTC/ETH/SOL
# 2) por ano
# 3) remove top 1 / 2 / 3 winners
# 4) bootstrap 20.000 amostras
# 5) leave-one-year-out
# 6) pooled research distribution (N total dos 3 ativos)
#
# Isso é robustez cross-asset, NÃO OOS temporal puro.
# ============================================================


FILES = {
    "BTC": "btc_data_bybit_perp_1h.csv",
    "ETH": "eth_data_bybit_perp_1h.csv",
    "SOL": "sol_data_bybit_perp_1h.csv",
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

BOOTSTRAP_SAMPLES = 20000
BOOTSTRAP_SEED = 42


def find_column(columns, prefix):
    matches = [
        c for c in columns
        if str(c).startswith(prefix)
    ]

    if not matches:
        raise KeyError(
            f"Nenhuma coluna começando com '{prefix}'."
        )

    return matches[0]


def load_ohlcv(path):
    df = pd.read_csv(
        path,
        index_col="timestamp",
        parse_dates=True,
    )

    df = df.sort_index()

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    df.columns = [
        str(c).strip().lower()
        for c in df.columns
    ]

    for col in [
        "open",
        "high",
        "low",
        "close",
    ]:
        if col not in df.columns:
            raise ValueError(
                f"{path}: coluna '{col}' ausente."
            )

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ],
        inplace=True,
    )

    return df


def prepare_indicators(df):
    df = df.copy()

    st = ta.supertrend(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        length=SUPERTREND_LENGTH,
        multiplier=SUPERTREND_MULTIPLIER,
    )

    df["Trend_Direction"] = st[
        find_column(
            st.columns,
            "SUPERTd_",
        )
    ]

    df["Signal"] = (
        df["Trend_Direction"].diff()
    )

    df["ST_Buy_Flip"] = (
        df["Signal"] == 2
    )

    adx_df = ta.adx(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        length=ADX_LENGTH,
    )

    df["ADX"] = adx_df[
        find_column(
            adx_df.columns,
            "ADX_",
        )
    ]

    df["ADX_Rising"] = (
        df["ADX"]
        >
        df["ADX"].shift(1)
    )

    df["EMA200"] = ta.ema(
        df["close"],
        length=EMA_LENGTH,
    )

    df["EMA200_Lag"] = (
        df["EMA200"]
        .shift(
            EMA_SLOPE_LOOKBACK
        )
    )

    df["Bull_Regime"] = (
        (df["close"] > df["EMA200"])
        &
        (
            df["EMA200"]
            >
            df["EMA200_Lag"]
        )
    )

    df["Buy_Signal"] = (
        df["ST_Buy_Flip"]
        &
        (df["ADX"] > ADX_MIN)
        &
        df["ADX_Rising"]
        &
        df["Bull_Regime"]
    )

    return df


def calculate_return(
    entry_price,
    exit_price,
    fee_pct=FEE_PCT_PER_SIDE,
    slippage_pct=SLIPPAGE_PCT_PER_SIDE,
):
    gross = (
        (
            exit_price
            -
            entry_price
        )
        /
        entry_price
    ) * 100.0

    costs = (
        fee_pct * 2.0
        +
        slippage_pct * 2.0
    )

    return gross - costs


def run_backtest(df):
    trades = []

    position = False
    entry_price = None
    entry_time = None
    bars_held = 0

    for i in range(
        1,
        len(df)
    ):
        previous = df.iloc[i - 1]
        current = df.iloc[i]

        if (
            not position
            and
            bool(
                previous["Buy_Signal"]
            )
        ):
            position = True
            entry_price = float(
                current["open"]
            )
            entry_time = df.index[i]
            bars_held = 0

        if not position:
            continue

        bars_held += 1

        if bars_held >= TIME_EXIT_BARS:
            exit_price = float(
                current["close"]
            )

            pnl = calculate_return(
                entry_price,
                exit_price,
            )

            trades.append({
                "entry_time": entry_time,
                "exit_time": df.index[i],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_pct": pnl,
            })

            position = False

    trades = pd.DataFrame(trades)

    if not trades.empty:
        trades["entry_time"] = pd.to_datetime(
            trades["entry_time"]
        )

        trades["exit_time"] = pd.to_datetime(
            trades["exit_time"]
        )

        trades["year"] = (
            trades["entry_time"].dt.year
        )

    return trades


def metrics(trades):
    if trades.empty:
        return {
            "trades": 0,
            "wr": 0.0,
            "pnl": 0.0,
            "pf": 0.0,
            "exp": 0.0,
            "median": 0.0,
            "dd": 0.0,
        }

    ordered = trades.sort_values(
        "entry_time"
    )

    wins = ordered[
        ordered["pnl_pct"] > 0
    ]

    losses = ordered[
        ordered["pnl_pct"] < 0
    ]

    gp = (
        float(
            wins["pnl_pct"].sum()
        )
        if len(wins)
        else 0.0
    )

    gl = (
        abs(
            float(
                losses["pnl_pct"].sum()
            )
        )
        if len(losses)
        else 0.0
    )

    pf = (
        gp / gl
        if gl > 0
        else np.inf
    )

    equity = [100.0]

    for pnl in ordered[
        "pnl_pct"
    ]:
        equity.append(
            equity[-1]
            *
            (
                1.0
                +
                pnl / 100.0
            )
        )

    equity = pd.Series(
        equity,
        dtype=float,
    )

    peak = equity.cummax()

    dd = (
        (
            equity / peak
        )
        -
        1.0
    ) * 100.0

    return {
        "trades": len(ordered),
        "wr": (
            len(wins)
            /
            len(ordered)
            *
            100.0
        ),
        "pnl": float(
            ordered["pnl_pct"].sum()
        ),
        "pf": pf,
        "exp": float(
            ordered["pnl_pct"].mean()
        ),
        "median": float(
            ordered["pnl_pct"].median()
        ),
        "dd": float(
            dd.min()
        ),
    }


def metric_line(label, trades):
    m = metrics(trades)

    pf_text = (
        "inf"
        if np.isinf(m["pf"])
        else f"{m['pf']:.3f}"
    )

    print(
        f"{label:<20} | "
        f"Trades {m['trades']:>3} | "
        f"WR {m['wr']:>6.2f}% | "
        f"PnL {m['pnl']:>8.2f}% | "
        f"PF {pf_text:>6} | "
        f"Exp {m['exp']:>8.4f}% | "
        f"Median {m['median']:>8.4f}% | "
        f"DD {m['dd']:>7.2f}%"
    )

    return m


def remove_top_winners(trades, n):
    if n <= 0:
        return trades.copy()

    return (
        trades
        .sort_values(
            "pnl_pct",
            ascending=False,
        )
        .iloc[n:]
        .copy()
    )


def bootstrap_report(
    trades,
    samples=BOOTSTRAP_SAMPLES,
    seed=BOOTSTRAP_SEED,
):
    values = (
        trades["pnl_pct"]
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    if len(values) == 0:
        return {}

    rng = np.random.default_rng(
        seed
    )

    n = len(values)

    means = np.empty(
        samples,
        dtype=float,
    )

    pfs = np.empty(
        samples,
        dtype=float,
    )

    for i in range(samples):
        sample = rng.choice(
            values,
            size=n,
            replace=True,
        )

        means[i] = sample.mean()

        gp = sample[
            sample > 0
        ].sum()

        gl = abs(
            sample[
                sample < 0
            ].sum()
        )

        pfs[i] = (
            gp / gl
            if gl > 0
            else np.inf
        )

    finite_pf = pfs[
        np.isfinite(pfs)
    ]

    return {
        "n": n,
        "exp_ci_low": float(
            np.quantile(
                means,
                0.025,
            )
        ),
        "exp_ci_high": float(
            np.quantile(
                means,
                0.975,
            )
        ),
        "prob_exp_positive": float(
            (
                means > 0
            ).mean()
            * 100.0
        ),
        "prob_pf_gt_1": float(
            (
                pfs > 1.0
            ).mean()
            * 100.0
        ),
        "median_pf": (
            float(
                np.median(
                    finite_pf
                )
            )
            if len(finite_pf)
            else np.inf
        ),
    }


def leave_one_year_out(trades):
    rows = []

    if trades.empty:
        return pd.DataFrame()

    for year in sorted(
        trades["year"].unique()
    ):
        subset = trades[
            trades["year"] != year
        ].copy()

        m = metrics(subset)

        rows.append({
            "removed_year": int(year),
            **m,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":

    print()
    print("=" * 150)
    print(
        "C2G V1.16 - CROSS-ASSET ROBUSTNESS AUDIT"
    )
    print("=" * 150)

    raw = {}

    for asset, path in FILES.items():
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Arquivo ausente: {path}"
            )

        raw[asset] = load_ohlcv(path)

    common_start = max(
        df.index.min()
        for df in raw.values()
    )

    common_end = min(
        df.index.max()
        for df in raw.values()
    )

    warmup_start = (
        common_start
        -
        pd.Timedelta(
            hours=1000
        )
    )

    prepared = {}

    for asset, df in raw.items():
        calc = prepare_indicators(
            df.loc[
                warmup_start:
                common_end
            ].copy()
        )

        prepared[asset] = (
            calc.loc[
                common_start:
                common_end
            ].copy()
        )

    print(
        f"Common period BTC/ETH/SOL: "
        f"{common_start} -> {common_end}"
    )

    trades_by_asset = {}

    print()
    print("=" * 150)
    print(
        "COST-STRESSED RESULTS - SAME PERIOD"
    )
    print("=" * 150)

    for asset, df in prepared.items():
        trades = run_backtest(df)

        trades["asset"] = asset

        trades_by_asset[
            asset
        ] = trades

        metric_line(
            asset,
            trades,
        )

    # ========================================================
    # YEARLY
    # ========================================================

    yearly_rows = []

    print()
    print("=" * 150)
    print("RESULT BY YEAR")
    print("=" * 150)

    for asset, trades in (
        trades_by_asset.items()
    ):
        print()
        print(asset)
        print("-" * 150)

        for year, group in (
            trades.groupby("year")
        ):
            m = metric_line(
                str(int(year)),
                group,
            )

            yearly_rows.append({
                "asset": asset,
                "year": int(year),
                **m,
            })

    # ========================================================
    # TOP WINNER DEPENDENCY
    # ========================================================

    outlier_rows = []

    print()
    print("=" * 150)
    print("TOP WINNER DEPENDENCY")
    print("=" * 150)

    for asset, trades in (
        trades_by_asset.items()
    ):
        print()
        print(asset)
        print("-" * 150)

        for n_remove in [
            0,
            1,
            2,
            3,
        ]:
            subset = remove_top_winners(
                trades,
                n_remove,
            )

            label = (
                "ORIGINAL"
                if n_remove == 0
                else
                f"REMOVE TOP {n_remove}"
            )

            m = metric_line(
                label,
                subset,
            )

            outlier_rows.append({
                "asset": asset,
                "removed_top_winners": n_remove,
                **m,
            })

    # ========================================================
    # BOOTSTRAP
    # ========================================================

    bootstrap_rows = []

    print()
    print("=" * 150)
    print(
        f"BOOTSTRAP - {BOOTSTRAP_SAMPLES} RESAMPLES"
    )
    print("=" * 150)

    for asset, trades in (
        trades_by_asset.items()
    ):
        r = bootstrap_report(
            trades
        )

        bootstrap_rows.append({
            "asset": asset,
            **r,
        })

        print(
            f"{asset:<5} | "
            f"N {r['n']:>3} | "
            f"Exp 95% CI "
            f"[{r['exp_ci_low']:.4f}%, "
            f"{r['exp_ci_high']:.4f}%] | "
            f"P(Exp>0) "
            f"{r['prob_exp_positive']:.2f}% | "
            f"P(PF>1) "
            f"{r['prob_pf_gt_1']:.2f}% | "
            f"Median PF "
            f"{r['median_pf']:.3f}"
        )

    # ========================================================
    # LEAVE ONE YEAR OUT
    # ========================================================

    loyo_rows = []

    print()
    print("=" * 150)
    print("LEAVE-ONE-YEAR-OUT")
    print("=" * 150)

    for asset, trades in (
        trades_by_asset.items()
    ):
        report = leave_one_year_out(
            trades
        )

        report["asset"] = asset

        loyo_rows.append(
            report
        )

        print()
        print(asset)
        print("-" * 150)

        for _, row in (
            report.iterrows()
        ):
            pf_text = (
                "inf"
                if np.isinf(
                    row["pf"]
                )
                else
                f"{row['pf']:.3f}"
            )

            print(
                f"Remove "
                f"{int(row['removed_year'])} | "
                f"Trades {int(row['trades']):>3} | "
                f"PF {pf_text:>6} | "
                f"Exp {row['exp']:>8.4f}% | "
                f"PnL {row['pnl']:>8.2f}% | "
                f"DD {row['dd']:>7.2f}%"
            )

    # ========================================================
    # POOLED RESEARCH DISTRIBUTION
    # ========================================================

    pooled = pd.concat(
        list(
            trades_by_asset.values()
        ),
        ignore_index=True,
    )

    pooled = pooled.sort_values(
        "entry_time"
    )

    print()
    print("=" * 150)
    print(
        "POOLED BTC + ETH + SOL RESEARCH DISTRIBUTION"
    )
    print("=" * 150)

    metric_line(
        "POOLED EVENTS",
        pooled,
    )

    pooled_bootstrap = bootstrap_report(
        pooled
    )

    print(
        f"Pooled bootstrap | "
        f"N {pooled_bootstrap['n']} | "
        f"Exp 95% CI "
        f"[{pooled_bootstrap['exp_ci_low']:.4f}%, "
        f"{pooled_bootstrap['exp_ci_high']:.4f}%] | "
        f"P(Exp>0) "
        f"{pooled_bootstrap['prob_exp_positive']:.2f}% | "
        f"P(PF>1) "
        f"{pooled_bootstrap['prob_pf_gt_1']:.2f}% | "
        f"Median PF "
        f"{pooled_bootstrap['median_pf']:.3f}"
    )

    print()
    print(
        "ATENÇÃO: pooled events NÃO é um portfolio "
        "executável. Trades simultâneos e sizing ainda "
        "não estão modelados."
    )

    # ========================================================
    # SAVE
    # ========================================================

    summary_rows = []

    for asset, trades in (
        trades_by_asset.items()
    ):
        summary_rows.append({
            "asset": asset,
            **metrics(trades),
        })

    pd.DataFrame(
        summary_rows
    ).to_csv(
        "c2g_v116_asset_summary.csv",
        index=False,
    )

    pd.DataFrame(
        yearly_rows
    ).to_csv(
        "c2g_v116_yearly.csv",
        index=False,
    )

    pd.DataFrame(
        outlier_rows
    ).to_csv(
        "c2g_v116_top_winner_dependency.csv",
        index=False,
    )

    pd.DataFrame(
        bootstrap_rows
    ).to_csv(
        "c2g_v116_bootstrap.csv",
        index=False,
    )

    pd.concat(
        loyo_rows,
        ignore_index=True,
    ).to_csv(
        "c2g_v116_leave_one_year_out.csv",
        index=False,
    )

    pooled.to_csv(
        "c2g_v116_pooled_events.csv",
        index=False,
    )

    print()
    print("=" * 150)
    print("ARQUIVOS GERADOS")
    print("=" * 150)
    print(
        "c2g_v116_asset_summary.csv"
    )
    print(
        "c2g_v116_yearly.csv"
    )
    print(
        "c2g_v116_top_winner_dependency.csv"
    )
    print(
        "c2g_v116_bootstrap.csv"
    )
    print(
        "c2g_v116_leave_one_year_out.csv"
    )
    print(
        "c2g_v116_pooled_events.csv"
    )
    print("=" * 150)
