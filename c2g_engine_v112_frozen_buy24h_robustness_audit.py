import pandas as pd
import pandas_ta as ta
import numpy as np


# ============================================================
# C2G SYSTEM PRO - V1.12 FROZEN BUY24H ROBUSTNESS AUDIT
#
# CANDIDATA CONGELADA:
#
# ENTRADA BUY:
# - Supertrend 10 / 3 flip BUY
# - ADX > 25
# - ADX atual > ADX anterior
# - Regime BULL:
#     Close > EMA200
#     EMA200 atual > EMA200 de 50 candles atrás
#
# SAÍDA:
# - fechar exatamente após 24 candles de 1h
# - sem SELL
# - sem SL / TP nesta versão
#
# Esta V1.12 NÃO procura novos parâmetros.
#
# Objetivos:
# 1) Repetir a candidata nos dois mercados no mesmo período.
# 2) Rodar Binance no histórico completo apenas como contexto.
# 3) Mostrar resultado por ano.
# 4) Medir dependência de poucos trades vencedores.
# 5) Fazer bootstrap dos retornos.
# 6) Fazer leave-one-year-out.
# 7) Comparar sinais Binance x Bybit casados em +/-1h.
#
# IMPORTANTE:
# A regra TIME24H foi descoberta olhando estes mesmos dados.
# Portanto isto é um ROBUSTNESS AUDIT, não um OOS puro.
# ============================================================


BINANCE_FILE = "btc_data_binance_full_1h.csv"
BYBIT_FILE = "btc_data_bybit_perp_1h.csv"

SUPERTREND_LENGTH = 10
SUPERTREND_MULTIPLIER = 3.0

ADX_LENGTH = 14
ADX_MIN = 25.0

EMA_LENGTH = 200
EMA_SLOPE_LOOKBACK = 50

TIME_EXIT_BARS = 24

# Mesmo stress usado anteriormente.
STRESS_FEE_PCT_PER_SIDE = 0.055
STRESS_SLIPPAGE_PCT_PER_SIDE = 0.020

BOOTSTRAP_SAMPLES = 20000
BOOTSTRAP_SEED = 42


# ============================================================
# DATA / INDICATORS
# ============================================================

def find_column(columns, prefix):
    matches = [
        col for col in columns
        if str(col).startswith(prefix)
    ]

    if not matches:
        raise KeyError(
            f"Nenhuma coluna começando com '{prefix}'. "
            f"Disponíveis: {list(columns)}"
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
        str(col).strip().lower()
        for col in df.columns
    ]

    required = [
        "open",
        "high",
        "low",
        "close",
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"{path}: coluna '{col}' ausente."
            )

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df.dropna(
        subset=required,
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


# ============================================================
# FROZEN BUY -> TIME 24H BACKTEST
# ============================================================

def calc_return(
    entry_price,
    exit_price,
    fee_pct=0.0,
    slippage_pct=0.0,
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

    round_trip_cost = (
        fee_pct * 2.0
        +
        slippage_pct * 2.0
    )

    return gross - round_trip_cost


def run_time24_backtest(
    df,
    fee_pct=0.0,
    slippage_pct=0.0,
):
    trades = []

    position = False
    entry_price = None
    entry_time = None
    bars_held = 0

    for i in range(
        1,
        len(df)
    ):
        previous = df.iloc[
            i - 1
        ]

        current = df.iloc[
            i
        ]

        # Entrada no OPEN do candle seguinte ao sinal.
        if (
            not position
            and
            bool(
                previous[
                    "Buy_Signal"
                ]
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

        # Fecha no CLOSE do 24º candle incluindo o candle de entrada.
        if bars_held >= TIME_EXIT_BARS:
            exit_price = float(
                current["close"]
            )

            pnl = calc_return(
                entry_price,
                exit_price,
                fee_pct,
                slippage_pct,
            )

            trades.append({
                "entry_time": entry_time,
                "exit_time": df.index[i],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "bars_held": bars_held,
                "pnl_pct": pnl,
            })

            position = False

    trades = pd.DataFrame(
        trades
    )

    if not trades.empty:
        trades[
            "entry_time"
        ] = pd.to_datetime(
            trades["entry_time"]
        )

        trades[
            "exit_time"
        ] = pd.to_datetime(
            trades["exit_time"]
        )

        trades["year"] = (
            trades[
                "entry_time"
            ].dt.year
        )

    return trades


# ============================================================
# METRICS
# ============================================================

def max_loss_streak(values):
    best = 0
    current = 0

    for value in values:
        if value < 0:
            current += 1
            best = max(
                best,
                current,
            )
        else:
            current = 0

    return best


def metrics(trades):
    if trades.empty:
        return {
            "trades": 0,
            "wins": 0,
            "win_rate": 0.0,
            "pnl": 0.0,
            "pf": 0.0,
            "expectancy": 0.0,
            "median_trade": 0.0,
            "dd": 0.0,
            "loss_streak": 0,
            "ending_equity": 100.0,
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
        "trades": len(
            ordered
        ),
        "wins": len(
            wins
        ),
        "win_rate": (
            len(wins)
            /
            len(ordered)
            *
            100.0
        ),
        "pnl": float(
            ordered[
                "pnl_pct"
            ].sum()
        ),
        "pf": pf,
        "expectancy": float(
            ordered[
                "pnl_pct"
            ].mean()
        ),
        "median_trade": float(
            ordered[
                "pnl_pct"
            ].median()
        ),
        "dd": float(
            dd.min()
        ),
        "loss_streak": (
            max_loss_streak(
                ordered[
                    "pnl_pct"
                ].tolist()
            )
        ),
        "ending_equity": float(
            equity.iloc[-1]
        ),
    }


def metric_line(label, trades):
    m = metrics(
        trades
    )

    pf_text = (
        "inf"
        if np.isinf(
            m["pf"]
        )
        else f"{m['pf']:.3f}"
    )

    print(
        f"{label:<30} | "
        f"Trades {m['trades']:>3} | "
        f"WR {m['win_rate']:>6.2f}% | "
        f"PnL {m['pnl']:>8.2f}% | "
        f"PF {pf_text:>6} | "
        f"Exp {m['expectancy']:>8.4f}% | "
        f"Median {m['median_trade']:>8.4f}% | "
        f"DD {m['dd']:>7.2f}% | "
        f"LStreak {m['loss_streak']:>2}"
    )

    return m


# ============================================================
# ROBUSTNESS CHECKS
# ============================================================

def remove_top_winners(
    trades,
    n_remove,
):
    if (
        trades.empty
        or
        n_remove <= 0
    ):
        return trades.copy()

    ordered = trades.sort_values(
        "pnl_pct",
        ascending=False,
    )

    return ordered.iloc[
        n_remove:
    ].copy()


def bootstrap_report(
    trades,
    samples=BOOTSTRAP_SAMPLES,
    seed=BOOTSTRAP_SEED,
):
    values = trades[
        "pnl_pct"
    ].dropna().to_numpy(
        dtype=float
    )

    if len(values) == 0:
        return {}

    rng = np.random.default_rng(
        seed
    )

    n = len(
        values
    )

    means = np.empty(
        samples,
        dtype=float,
    )

    pfs = np.empty(
        samples,
        dtype=float,
    )

    terminal = np.empty(
        samples,
        dtype=float,
    )

    for i in range(
        samples
    ):
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

        terminal[i] = (
            np.prod(
                1.0
                +
                sample / 100.0
            )
            *
            100.0
        )

    finite_pfs = pfs[
        np.isfinite(
            pfs
        )
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
                    finite_pfs
                )
            )
            if len(
                finite_pfs
            )
            else np.inf
        ),

        "prob_terminal_gt_100": float(
            (
                terminal > 100.0
            ).mean()
            * 100.0
        ),

        "terminal_ci_low": float(
            np.quantile(
                terminal,
                0.025,
            )
        ),

        "terminal_ci_high": float(
            np.quantile(
                terminal,
                0.975,
            )
        ),
    }


def leave_one_year_out(
    trades,
):
    rows = []

    if trades.empty:
        return pd.DataFrame()

    years = sorted(
        trades["year"].unique()
    )

    for year in years:
        subset = trades[
            trades["year"]
            != year
        ].copy()

        m = metrics(
            subset
        )

        rows.append({
            "removed_year": int(
                year
            ),
            **m,
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# MATCHED BINANCE / BYBIT SIGNAL AUDIT
# ============================================================

def extract_signal_events(
    df,
    market_name,
):
    rows = []

    for i in range(
        0,
        len(df) - TIME_EXIT_BARS
    ):
        if not bool(
            df[
                "Buy_Signal"
            ].iloc[i]
        ):
            continue

        entry_i = i + 1

        exit_i = (
            entry_i
            +
            TIME_EXIT_BARS
            -
            1
        )

        if exit_i >= len(
            df
        ):
            continue

        entry_price = float(
            df["open"].iloc[
                entry_i
            ]
        )

        exit_price = float(
            df["close"].iloc[
                exit_i
            ]
        )

        rows.append({
            "market": market_name,
            "signal_time": df.index[i],
            "entry_time": df.index[
                entry_i
            ],
            "exit_time": df.index[
                exit_i
            ],
            "gross_ret_24h_pct": (
                (
                    exit_price
                    -
                    entry_price
                )
                /
                entry_price
                *
                100.0
            ),
        })

    return pd.DataFrame(
        rows
    )


def greedy_match_events(
    binance_events,
    bybit_events,
    tolerance_hours=1,
):
    tolerance = pd.Timedelta(
        hours=tolerance_hours
    )

    b_rows = (
        binance_events
        .sort_values(
            "signal_time"
        )
        .to_dict(
            "records"
        )
    )

    y_rows = (
        bybit_events
        .sort_values(
            "signal_time"
        )
        .to_dict(
            "records"
        )
    )

    unused = set(
        range(
            len(
                y_rows
            )
        )
    )

    pairs = []

    for b in b_rows:
        candidates = []

        for j in unused:
            y = y_rows[j]

            delta = (
                y[
                    "signal_time"
                ]
                -
                b[
                    "signal_time"
                ]
            )

            if abs(
                delta
            ) <= tolerance:
                candidates.append(
                    (
                        j,
                        abs(
                            delta
                        ),
                    )
                )

        if not candidates:
            continue

        best_j = min(
            candidates,
            key=lambda x: x[1],
        )[0]

        unused.remove(
            best_j
        )

        y = y_rows[
            best_j
        ]

        pairs.append({
            "binance_signal_time": (
                b[
                    "signal_time"
                ]
            ),
            "bybit_signal_time": (
                y[
                    "signal_time"
                ]
            ),
            "delta_hours": (
                (
                    y[
                        "signal_time"
                    ]
                    -
                    b[
                        "signal_time"
                    ]
                ).total_seconds()
                /
                3600.0
            ),
            "binance_ret_24h_pct": (
                b[
                    "gross_ret_24h_pct"
                ]
            ),
            "bybit_ret_24h_pct": (
                y[
                    "gross_ret_24h_pct"
                ]
            ),
        })

    pairs = pd.DataFrame(
        pairs
    )

    if not pairs.empty:
        pairs[
            "same_return_sign"
        ] = (
            np.sign(
                pairs[
                    "binance_ret_24h_pct"
                ]
            )
            ==
            np.sign(
                pairs[
                    "bybit_ret_24h_pct"
                ]
            )
        )

        pairs[
            "abs_return_diff_pct"
        ] = (
            pairs[
                "binance_ret_24h_pct"
            ]
            -
            pairs[
                "bybit_ret_24h_pct"
            ]
        ).abs()

    return pairs


# ============================================================
# REPORTS
# ============================================================

def print_year_report(
    label,
    trades,
):
    print()
    print(
        f"{label} - RESULT BY YEAR"
    )
    print(
        "-" * 142
    )

    rows = []

    for year, group in trades.groupby(
        "year"
    ):
        m = metric_line(
            str(
                int(
                    year
                )
            ),
            group,
        )

        rows.append({
            "year": int(
                year
            ),
            **m,
        })

    return pd.DataFrame(
        rows
    )


def print_outlier_report(
    label,
    trades,
):
    print()
    print(
        f"{label} - TOP WINNER DEPENDENCY"
    )
    print(
        "-" * 142
    )

    rows = []

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

        row_label = (
            "ORIGINAL"
            if n_remove == 0
            else
            f"REMOVE TOP {n_remove}"
        )

        m = metric_line(
            row_label,
            subset,
        )

        rows.append({
            "removed_top_winners": (
                n_remove
            ),
            **m,
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 142)
    print(
        "C2G V1.12 - FROZEN BUY24H ROBUSTNESS AUDIT"
    )
    print("=" * 142)

    binance_raw = load_ohlcv(
        BINANCE_FILE
    )

    bybit_raw = load_ohlcv(
        BYBIT_FILE
    )

    overlap_start = (
        bybit_raw.index.min()
    )

    overlap_end = (
        bybit_raw.index.max()
    )

    warmup_start = (
        overlap_start
        -
        pd.Timedelta(
            hours=1000
        )
    )

    # Mesmo período.
    binance_same_calc = (
        prepare_indicators(
            binance_raw.loc[
                warmup_start:
                overlap_end
            ].copy()
        )
    )

    bybit_same_calc = (
        prepare_indicators(
            bybit_raw.loc[
                warmup_start:
                overlap_end
            ].copy()
        )
    )

    binance_same = (
        binance_same_calc.loc[
            overlap_start:
            overlap_end
        ].copy()
    )

    bybit_same = (
        bybit_same_calc.loc[
            overlap_start:
            overlap_end
        ].copy()
    )

    # Histórico completo Binance como CONTEXTO,
    # não como novo OOS.
    binance_full = (
        prepare_indicators(
            binance_raw.copy()
        )
    )

    print(
        f"Same period: "
        f"{overlap_start} -> {overlap_end}"
    )

    print(
        f"Binance same candles: "
        f"{len(binance_same)}"
    )

    print(
        f"Bybit same candles:   "
        f"{len(bybit_same)}"
    )

    print(
        f"Binance full candles: "
        f"{len(binance_full)}"
    )

    # ========================================================
    # GROSS
    # ========================================================

    b_same_gross = run_time24_backtest(
        binance_same
    )

    y_same_gross = run_time24_backtest(
        bybit_same
    )

    b_full_gross = run_time24_backtest(
        binance_full
    )

    print()
    print("=" * 142)
    print("FROZEN CANDIDATE - GROSS")
    print("=" * 142)

    metric_line(
        "BINANCE SAME PERIOD",
        b_same_gross,
    )

    metric_line(
        "BYBIT SAME PERIOD",
        y_same_gross,
    )

    metric_line(
        "BINANCE FULL CONTEXT",
        b_full_gross,
    )

    # ========================================================
    # COST STRESS
    # ========================================================

    b_same_cost = run_time24_backtest(
        binance_same,
        fee_pct=(
            STRESS_FEE_PCT_PER_SIDE
        ),
        slippage_pct=(
            STRESS_SLIPPAGE_PCT_PER_SIDE
        ),
    )

    y_same_cost = run_time24_backtest(
        bybit_same,
        fee_pct=(
            STRESS_FEE_PCT_PER_SIDE
        ),
        slippage_pct=(
            STRESS_SLIPPAGE_PCT_PER_SIDE
        ),
    )

    b_full_cost = run_time24_backtest(
        binance_full,
        fee_pct=(
            STRESS_FEE_PCT_PER_SIDE
        ),
        slippage_pct=(
            STRESS_SLIPPAGE_PCT_PER_SIDE
        ),
    )

    print()
    print("=" * 142)
    print("FROZEN CANDIDATE - COST STRESS")
    print("=" * 142)

    metric_line(
        "BINANCE SAME PERIOD",
        b_same_cost,
    )

    metric_line(
        "BYBIT SAME PERIOD",
        y_same_cost,
    )

    metric_line(
        "BINANCE FULL CONTEXT",
        b_full_cost,
    )

    # ========================================================
    # YEARLY
    # ========================================================

    b_year = print_year_report(
        "BINANCE SAME GROSS",
        b_same_gross,
    )

    y_year = print_year_report(
        "BYBIT SAME GROSS",
        y_same_gross,
    )

    full_year = print_year_report(
        "BINANCE FULL GROSS",
        b_full_gross,
    )

    # ========================================================
    # TOP WINNER DEPENDENCY
    # ========================================================

    b_outlier = print_outlier_report(
        "BINANCE SAME GROSS",
        b_same_gross,
    )

    y_outlier = print_outlier_report(
        "BYBIT SAME GROSS",
        y_same_gross,
    )

    # ========================================================
    # BOOTSTRAP
    # ========================================================

    print()
    print("=" * 142)
    print(
        f"BOOTSTRAP - {BOOTSTRAP_SAMPLES} RESAMPLES"
    )
    print("=" * 142)

    bootstrap_rows = []

    for label, trades in [
        (
            "BINANCE SAME GROSS",
            b_same_gross,
        ),
        (
            "BYBIT SAME GROSS",
            y_same_gross,
        ),
        (
            "BINANCE SAME COST",
            b_same_cost,
        ),
        (
            "BYBIT SAME COST",
            y_same_cost,
        ),
    ]:
        r = bootstrap_report(
            trades
        )

        bootstrap_rows.append({
            "sample": label,
            **r,
        })

        print(
            f"{label:<22} | "
            f"N {r['n']:>3} | "
            f"Exp 95% CI "
            f"[{r['exp_ci_low']:.4f}%, "
            f"{r['exp_ci_high']:.4f}%] | "
            f"P(Exp>0) {r['prob_exp_positive']:.2f}% | "
            f"P(PF>1) {r['prob_pf_gt_1']:.2f}% | "
            f"Median PF {r['median_pf']:.3f} | "
            f"P(Equity>100) "
            f"{r['prob_terminal_gt_100']:.2f}%"
        )

    bootstrap_df = pd.DataFrame(
        bootstrap_rows
    )

    # ========================================================
    # LEAVE ONE YEAR OUT
    # ========================================================

    print()
    print("=" * 142)
    print("LEAVE-ONE-YEAR-OUT")
    print("=" * 142)

    loyo_rows = []

    for label, trades in [
        (
            "BINANCE_SAME",
            b_same_gross,
        ),
        (
            "BYBIT_SAME",
            y_same_gross,
        ),
    ]:
        report = leave_one_year_out(
            trades
        )

        if report.empty:
            continue

        report[
            "market"
        ] = label

        loyo_rows.append(
            report
        )

        print()
        print(label)
        print(
            "-" * 142
        )

        for _, row in report.iterrows():
            pf_text = (
                "inf"
                if np.isinf(
                    row["pf"]
                )
                else
                f"{row['pf']:.3f}"
            )

            print(
                f"Remove {int(row['removed_year'])} | "
                f"Trades {int(row['trades']):>3} | "
                f"PF {pf_text:>6} | "
                f"Exp {row['expectancy']:>8.4f}% | "
                f"PnL {row['pnl']:>8.2f}% | "
                f"DD {row['dd']:>7.2f}%"
            )

    loyo_df = (
        pd.concat(
            loyo_rows,
            ignore_index=True,
        )
        if loyo_rows
        else pd.DataFrame()
    )

    # ========================================================
    # MATCHED SIGNALS
    # ========================================================

    b_events = extract_signal_events(
        binance_same,
        "BINANCE_SPOT",
    )

    y_events = extract_signal_events(
        bybit_same,
        "BYBIT_PERPETUAL",
    )

    matched = greedy_match_events(
        b_events,
        y_events,
        tolerance_hours=1,
    )

    print()
    print("=" * 142)
    print("MATCHED BUY SIGNALS +/- 1H")
    print("=" * 142)

    print(
        f"Binance signals available: "
        f"{len(b_events)}"
    )

    print(
        f"Bybit signals available:   "
        f"{len(y_events)}"
    )

    print(
        f"Matched pairs:              "
        f"{len(matched)}"
    )

    if not matched.empty:
        same_sign_pct = (
            matched[
                "same_return_sign"
            ].mean()
            * 100.0
        )

        mean_abs_diff = float(
            matched[
                "abs_return_diff_pct"
            ].mean()
        )

        median_abs_diff = float(
            matched[
                "abs_return_diff_pct"
            ].median()
        )

        correlation = (
            matched[
                [
                    "binance_ret_24h_pct",
                    "bybit_ret_24h_pct",
                ]
            ]
            .corr()
            .iloc[
                0,
                1
            ]
        )

        print(
            f"Same 24h return sign:       "
            f"{same_sign_pct:.2f}%"
        )

        print(
            f"Mean abs return difference: "
            f"{mean_abs_diff:.4f}%"
        )

        print(
            f"Median abs return diff:      "
            f"{median_abs_diff:.4f}%"
        )

        print(
            f"24h return correlation:     "
            f"{correlation:.4f}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    b_same_gross.to_csv(
        "c2g_v112_binance_same_gross_trades.csv",
        index=False,
    )

    y_same_gross.to_csv(
        "c2g_v112_bybit_same_gross_trades.csv",
        index=False,
    )

    b_same_cost.to_csv(
        "c2g_v112_binance_same_cost_trades.csv",
        index=False,
    )

    y_same_cost.to_csv(
        "c2g_v112_bybit_same_cost_trades.csv",
        index=False,
    )

    b_year.to_csv(
        "c2g_v112_binance_same_yearly.csv",
        index=False,
    )

    y_year.to_csv(
        "c2g_v112_bybit_same_yearly.csv",
        index=False,
    )

    full_year.to_csv(
        "c2g_v112_binance_full_yearly.csv",
        index=False,
    )

    b_outlier.to_csv(
        "c2g_v112_binance_outlier_test.csv",
        index=False,
    )

    y_outlier.to_csv(
        "c2g_v112_bybit_outlier_test.csv",
        index=False,
    )

    bootstrap_df.to_csv(
        "c2g_v112_bootstrap.csv",
        index=False,
    )

    loyo_df.to_csv(
        "c2g_v112_leave_one_year_out.csv",
        index=False,
    )

    matched.to_csv(
        "c2g_v112_matched_signal_returns.csv",
        index=False,
    )

    print()
    print("=" * 142)
    print("ARQUIVOS GERADOS")
    print("=" * 142)
    print(
        "c2g_v112_binance_same_gross_trades.csv"
    )
    print(
        "c2g_v112_bybit_same_gross_trades.csv"
    )
    print(
        "c2g_v112_binance_same_cost_trades.csv"
    )
    print(
        "c2g_v112_bybit_same_cost_trades.csv"
    )
    print(
        "c2g_v112_binance_same_yearly.csv"
    )
    print(
        "c2g_v112_bybit_same_yearly.csv"
    )
    print(
        "c2g_v112_binance_full_yearly.csv"
    )
    print(
        "c2g_v112_binance_outlier_test.csv"
    )
    print(
        "c2g_v112_bybit_outlier_test.csv"
    )
    print(
        "c2g_v112_bootstrap.csv"
    )
    print(
        "c2g_v112_leave_one_year_out.csv"
    )
    print(
        "c2g_v112_matched_signal_returns.csv"
    )

    print()
    print("=" * 142)
    print("REGRAS CONGELADAS")
    print("=" * 142)
    print(
        "BUY only | Supertrend 10/3 | ADX > 25 + 1-bar Rising | "
        "BULL regime | Entry next open | Exit after 24 x 1H candles"
    )
    print(
        "Nenhum SL/TP e nenhum SELL nesta fase."
    )
    print(
        "A regra TIME24H já foi escolhida olhando estes dados; "
        "logo esta V1.12 mede robustez, não prova OOS."
    )
    print("=" * 142)
