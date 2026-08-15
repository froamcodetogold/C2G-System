import pandas as pd
import pandas_ta as ta
import numpy as np


# ============================================================
# C2G SYSTEM PRO - V1.8 SIGNAL TIMING & THRESHOLD AUDIT
#
# Objetivo:
# Descobrir se Binance Spot e Bybit Perpetual estão gerando
# "eventos diferentes" ou os MESMOS eventos com deslocamento
# de 1-3 candles por pequenas diferenças de OHLC.
#
# NÃO otimiza a estratégia.
# NÃO altera SL/TP.
#
# Mede:
# 1) concordância exata dos sinais
# 2) concordância tolerante em +/- 1h, 2h, 3h e 6h
# 3) diferença numérica de ADX / ATR Ratio
# 4) proximidade dos sinais aos thresholds
# 5) sensibilidade de Supertrend / ADX Rising / ATR Ratio
#
# Arquivos necessários:
# - btc_data_binance_full_1h.csv
# - btc_data_bybit_perp_1h.csv
# ============================================================


BINANCE_FILE = "btc_data_binance_full_1h.csv"
BYBIT_FILE = "btc_data_bybit_perp_1h.csv"

SUPERTREND_LENGTH = 10
SUPERTREND_MULTIPLIER = 3.0

ADX_LENGTH = 14
ADX_MIN = 25.0

ATR_LENGTH = 14
ATR_MA_LENGTH = 50

EMA_LENGTH = 200
EMA_SLOPE_LOOKBACK = 50


# ============================================================
# HELPERS
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

    # ---------------- SUPERTREND ----------------
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

    df["ST_Sell_Flip"] = (
        df["Signal"] == -2
    )

    # ---------------- ADX ----------------
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

    df["ADX_Change"] = (
        df["ADX"]
        -
        df["ADX"].shift(1)
    )

    df["ADX_Rising"] = (
        df["ADX_Change"] > 0
    )

    # ---------------- ATR ----------------
    df["ATR"] = ta.atr(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        length=ATR_LENGTH,
    )

    df["ATR_MA"] = (
        df["ATR"]
        .rolling(
            ATR_MA_LENGTH
        )
        .mean()
    )

    df["ATR_Ratio"] = (
        df["ATR"]
        /
        df["ATR_MA"]
    )

    # ---------------- EMA REGIME ----------------
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

    bull = (
        (df["close"] > df["EMA200"])
        &
        (
            df["EMA200"]
            >
            df["EMA200_Lag"]
        )
    )

    bear = (
        (df["close"] < df["EMA200"])
        &
        (
            df["EMA200"]
            <
            df["EMA200_Lag"]
        )
    )

    df["Regime"] = "NEUTRAL"

    df.loc[
        bull,
        "Regime"
    ] = "BULL"

    df.loc[
        bear,
        "Regime"
    ] = "BEAR"

    # ---------------- BASE V1.2C ----------------
    df["Base_Buy"] = (
        df["ST_Buy_Flip"]
        &
        (df["ADX"] > ADX_MIN)
        &
        df["ADX_Rising"]
    )

    df["Base_Sell"] = (
        df["ST_Sell_Flip"]
        &
        (df["ADX"] > ADX_MIN)
        &
        df["ADX_Rising"]
    )

    # ---------------- V1.5A ----------------
    df["V15A_Buy"] = (
        df["Base_Buy"]
        &
        (
            df["Regime"]
            == "BULL"
        )
    )

    df["V15A_Sell"] = (
        df["Base_Sell"]
        &
        (
            df["Regime"]
            == "BEAR"
        )
    )

    # ---------------- V1.5B ----------------
    atr_ok = (
        (df["ATR_Ratio"] >= 0.80)
        &
        (df["ATR_Ratio"] < 1.20)
    )

    df["V15B_Buy"] = (
        df["V15A_Buy"]
        &
        atr_ok
    )

    df["V15B_Sell"] = (
        df["V15A_Sell"]
        &
        atr_ok
    )

    return df


def greedy_match(
    times_a,
    times_b,
    tolerance_hours,
):
    """
    Faz matching 1-para-1 pelo timestamp mais próximo dentro da tolerância.
    Retorna pares e sinais não casados.
    """

    a = sorted(
        pd.Timestamp(x)
        for x in times_a
    )

    b = sorted(
        pd.Timestamp(x)
        for x in times_b
    )

    unused_b = set(b)

    pairs = []

    tolerance = pd.Timedelta(
        hours=tolerance_hours
    )

    for ta in a:
        candidates = [
            tb
            for tb in unused_b
            if abs(tb - ta) <= tolerance
        ]

        if not candidates:
            continue

        best = min(
            candidates,
            key=lambda tb: abs(
                tb - ta
            )
        )

        unused_b.remove(
            best
        )

        pairs.append({
            "binance_time": ta,
            "bybit_time": best,
            "delta_hours": (
                best - ta
            ).total_seconds()
            / 3600.0,
            "abs_delta_hours": abs(
                (
                    best - ta
                ).total_seconds()
                / 3600.0
            ),
        })

    matched_a = {
        row["binance_time"]
        for row in pairs
    }

    matched_b = {
        row["bybit_time"]
        for row in pairs
    }

    unmatched_a = [
        ta
        for ta in a
        if ta not in matched_a
    ]

    unmatched_b = [
        tb
        for tb in b
        if tb not in matched_b
    ]

    return (
        pd.DataFrame(pairs),
        unmatched_a,
        unmatched_b,
    )


def tolerant_report(
    binance,
    bybit,
    signal_col,
):
    times_a = list(
        binance.index[
            binance[
                signal_col
            ].astype(bool)
        ]
    )

    times_b = list(
        bybit.index[
            bybit[
                signal_col
            ].astype(bool)
        ]
    )

    rows = []

    for tolerance in [
        0,
        1,
        2,
        3,
        6,
    ]:
        pairs, ua, ub = greedy_match(
            times_a,
            times_b,
            tolerance_hours=tolerance,
        )

        matched = len(
            pairs
        )

        denom = max(
            len(times_a),
            len(times_b),
            1,
        )

        coverage = (
            matched
            /
            denom
            *
            100.0
        )

        rows.append({
            "signal": signal_col,
            "tolerance_hours": tolerance,
            "binance_signals": len(
                times_a
            ),
            "bybit_signals": len(
                times_b
            ),
            "matched_pairs": matched,
            "match_coverage_pct": coverage,
            "binance_unmatched": len(
                ua
            ),
            "bybit_unmatched": len(
                ub
            ),
        })

    return pd.DataFrame(
        rows
    )


def pct_summary(series):
    s = (
        pd.Series(series)
        .dropna()
        .abs()
    )

    if s.empty:
        return {
            "median": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }

    return {
        "median": float(
            s.median()
        ),
        "p75": float(
            s.quantile(0.75)
        ),
        "p95": float(
            s.quantile(0.95)
        ),
        "max": float(
            s.max()
        ),
    }


def print_indicator_difference(
    aligned,
):
    print()
    print("=" * 128)
    print("INDICATOR DIFFERENCES - SAME CANDLE")
    print("=" * 128)

    metrics = [
        (
            "ADX",
            "binance_ADX",
            "bybit_ADX",
        ),
        (
            "ADX Change",
            "binance_ADX_Change",
            "bybit_ADX_Change",
        ),
        (
            "ATR Ratio",
            "binance_ATR_Ratio",
            "bybit_ATR_Ratio",
        ),
    ]

    rows = []

    for name, a_col, b_col in metrics:
        diff = (
            aligned[
                b_col
            ]
            -
            aligned[
                a_col
            ]
        )

        s = pct_summary(
            diff
        )

        print(
            f"{name:<14} | "
            f"Median abs diff {s['median']:.6f} | "
            f"P75 {s['p75']:.6f} | "
            f"P95 {s['p95']:.6f} | "
            f"Max {s['max']:.6f}"
        )

        rows.append({
            "indicator": name,
            **s,
        })

    return pd.DataFrame(
        rows
    )


def threshold_audit(
    df,
    market_name,
):
    """
    Mede quantos sinais BASE estão próximos de cliffs de threshold.
    A ideia é saber se pequenas diferenças de feed conseguem trocar
    facilmente TRUE/FALSE.
    """

    events = df[
        df[
            "ST_Buy_Flip"
        ]
        |
        df[
            "ST_Sell_Flip"
        ]
    ].copy()

    events = events[
        pd.notna(
            events["ADX"]
        )
    ]

    if events.empty:
        return {
            "market": market_name,
            "supertrend_flips": 0,
            "adx_24_26_pct": 0.0,
            "adx_change_abs_lt_0_25_pct": 0.0,
            "atr_ratio_near_080_pct": 0.0,
            "atr_ratio_near_120_pct": 0.0,
        }

    n = len(events)

    return {
        "market": market_name,
        "supertrend_flips": n,

        "adx_24_26_pct": (
            (
                events["ADX"]
                .between(
                    24.0,
                    26.0,
                    inclusive="both",
                )
            ).mean()
            * 100.0
        ),

        "adx_change_abs_lt_0_25_pct": (
            (
                events[
                    "ADX_Change"
                ].abs()
                < 0.25
            ).mean()
            * 100.0
        ),

        "atr_ratio_near_080_pct": (
            (
                events[
                    "ATR_Ratio"
                ].between(
                    0.75,
                    0.85,
                    inclusive="both",
                )
            ).mean()
            * 100.0
        ),

        "atr_ratio_near_120_pct": (
            (
                events[
                    "ATR_Ratio"
                ].between(
                    1.15,
                    1.25,
                    inclusive="both",
                )
            ).mean()
            * 100.0
        ),
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 128)
    print("C2G V1.8 - SIGNAL TIMING & THRESHOLD AUDIT")
    print("=" * 128)

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

    binance_calc = (
        binance_raw.loc[
            warmup_start:overlap_end
        ].copy()
    )

    bybit_calc = (
        bybit_raw.loc[
            warmup_start:overlap_end
        ].copy()
    )

    binance_calc = (
        prepare_indicators(
            binance_calc
        )
    )

    bybit_calc = (
        prepare_indicators(
            bybit_calc
        )
    )

    binance = (
        binance_calc.loc[
            overlap_start:overlap_end
        ].copy()
    )

    bybit = (
        bybit_calc.loc[
            overlap_start:overlap_end
        ].copy()
    )

    common_index = (
        binance.index
        .intersection(
            bybit.index
        )
    )

    print(
        f"Período comum: "
        f"{overlap_start} -> {overlap_end}"
    )

    print(
        f"Common candles: {len(common_index)}"
    )

    # ========================================================
    # TOLERANT SIGNAL MATCHING
    # ========================================================

    signal_columns = [
        "Base_Buy",
        "Base_Sell",
        "V15A_Buy",
        "V15A_Sell",
        "V15B_Buy",
        "V15B_Sell",
    ]

    all_match_rows = []

    print()
    print("=" * 128)
    print("TOLERANT SIGNAL MATCHING")
    print("=" * 128)

    for signal_col in signal_columns:
        report = tolerant_report(
            binance,
            bybit,
            signal_col,
        )

        all_match_rows.append(
            report
        )

        print()
        print(signal_col)
        print("-" * 128)

        for _, row in report.iterrows():
            print(
                f"+/- {int(row['tolerance_hours']):>1}h | "
                f"Binance {int(row['binance_signals']):>3} | "
                f"Bybit {int(row['bybit_signals']):>3} | "
                f"Matched {int(row['matched_pairs']):>3} | "
                f"Coverage {row['match_coverage_pct']:>6.2f}% | "
                f"Unmatched B {int(row['binance_unmatched']):>3} | "
                f"Unmatched Y {int(row['bybit_unmatched']):>3}"
            )

    match_df = pd.concat(
        all_match_rows,
        ignore_index=True,
    )

    # ========================================================
    # ALIGNED INDICATOR DIFFERENCES
    # ========================================================

    aligned = pd.DataFrame(
        index=common_index
    )

    columns_to_copy = [
        "close",
        "ADX",
        "ADX_Change",
        "ATR_Ratio",
        "Regime",
        "ST_Buy_Flip",
        "ST_Sell_Flip",
        "Base_Buy",
        "Base_Sell",
        "V15A_Buy",
        "V15A_Sell",
        "V15B_Buy",
        "V15B_Sell",
    ]

    for col in columns_to_copy:
        aligned[
            f"binance_{col}"
        ] = (
            binance.loc[
                common_index,
                col
            ]
        )

        aligned[
            f"bybit_{col}"
        ] = (
            bybit.loc[
                common_index,
                col
            ]
        )

    indicator_diff_df = (
        print_indicator_difference(
            aligned
        )
    )

    # ========================================================
    # THRESHOLD CLIFF AUDIT
    # ========================================================

    threshold_rows = [
        threshold_audit(
            binance,
            "BINANCE_SPOT",
        ),
        threshold_audit(
            bybit,
            "BYBIT_PERPETUAL",
        ),
    ]

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    print()
    print("=" * 128)
    print("THRESHOLD CLIFF AUDIT - ALL SUPERTREND FLIPS")
    print("=" * 128)

    print(
        threshold_df.to_string(
            index=False,
            formatters={
                "adx_24_26_pct": "{:.2f}%".format,
                "adx_change_abs_lt_0_25_pct": "{:.2f}%".format,
                "atr_ratio_near_080_pct": "{:.2f}%".format,
                "atr_ratio_near_120_pct": "{:.2f}%".format,
            },
        )
    )

    # ========================================================
    # +/-2H MATCH PAIRS FOR DETAILED CSV
    # ========================================================

    detailed_pairs = []

    for signal_col in signal_columns:
        times_a = list(
            binance.index[
                binance[
                    signal_col
                ].astype(bool)
            ]
        )

        times_b = list(
            bybit.index[
                bybit[
                    signal_col
                ].astype(bool)
            ]
        )

        pairs, ua, ub = greedy_match(
            times_a,
            times_b,
            tolerance_hours=2,
        )

        if not pairs.empty:
            pairs[
                "signal"
            ] = signal_col

            detailed_pairs.append(
                pairs
            )

    if detailed_pairs:
        detailed_pairs_df = (
            pd.concat(
                detailed_pairs,
                ignore_index=True,
            )
        )
    else:
        detailed_pairs_df = pd.DataFrame(
            columns=[
                "binance_time",
                "bybit_time",
                "delta_hours",
                "abs_delta_hours",
                "signal",
            ]
        )

    # ========================================================
    # SAVE
    # ========================================================

    match_df.to_csv(
        "c2g_v18_signal_tolerance_report.csv",
        index=False,
    )

    indicator_diff_df.to_csv(
        "c2g_v18_indicator_difference_report.csv",
        index=False,
    )

    threshold_df.to_csv(
        "c2g_v18_threshold_cliff_report.csv",
        index=False,
    )

    detailed_pairs_df.to_csv(
        "c2g_v18_matched_pairs_2h.csv",
        index=False,
    )

    aligned.to_csv(
        "c2g_v18_aligned_indicators.csv"
    )

    print()
    print("=" * 128)
    print("ARQUIVOS GERADOS")
    print("=" * 128)
    print(
        "c2g_v18_signal_tolerance_report.csv"
    )
    print(
        "c2g_v18_indicator_difference_report.csv"
    )
    print(
        "c2g_v18_threshold_cliff_report.csv"
    )
    print(
        "c2g_v18_matched_pairs_2h.csv"
    )
    print(
        "c2g_v18_aligned_indicators.csv"
    )
    print("=" * 128)
