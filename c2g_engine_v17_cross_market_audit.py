import pandas as pd
import pandas_ta as ta
import numpy as np


# ============================================================
# C2G SYSTEM PRO - V1.7 CROSS-MARKET CONSISTENCY AUDIT
#
# Objetivo:
# Descobrir POR QUE a estratégia que pareceu boa no
# BTC/USDT Spot da Binance falhou no BTCUSDT Perpetual da Bybit.
#
# Este script NÃO otimiza parâmetros.
#
# Ele faz 4 coisas:
#
# 1) Usa exatamente o MESMO período disponível na Bybit
#    para Binance e Bybit.
#
# 2) Recalcula os mesmos indicadores:
#    - Supertrend 10/3
#    - ADX > 25 + Rising
#    - EMA200 + slope 50
#    - ATR Ratio
#
# 3) Testa no mesmo período:
#    - V1.5A = BUY BULL / SELL BEAR
#    - V1.5B = V1.5A + ATR Ratio 0.80-1.20
#
# 4) Mede:
#    - diferença de preços
#    - concordância de regime
#    - concordância dos sinais
#    - performance por mercado
#
# Isso permite separar:
#
# A) problema de período
# B) dependência do feed/mercado
# C) sinais muito sensíveis ao OHLC
# D) diferença na execução / comportamento spot x perp
# ============================================================


BINANCE_FILE = "btc_data_binance_full_1h.csv"
BYBIT_FILE = "btc_data_bybit_perp_1h.csv"

SUPERTREND_LENGTH = 10
SUPERTREND_MULTIPLIER = 3.0

ADX_LENGTH = 14
ADX_MIN = 25.0

ATR_LENGTH = 14
STOP_ATR = 1.5
TARGET_ATR = 3.0

EMA_LENGTH = 200
EMA_SLOPE_LOOKBACK = 50

ATR_MA_LENGTH = 50


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
        str(c).strip().lower()
        for c in df.columns
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

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(
            df["volume"],
            errors="coerce",
        )

    df.dropna(
        subset=required,
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

    df["ADX_Rising"] = (
        df["ADX"]
        >
        df["ADX"].shift(1)
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
        (df["Signal"] == 2)
        &
        (df["ADX"] > ADX_MIN)
        &
        df["ADX_Rising"]
    )

    df["Base_Sell"] = (
        (df["Signal"] == -2)
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


def calculate_pnl(
    side,
    entry_price,
    exit_price,
):
    if side == "BUY":
        return (
            (
                exit_price
                -
                entry_price
            )
            /
            entry_price
        ) * 100.0

    return (
        (
            entry_price
            -
            exit_price
        )
        /
        entry_price
    ) * 100.0


def run_backtest(
    df,
    buy_col,
    sell_col,
):
    trades = []

    position = None

    entry_price = None
    entry_time = None

    stop = None
    target = None

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

        o = float(
            current["open"]
        )

        h = float(
            current["high"]
        )

        l = float(
            current["low"]
        )

        c = float(
            current["close"]
        )

        # ====================================================
        # GAP MANAGEMENT FOR EXISTING POSITION
        # ====================================================
        if position == "BUY":
            if o <= stop:
                trades.append({
                    "side": "BUY",
                    "entry_time": entry_time,
                    "exit_time": df.index[i],
                    "entry_price": entry_price,
                    "exit_price": o,
                    "reason": "STOP_GAP",
                    "pnl_pct": calculate_pnl(
                        "BUY",
                        entry_price,
                        o,
                    ),
                })

                position = None

            elif o >= target:
                trades.append({
                    "side": "BUY",
                    "entry_time": entry_time,
                    "exit_time": df.index[i],
                    "entry_price": entry_price,
                    "exit_price": o,
                    "reason": "TARGET_GAP",
                    "pnl_pct": calculate_pnl(
                        "BUY",
                        entry_price,
                        o,
                    ),
                })

                position = None

        elif position == "SELL":
            if o >= stop:
                trades.append({
                    "side": "SELL",
                    "entry_time": entry_time,
                    "exit_time": df.index[i],
                    "entry_price": entry_price,
                    "exit_price": o,
                    "reason": "STOP_GAP",
                    "pnl_pct": calculate_pnl(
                        "SELL",
                        entry_price,
                        o,
                    ),
                })

                position = None

            elif o <= target:
                trades.append({
                    "side": "SELL",
                    "entry_time": entry_time,
                    "exit_time": df.index[i],
                    "entry_price": entry_price,
                    "exit_price": o,
                    "reason": "TARGET_GAP",
                    "pnl_pct": calculate_pnl(
                        "SELL",
                        entry_price,
                        o,
                    ),
                })

                position = None

        # ====================================================
        # ENTRY AT CURRENT OPEN USING PREVIOUS CLOSED BAR
        # ====================================================
        if position is None:
            atr = previous[
                "ATR"
            ]

            if (
                pd.notna(atr)
                and atr > 0
            ):
                if bool(
                    previous[
                        buy_col
                    ]
                ):
                    position = "BUY"
                    entry_time = df.index[i]
                    entry_price = o

                    stop = (
                        entry_price
                        -
                        STOP_ATR
                        * atr
                    )

                    target = (
                        entry_price
                        +
                        TARGET_ATR
                        * atr
                    )

                elif bool(
                    previous[
                        sell_col
                    ]
                ):
                    position = "SELL"
                    entry_time = df.index[i]
                    entry_price = o

                    stop = (
                        entry_price
                        +
                        STOP_ATR
                        * atr
                    )

                    target = (
                        entry_price
                        -
                        TARGET_ATR
                        * atr
                    )

        # ====================================================
        # INTRABAR
        # ====================================================
        if position == "BUY":
            exit_price = None
            reason = None

            if l <= stop:
                exit_price = stop
                reason = "STOP"

            elif h >= target:
                exit_price = target
                reason = "TAKE_PROFIT"

            elif (
                current[
                    "Trend_Direction"
                ]
                == -1
            ):
                exit_price = c
                reason = "TREND_REVERSAL"

            if exit_price is not None:
                trades.append({
                    "side": "BUY",
                    "entry_time": entry_time,
                    "exit_time": df.index[i],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "reason": reason,
                    "pnl_pct": calculate_pnl(
                        "BUY",
                        entry_price,
                        exit_price,
                    ),
                })

                position = None

        elif position == "SELL":
            exit_price = None
            reason = None

            if h >= stop:
                exit_price = stop
                reason = "STOP"

            elif l <= target:
                exit_price = target
                reason = "TAKE_PROFIT"

            elif (
                current[
                    "Trend_Direction"
                ]
                == 1
            ):
                exit_price = c
                reason = "TREND_REVERSAL"

            if exit_price is not None:
                trades.append({
                    "side": "SELL",
                    "entry_time": entry_time,
                    "exit_time": df.index[i],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "reason": reason,
                    "pnl_pct": calculate_pnl(
                        "SELL",
                        entry_price,
                        exit_price,
                    ),
                })

                position = None

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


def calculate_metrics(trades):
    if trades.empty:
        return {
            "trades": 0,
            "wr": 0.0,
            "pnl": 0.0,
            "pf": 0.0,
            "exp": 0.0,
            "dd": 0.0,
        }

    trades = trades.sort_values(
        "entry_time"
    )

    wins = trades[
        trades["pnl_pct"] > 0
    ]

    losses = trades[
        trades["pnl_pct"] < 0
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

    for pnl in trades[
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
        equity
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
        "trades": len(trades),
        "wr": (
            len(wins)
            /
            len(trades)
            *
            100.0
        ),
        "pnl": float(
            trades["pnl_pct"].sum()
        ),
        "pf": pf,
        "exp": float(
            trades["pnl_pct"].mean()
        ),
        "dd": float(
            dd.min()
        ),
    }


def print_metric(
    label,
    trades,
):
    m = calculate_metrics(
        trades
    )

    pf = (
        "inf"
        if np.isinf(
            m["pf"]
        )
        else f"{m['pf']:.3f}"
    )

    print(
        f"{label:<30} | "
        f"Trades {m['trades']:>3} | "
        f"WR {m['wr']:>6.2f}% | "
        f"PnL {m['pnl']:>8.2f}% | "
        f"PF {pf:>6} | "
        f"Exp {m['exp']:>8.4f}% | "
        f"DD {m['dd']:>7.2f}%"
    )

    return m


def signal_set(df, col):
    return set(
        df.index[
            df[col].astype(bool)
        ]
    )


def agreement_stats(
    binance,
    bybit,
    col,
):
    a = signal_set(
        binance,
        col
    )

    b = signal_set(
        bybit,
        col
    )

    intersection = (
        a.intersection(b)
    )

    union = (
        a.union(b)
    )

    jaccard = (
        len(intersection)
        /
        len(union)
        *
        100.0
        if union
        else 100.0
    )

    return {
        "column": col,
        "binance_signals": len(a),
        "bybit_signals": len(b),
        "same_timestamp": len(
            intersection
        ),
        "union": len(union),
        "jaccard_pct": jaccard,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 126)
    print("C2G V1.7 - CROSS-MARKET CONSISTENCY AUDIT")
    print("=" * 126)

    binance_raw = load_ohlcv(
        BINANCE_FILE
    )

    bybit_raw = load_ohlcv(
        BYBIT_FILE
    )

    # Usa exatamente o período disponível na Bybit.
    overlap_start = bybit_raw.index.min()
    overlap_end = bybit_raw.index.max()

    # Adiciona warmup anterior à data de início quando possível,
    # para EMA/ATR/ADX começarem corretamente.
    warmup_hours = 1000

    warmup_start = (
        overlap_start
        -
        pd.Timedelta(
            hours=warmup_hours
        )
    )

    binance_calc = binance_raw.loc[
        (
            binance_raw.index
            >= warmup_start
        )
        &
        (
            binance_raw.index
            <= overlap_end
        )
    ].copy()

    bybit_calc = bybit_raw.loc[
        (
            bybit_raw.index
            >= warmup_start
        )
        &
        (
            bybit_raw.index
            <= overlap_end
        )
    ].copy()

    binance_calc = prepare_indicators(
        binance_calc
    )

    bybit_calc = prepare_indicators(
        bybit_calc
    )

    # Agora corta exatamente para o mesmo range operacional.
    binance = binance_calc.loc[
        overlap_start:overlap_end
    ].copy()

    bybit = bybit_calc.loc[
        overlap_start:overlap_end
    ].copy()

    print(
        f"Período comum: "
        f"{overlap_start} -> {overlap_end}"
    )

    print(
        f"Binance candles no período: "
        f"{len(binance)}"
    )

    print(
        f"Bybit candles no período:   "
        f"{len(bybit)}"
    )

    # ========================================================
    # PREÇO / OHLC AGREEMENT
    # ========================================================

    common_index = (
        binance.index
        .intersection(
            bybit.index
        )
    )

    b_close = binance.loc[
        common_index,
        "close"
    ]

    y_close = bybit.loc[
        common_index,
        "close"
    ]

    close_diff_pct = (
        (
            y_close
            -
            b_close
        )
        /
        b_close
        *
        100.0
    )

    abs_close_diff = (
        close_diff_pct.abs()
    )

    print()
    print("=" * 126)
    print("PRICE FEED DIFFERENCE - SAME TIMESTAMPS")
    print("=" * 126)

    print(
        f"Common candles:              "
        f"{len(common_index)}"
    )

    print(
        f"Mean Bybit-vs-Binance close: "
        f"{close_diff_pct.mean():.4f}%"
    )

    print(
        f"Median absolute difference:  "
        f"{abs_close_diff.median():.4f}%"
    )

    print(
        f"95th pct absolute difference:"
        f" {abs_close_diff.quantile(0.95):.4f}%"
    )

    print(
        f"Max absolute difference:     "
        f"{abs_close_diff.max():.4f}%"
    )

    # ========================================================
    # REGIME AGREEMENT
    # ========================================================

    regime_compare = pd.DataFrame({
        "binance_regime": (
            binance.loc[
                common_index,
                "Regime"
            ]
        ),
        "bybit_regime": (
            bybit.loc[
                common_index,
                "Regime"
            ]
        ),
    })

    regime_agreement = (
        regime_compare[
            "binance_regime"
        ]
        ==
        regime_compare[
            "bybit_regime"
        ]
    ).mean() * 100.0

    print()
    print("=" * 126)
    print("REGIME AGREEMENT")
    print("=" * 126)
    print(
        f"Mesmo regime no mesmo candle: "
        f"{regime_agreement:.2f}%"
    )

    regime_matrix = pd.crosstab(
        regime_compare[
            "binance_regime"
        ],
        regime_compare[
            "bybit_regime"
        ],
    )

    print()
    print(regime_matrix)

    # ========================================================
    # SIGNAL AGREEMENT
    # ========================================================

    signal_columns = [
        "Base_Buy",
        "Base_Sell",
        "V15A_Buy",
        "V15A_Sell",
        "V15B_Buy",
        "V15B_Sell",
    ]

    signal_rows = []

    print()
    print("=" * 126)
    print("SIGNAL AGREEMENT")
    print("=" * 126)

    for col in signal_columns:
        row = agreement_stats(
            binance,
            bybit,
            col,
        )

        signal_rows.append(
            row
        )

        print(
            f"{col:<12} | "
            f"Binance {row['binance_signals']:>3} | "
            f"Bybit {row['bybit_signals']:>3} | "
            f"Mesmo timestamp {row['same_timestamp']:>3} | "
            f"Jaccard {row['jaccard_pct']:>6.2f}%"
        )

    # ========================================================
    # APPLES-TO-APPLES BACKTEST
    # ========================================================

    results = []

    experiments = [
        (
            "V1.5A",
            "V15A_Buy",
            "V15A_Sell",
        ),
        (
            "V1.5B",
            "V15B_Buy",
            "V15B_Sell",
        ),
    ]

    for name, buy_col, sell_col in experiments:

        binance_trades = run_backtest(
            binance,
            buy_col,
            sell_col,
        )

        bybit_trades = run_backtest(
            bybit,
            buy_col,
            sell_col,
        )

        print()
        print("=" * 126)
        print(
            f"{name} - SAME PERIOD: BINANCE SPOT vs BYBIT PERPETUAL"
        )
        print("=" * 126)

        m_binance = print_metric(
            "BINANCE SPOT",
            binance_trades,
        )

        m_bybit = print_metric(
            "BYBIT PERPETUAL",
            bybit_trades,
        )

        results.append({
            "strategy": name,
            "market": "BINANCE_SPOT",
            **m_binance,
        })

        results.append({
            "strategy": name,
            "market": "BYBIT_PERPETUAL",
            **m_bybit,
        })

        print()
        print("BINANCE BY YEAR")
        print("-" * 126)

        for year, group in binance_trades.groupby(
            "year"
        ):
            print_metric(
                str(year),
                group,
            )

        print()
        print("BYBIT BY YEAR")
        print("-" * 126)

        for year, group in bybit_trades.groupby(
            "year"
        ):
            print_metric(
                str(year),
                group,
            )

        binance_trades.to_csv(
            f"c2g_v17_{name.lower()}_binance_same_period.csv",
            index=False,
        )

        bybit_trades.to_csv(
            f"c2g_v17_{name.lower()}_bybit_same_period.csv",
            index=False,
        )

    # ========================================================
    # SAVE DIAGNOSTICS
    # ========================================================

    aligned = pd.DataFrame(
        index=common_index
    )

    aligned[
        "binance_close"
    ] = b_close

    aligned[
        "bybit_close"
    ] = y_close

    aligned[
        "close_diff_pct"
    ] = close_diff_pct

    aligned[
        "binance_regime"
    ] = binance.loc[
        common_index,
        "Regime"
    ]

    aligned[
        "bybit_regime"
    ] = bybit.loc[
        common_index,
        "Regime"
    ]

    for col in signal_columns:
        aligned[
            f"binance_{col}"
        ] = (
            binance.loc[
                common_index,
                col
            ]
            .astype(int)
        )

        aligned[
            f"bybit_{col}"
        ] = (
            bybit.loc[
                common_index,
                col
            ]
            .astype(int)
        )

    aligned.to_csv(
        "c2g_v17_aligned_market_comparison.csv"
    )

    pd.DataFrame(
        signal_rows
    ).to_csv(
        "c2g_v17_signal_agreement.csv",
        index=False,
    )

    pd.DataFrame(
        results
    ).to_csv(
        "c2g_v17_same_period_results.csv",
        index=False,
    )

    print()
    print("=" * 126)
    print("C2G V1.7 - DIAGNOSTIC FILES")
    print("=" * 126)
    print(
        "c2g_v17_aligned_market_comparison.csv"
    )
    print(
        "c2g_v17_signal_agreement.csv"
    )
    print(
        "c2g_v17_same_period_results.csv"
    )
    print("=" * 126)
