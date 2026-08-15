import pandas as pd
import pandas_ta as ta
import numpy as np


# ============================================================
# C2G SYSTEM PRO - V1.10 ENTRY EDGE DIAGNOSTIC
#
# Objetivo:
# Antes de mexer novamente em filtros, descobrir se o problema
# está na ENTRADA ou na SAÍDA.
#
# Este script NÃO faz otimização de parâmetros de entrada.
#
# Ele compara duas lógicas já existentes:
#
# V1.5A_CURRENT
#   Supertrend flip
#   ADX > 25
#   ADX atual > ADX anterior
#   BUY apenas em BULL
#   SELL apenas em BEAR
#
# V1.9B_ROBUST
#   confirmação no candle do flip OU no próximo candle
#   ADX > 25
#   ADX atual > ADX de 3 candles atrás
#   BUY apenas em BULL
#   SELL apenas em BEAR
#
# Mercados:
# - Binance BTC/USDT Spot
# - Bybit BTCUSDT Perpetual
#
# Mesmo período:
# usa exatamente o período disponível na Bybit.
#
# Para cada sinal, a entrada teórica ocorre no OPEN do candle
# seguinte, exatamente como no backtest.
#
# Depois mede:
# - retorno direcional em 6h / 12h / 24h / 48h / 72h
# - MFE (máxima excursão favorável)
# - MAE (máxima excursão adversa)
# - MFE e MAE normalizados por ATR
# - % dos sinais que alcançam 1 / 1.5 / 2 / 3 ATR a favor
#
# IMPORTANTE:
# Este é um EVENT STUDY, não um backtest executável.
# O objetivo é descobrir se existe edge direcional na entrada.
# ============================================================


BINANCE_FILE = "btc_data_binance_full_1h.csv"
BYBIT_FILE = "btc_data_bybit_perp_1h.csv"

SUPERTREND_LENGTH = 10
SUPERTREND_MULTIPLIER = 3.0

ADX_LENGTH = 14
ADX_MIN = 25.0

ATR_LENGTH = 14

EMA_LENGTH = 200
EMA_SLOPE_LOOKBACK = 50

HORIZONS = [6, 12, 24, 48, 72]


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

    df["ADX_Diff1"] = (
        df["ADX"]
        -
        df["ADX"].shift(1)
    )

    df["ADX_Rising_1"] = (
        df["ADX_Diff1"] > 0
    )

    df["ADX_Slope_3"] = (
        df["ADX"]
        >
        df["ADX"].shift(3)
    )

    # ---------------- ATR ----------------
    df["ATR"] = ta.atr(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        length=ATR_LENGTH,
    )

    # ---------------- REGIME ----------------
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

    return df


# ============================================================
# SIGNAL BUILDERS
# ============================================================

def first_confirmation_after_flip(
    df,
    flip_col,
    condition,
    window_bars=2,
):
    signal = pd.Series(
        False,
        index=df.index,
    )

    flip_positions = np.flatnonzero(
        df[
            flip_col
        ]
        .fillna(False)
        .to_numpy()
    )

    cond = (
        condition
        .fillna(False)
        .to_numpy()
    )

    for flip_pos in flip_positions:
        end_pos = min(
            flip_pos + window_bars,
            len(df),
        )

        for j in range(
            flip_pos,
            end_pos,
        ):
            if cond[j]:
                signal.iloc[j] = True
                break

    return signal


def build_signals(
    base,
    variant,
):
    df = base.copy()

    bull = (
        df["Regime"]
        == "BULL"
    )

    bear = (
        df["Regime"]
        == "BEAR"
    )

    adx_ok = (
        df["ADX"] > ADX_MIN
    )

    if variant == "V1.5A_CURRENT":
        df["Buy_Signal"] = (
            df["ST_Buy_Flip"]
            &
            adx_ok
            &
            df["ADX_Rising_1"]
            &
            bull
        )

        df["Sell_Signal"] = (
            df["ST_Sell_Flip"]
            &
            adx_ok
            &
            df["ADX_Rising_1"]
            &
            bear
        )

    elif variant == "V1.9B_ROBUST":
        buy_condition = (
            adx_ok
            &
            df["ADX_Slope_3"]
            &
            bull
            &
            (
                df["Trend_Direction"]
                == 1
            )
        )

        sell_condition = (
            adx_ok
            &
            df["ADX_Slope_3"]
            &
            bear
            &
            (
                df["Trend_Direction"]
                == -1
            )
        )

        df["Buy_Signal"] = (
            first_confirmation_after_flip(
                df,
                "ST_Buy_Flip",
                buy_condition,
                window_bars=2,
            )
        )

        df["Sell_Signal"] = (
            first_confirmation_after_flip(
                df,
                "ST_Sell_Flip",
                sell_condition,
                window_bars=2,
            )
        )

    else:
        raise ValueError(
            f"Variante desconhecida: {variant}"
        )

    df["Buy_Signal"] = (
        df["Buy_Signal"]
        .fillna(False)
        .astype(bool)
    )

    df["Sell_Signal"] = (
        df["Sell_Signal"]
        .fillna(False)
        .astype(bool)
    )

    return df


# ============================================================
# EVENT STUDY
# ============================================================

def directional_return(
    side,
    entry,
    price,
):
    if side == "BUY":
        return (
            (
                price - entry
            )
            /
            entry
        ) * 100.0

    return (
        (
            entry - price
        )
        /
        entry
    ) * 100.0


def build_events(
    df,
    market,
    variant,
):
    events = []

    for i in range(
        0,
        len(df) - 1
    ):
        row = df.iloc[i]

        side = None

        if bool(
            row["Buy_Signal"]
        ):
            side = "BUY"

        elif bool(
            row["Sell_Signal"]
        ):
            side = "SELL"

        if side is None:
            continue

        entry_i = i + 1

        entry_price = float(
            df["open"].iloc[
                entry_i
            ]
        )

        atr = row["ATR"]

        if (
            pd.isna(atr)
            or atr <= 0
        ):
            continue

        max_horizon = max(
            HORIZONS
        )

        end_i = min(
            entry_i
            +
            max_horizon
            -
            1,
            len(df) - 1,
        )

        window = df.iloc[
            entry_i:
            end_i + 1
        ]

        if window.empty:
            continue

        if side == "BUY":
            mfe_pct = (
                (
                    float(
                        window["high"].max()
                    )
                    -
                    entry_price
                )
                /
                entry_price
            ) * 100.0

            mae_pct = (
                (
                    entry_price
                    -
                    float(
                        window["low"].min()
                    )
                )
                /
                entry_price
            ) * 100.0

            mfe_price = (
                float(
                    window["high"].max()
                )
                -
                entry_price
            )

            mae_price = (
                entry_price
                -
                float(
                    window["low"].min()
                )
            )

        else:
            mfe_pct = (
                (
                    entry_price
                    -
                    float(
                        window["low"].min()
                    )
                )
                /
                entry_price
            ) * 100.0

            mae_pct = (
                (
                    float(
                        window["high"].max()
                    )
                    -
                    entry_price
                )
                /
                entry_price
            ) * 100.0

            mfe_price = (
                entry_price
                -
                float(
                    window["low"].min()
                )
            )

            mae_price = (
                float(
                    window["high"].max()
                )
                -
                entry_price
            )

        event = {
            "market": market,
            "variant": variant,
            "side": side,

            "signal_time": (
                df.index[i]
            ),

            "entry_time": (
                df.index[
                    entry_i
                ]
            ),

            "entry_price": entry_price,
            "signal_atr": float(
                atr
            ),

            "signal_adx": (
                float(
                    row["ADX"]
                )
                if pd.notna(
                    row["ADX"]
                )
                else np.nan
            ),

            "regime": (
                row["Regime"]
            ),

            "mfe_72h_pct": mfe_pct,
            "mae_72h_pct": mae_pct,

            "mfe_72h_atr": (
                mfe_price
                /
                float(atr)
            ),

            "mae_72h_atr": (
                mae_price
                /
                float(atr)
            ),
        }

        for horizon in HORIZONS:
            target_i = (
                entry_i
                +
                horizon
                -
                1
            )

            if target_i >= len(df):
                event[
                    f"ret_{horizon}h_pct"
                ] = np.nan

                continue

            future_close = float(
                df["close"].iloc[
                    target_i
                ]
            )

            event[
                f"ret_{horizon}h_pct"
            ] = directional_return(
                side,
                entry_price,
                future_close,
            )

        events.append(
            event
        )

    return pd.DataFrame(
        events
    )


# ============================================================
# REPORT
# ============================================================

def summarize_group(
    group,
    market,
    variant,
    side,
):
    row = {
        "market": market,
        "variant": variant,
        "side": side,
        "events": len(group),
    }

    if group.empty:
        for horizon in HORIZONS:
            row[
                f"win_{horizon}h_pct"
            ] = np.nan

            row[
                f"mean_ret_{horizon}h_pct"
            ] = np.nan

            row[
                f"median_ret_{horizon}h_pct"
            ] = np.nan

        return row

    for horizon in HORIZONS:
        col = (
            f"ret_{horizon}h_pct"
        )

        valid = group[
            col
        ].dropna()

        if valid.empty:
            row[
                f"win_{horizon}h_pct"
            ] = np.nan

            row[
                f"mean_ret_{horizon}h_pct"
            ] = np.nan

            row[
                f"median_ret_{horizon}h_pct"
            ] = np.nan

        else:
            row[
                f"win_{horizon}h_pct"
            ] = (
                (
                    valid > 0
                ).mean()
                * 100.0
            )

            row[
                f"mean_ret_{horizon}h_pct"
            ] = float(
                valid.mean()
            )

            row[
                f"median_ret_{horizon}h_pct"
            ] = float(
                valid.median()
            )

    row[
        "median_mfe_72h_atr"
    ] = float(
        group[
            "mfe_72h_atr"
        ].median()
    )

    row[
        "mean_mfe_72h_atr"
    ] = float(
        group[
            "mfe_72h_atr"
        ].mean()
    )

    row[
        "median_mae_72h_atr"
    ] = float(
        group[
            "mae_72h_atr"
        ].median()
    )

    row[
        "mean_mae_72h_atr"
    ] = float(
        group[
            "mae_72h_atr"
        ].mean()
    )

    for threshold in [
        1.0,
        1.5,
        2.0,
        3.0,
    ]:
        safe_name = str(
            threshold
        ).replace(
            ".",
            "_"
        )

        row[
            f"mfe_ge_{safe_name}atr_pct"
        ] = (
            (
                group[
                    "mfe_72h_atr"
                ]
                >= threshold
            ).mean()
            * 100.0
        )

    for threshold in [
        1.0,
        1.5,
        2.0,
    ]:
        safe_name = str(
            threshold
        ).replace(
            ".",
            "_"
        )

        row[
            f"mae_ge_{safe_name}atr_pct"
        ] = (
            (
                group[
                    "mae_72h_atr"
                ]
                >= threshold
            ).mean()
            * 100.0
        )

    return row


def print_summary_row(row):
    print(
        f"{row['market']:<17} | "
        f"{row['variant']:<15} | "
        f"{row['side']:<4} | "
        f"N {row['events']:>3} | "
        f"12h mean {row['mean_ret_12h_pct']:>7.3f}% | "
        f"24h mean {row['mean_ret_24h_pct']:>7.3f}% | "
        f"48h mean {row['mean_ret_48h_pct']:>7.3f}% | "
        f"72h mean {row['mean_ret_72h_pct']:>7.3f}% | "
        f"MFE med {row['median_mfe_72h_atr']:>5.2f} ATR | "
        f"MAE med {row['median_mae_72h_atr']:>5.2f} ATR"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 154)
    print("C2G V1.10 - ENTRY EDGE DIAGNOSTIC")
    print("=" * 154)

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

    binance_calc = prepare_indicators(
        binance_raw.loc[
            warmup_start:
            overlap_end
        ].copy()
    )

    bybit_calc = prepare_indicators(
        bybit_raw.loc[
            warmup_start:
            overlap_end
        ].copy()
    )

    binance = (
        binance_calc.loc[
            overlap_start:
            overlap_end
        ].copy()
    )

    bybit = (
        bybit_calc.loc[
            overlap_start:
            overlap_end
        ].copy()
    )

    print(
        f"Período comum: "
        f"{overlap_start} -> {overlap_end}"
    )

    print(
        f"Binance candles: "
        f"{len(binance)}"
    )

    print(
        f"Bybit candles:   "
        f"{len(bybit)}"
    )

    variants = [
        "V1.5A_CURRENT",
        "V1.9B_ROBUST",
    ]

    all_events = []
    summary_rows = []

    market_map = {
        "BINANCE_SPOT": binance,
        "BYBIT_PERPETUAL": bybit,
    }

    for market_name, market_df in (
        market_map.items()
    ):
        for variant in variants:
            signal_df = build_signals(
                market_df,
                variant,
            )

            events = build_events(
                signal_df,
                market_name,
                variant,
            )

            all_events.append(
                events
            )

            print()
            print("=" * 154)
            print(
                f"{market_name} | {variant}"
            )
            print("=" * 154)

            print(
                f"Signals BUY:  "
                f"{int(signal_df['Buy_Signal'].sum())}"
            )

            print(
                f"Signals SELL: "
                f"{int(signal_df['Sell_Signal'].sum())}"
            )

            for side in [
                "ALL",
                "BUY",
                "SELL",
            ]:
                if side == "ALL":
                    group = events
                else:
                    group = events[
                        events[
                            "side"
                        ] == side
                    ]

                row = summarize_group(
                    group,
                    market_name,
                    variant,
                    side,
                )

                summary_rows.append(
                    row
                )

                print_summary_row(
                    row
                )

                if not group.empty:
                    print(
                        f"    MFE >= 1ATR "
                        f"{row['mfe_ge_1_0atr_pct']:.1f}% | "
                        f">=1.5ATR "
                        f"{row['mfe_ge_1_5atr_pct']:.1f}% | "
                        f">=2ATR "
                        f"{row['mfe_ge_2_0atr_pct']:.1f}% | "
                        f">=3ATR "
                        f"{row['mfe_ge_3_0atr_pct']:.1f}%"
                    )

                    print(
                        f"    MAE >= 1ATR "
                        f"{row['mae_ge_1_0atr_pct']:.1f}% | "
                        f">=1.5ATR "
                        f"{row['mae_ge_1_5atr_pct']:.1f}% | "
                        f">=2ATR "
                        f"{row['mae_ge_2_0atr_pct']:.1f}%"
                    )

    events_df = pd.concat(
        all_events,
        ignore_index=True,
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    print()
    print("=" * 154)
    print("C2G V1.10 - KEY COMPARISON")
    print("=" * 154)

    key = summary_df[
        summary_df[
            "side"
        ] == "ALL"
    ].copy()

    show_cols = [
        "market",
        "variant",
        "events",
        "win_24h_pct",
        "mean_ret_24h_pct",
        "median_ret_24h_pct",
        "win_48h_pct",
        "mean_ret_48h_pct",
        "median_ret_48h_pct",
        "median_mfe_72h_atr",
        "median_mae_72h_atr",
        "mfe_ge_1_0atr_pct",
        "mfe_ge_1_5atr_pct",
        "mfe_ge_2_0atr_pct",
        "mfe_ge_3_0atr_pct",
    ]

    print(
        key[
            show_cols
        ].to_string(
            index=False,
            formatters={
                "win_24h_pct": "{:.2f}%".format,
                "mean_ret_24h_pct": "{:.4f}%".format,
                "median_ret_24h_pct": "{:.4f}%".format,
                "win_48h_pct": "{:.2f}%".format,
                "mean_ret_48h_pct": "{:.4f}%".format,
                "median_ret_48h_pct": "{:.4f}%".format,
                "median_mfe_72h_atr": "{:.3f}".format,
                "median_mae_72h_atr": "{:.3f}".format,
                "mfe_ge_1_0atr_pct": "{:.2f}%".format,
                "mfe_ge_1_5atr_pct": "{:.2f}%".format,
                "mfe_ge_2_0atr_pct": "{:.2f}%".format,
                "mfe_ge_3_0atr_pct": "{:.2f}%".format,
            },
        )
    )

    print("=" * 154)

    events_df.to_csv(
        "c2g_v110_entry_edge_events.csv",
        index=False,
    )

    summary_df.to_csv(
        "c2g_v110_entry_edge_summary.csv",
        index=False,
    )

    print()
    print(
        "Eventos salvos em: "
        "c2g_v110_entry_edge_events.csv"
    )

    print(
        "Resumo salvo em: "
        "c2g_v110_entry_edge_summary.csv"
    )

    print()
    print("=" * 154)
    print("COMO INTERPRETAR")
    print("=" * 154)
    print(
        "Se os retornos médios/medianos após os sinais forem positivos "
        "nos dois mercados, mas o backtest continuar negativo, o problema "
        "provavelmente está na saída/gestão."
    )
    print(
        "Se os retornos após os sinais também forem negativos na Bybit, "
        "então o problema está na própria arquitetura de entrada."
    )
    print(
        "MFE mostra quanto o trade normalmente anda a favor; MAE mostra "
        "quanto normalmente anda contra. Isso ajuda a decidir se 3 ATR "
        "de alvo e 1.5 ATR de stop fazem sentido."
    )
    print("=" * 154)
