import pandas as pd
import pandas_ta as ta
import numpy as np


# ============================================================
# C2G SYSTEM PRO - V1.11 BUY EXIT PATH DIAGNOSTIC
#
# Objetivo:
# A V1.10 mostrou que:
# - BUY tem comportamento direcional positivo em Binance e Bybit
#   após 24h/48h/72h;
# - SELL continua fraco/inconsistente;
# - o backtest BUY ainda sofre com a lógica SL 1.5 ATR / TP 3 ATR.
#
# Portanto esta versão:
# 1) DESLIGA SELL temporariamente;
# 2) mantém duas entradas BUY já conhecidas:
#    - V1.5A_CURRENT
#    - V1.9B_ROBUST
# 3) NÃO procura um "melhor parâmetro" em centenas de combinações;
# 4) mede a ORDEM real dos eventos de preço após cada entrada:
#    - qual nível ATR é tocado primeiro;
#    - quanto tempo leva;
#    - se o trade é parado antes de andar a favor;
# 5) testa uma pequena matriz PREDEFINIDA de saídas para diagnóstico.
#
# Mercados:
# - Binance Spot
# - Bybit Perpetual
# Mesmo período disponível na Bybit.
#
# Regras de entrada:
#
# V1.5A_CURRENT BUY
# - Supertrend flip BUY
# - ADX > 25
# - ADX atual > ADX anterior
# - Regime BULL
#
# V1.9B_ROBUST BUY
# - confirmação no flip ou no candle seguinte
# - ADX > 25
# - ADX atual > ADX de 3 candles atrás
# - Regime BULL
#
# Saídas testadas (pequena matriz):
# A) SL 1.5 ATR / TP 3.0 ATR  [baseline]
# B) SL 2.0 ATR / TP 3.0 ATR
# C) SL 2.0 ATR / TP 2.0 ATR
# D) SL 2.5 ATR / TP 3.0 ATR
# E) Time Exit 24h
# F) Time Exit 48h
#
# IMPORTANTE:
# Isso ainda é pesquisa in-sample/cross-market, não validação final.
# O objetivo é entender o formato do payoff e reduzir stop-outs
# prematuros, NÃO escolher automaticamente o maior PF.
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

# Stress de custos fixo (por lado), apenas para relatório secundário.
STRESS_FEE_PCT_PER_SIDE = 0.055
STRESS_SLIPPAGE_PCT_PER_SIDE = 0.020


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

    df["Regime"] = "OTHER"

    df.loc[
        bull,
        "Regime"
    ] = "BULL"

    return df


def first_confirmation_after_flip(
    df,
    condition,
    window_bars=2,
):
    signal = pd.Series(
        False,
        index=df.index,
    )

    flip_positions = np.flatnonzero(
        df[
            "ST_Buy_Flip"
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


def build_buy_signals(
    base,
    variant,
):
    df = base.copy()

    bull = (
        df["Regime"] == "BULL"
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

    elif variant == "V1.9B_ROBUST":
        condition = (
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

        df["Buy_Signal"] = (
            first_confirmation_after_flip(
                df,
                condition,
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

    return df


# ============================================================
# EVENT PATH ANALYSIS
# ============================================================

def build_buy_events(
    df,
    market,
    variant,
    max_hours=72,
):
    events = []

    for i in range(
        len(df) - 1
    ):
        if not bool(
            df["Buy_Signal"].iloc[i]
        ):
            continue

        atr = df["ATR"].iloc[i]

        if (
            pd.isna(atr)
            or atr <= 0
        ):
            continue

        entry_i = i + 1

        entry_price = float(
            df["open"].iloc[
                entry_i
            ]
        )

        end_i = min(
            entry_i
            +
            max_hours
            -
            1,
            len(df) - 1,
        )

        window = df.iloc[
            entry_i:
            end_i + 1
        ]

        event = {
            "market": market,
            "variant": variant,
            "signal_time": df.index[i],
            "entry_time": df.index[
                entry_i
            ],
            "entry_price": entry_price,
            "atr": float(
                atr
            ),
            "adx": float(
                df["ADX"].iloc[i]
            ),
        }

        # Primeiro toque de níveis favoráveis/adversos.
        for atr_multiple in [
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
        ]:
            up_level = (
                entry_price
                +
                atr_multiple
                * atr
            )

            down_level = (
                entry_price
                -
                atr_multiple
                * atr
            )

            up_hit_hour = np.nan
            down_hit_hour = np.nan

            for h_idx, (
                ts,
                row
            ) in enumerate(
                window.iterrows(),
                start=1,
            ):
                if (
                    pd.isna(
                        up_hit_hour
                    )
                    and
                    float(
                        row["high"]
                    )
                    >= up_level
                ):
                    up_hit_hour = (
                        h_idx
                    )

                if (
                    pd.isna(
                        down_hit_hour
                    )
                    and
                    float(
                        row["low"]
                    )
                    <= down_level
                ):
                    down_hit_hour = (
                        h_idx
                    )

                if (
                    not pd.isna(
                        up_hit_hour
                    )
                    and
                    not pd.isna(
                        down_hit_hour
                    )
                ):
                    break

            safe = str(
                atr_multiple
            ).replace(
                ".",
                "_"
            )

            event[
                f"up_{safe}atr_hour"
            ] = up_hit_hour

            event[
                f"down_{safe}atr_hour"
            ] = down_hit_hour

        # Retorno de tempo fixo.
        for hours in [
            24,
            48,
        ]:
            target_i = (
                entry_i
                +
                hours
                -
                1
            )

            if target_i < len(df):
                exit_price = float(
                    df["close"].iloc[
                        target_i
                    ]
                )

                event[
                    f"time_{hours}h_ret_pct"
                ] = (
                    (
                        exit_price
                        -
                        entry_price
                    )
                    /
                    entry_price
                ) * 100.0
            else:
                event[
                    f"time_{hours}h_ret_pct"
                ] = np.nan

        events.append(
            event
        )

    return pd.DataFrame(
        events
    )


def first_touch_outcome(
    event,
    stop_atr,
    target_atr,
):
    stop_key = (
        "down_"
        +
        str(
            stop_atr
        ).replace(
            ".",
            "_"
        )
        +
        "atr_hour"
    )

    target_key = (
        "up_"
        +
        str(
            target_atr
        ).replace(
            ".",
            "_"
        )
        +
        "atr_hour"
    )

    stop_hour = event[
        stop_key
    ]

    target_hour = event[
        target_key
    ]

    if (
        pd.isna(stop_hour)
        and
        pd.isna(target_hour)
    ):
        return (
            "NONE",
            np.nan,
        )

    if pd.isna(stop_hour):
        return (
            "TARGET",
            target_hour,
        )

    if pd.isna(target_hour):
        return (
            "STOP",
            stop_hour,
        )

    # Se ambos aparecem na mesma hora/candle,
    # mantém abordagem conservadora: STOP primeiro.
    if stop_hour <= target_hour:
        return (
            "STOP",
            stop_hour,
        )

    return (
        "TARGET",
        target_hour,
    )


def summarize_first_touch(
    events,
    stop_atr,
    target_atr,
):
    outcomes = []

    for _, event in events.iterrows():
        outcome, hours = (
            first_touch_outcome(
                event,
                stop_atr,
                target_atr,
            )
        )

        outcomes.append({
            "outcome": outcome,
            "hours": hours,
        })

    outcomes = pd.DataFrame(
        outcomes
    )

    n = len(outcomes)

    if n == 0:
        return {
            "events": 0,
            "target_first_pct": 0.0,
            "stop_first_pct": 0.0,
            "none_pct": 0.0,
            "median_hours_target": np.nan,
            "median_hours_stop": np.nan,
        }

    target = (
        outcomes[
            "outcome"
        ] == "TARGET"
    )

    stop = (
        outcomes[
            "outcome"
        ] == "STOP"
    )

    none = (
        outcomes[
            "outcome"
        ] == "NONE"
    )

    return {
        "events": n,

        "target_first_pct": (
            target.mean()
            * 100.0
        ),

        "stop_first_pct": (
            stop.mean()
            * 100.0
        ),

        "none_pct": (
            none.mean()
            * 100.0
        ),

        "median_hours_target": (
            outcomes.loc[
                target,
                "hours"
            ].median()
            if target.any()
            else np.nan
        ),

        "median_hours_stop": (
            outcomes.loc[
                stop,
                "hours"
            ].median()
            if stop.any()
            else np.nan
        ),
    }


# ============================================================
# BACKTEST EXIT MODELS
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

    costs = (
        fee_pct * 2.0
        +
        slippage_pct * 2.0
    )

    return gross - costs


def run_buy_exit_backtest(
    df,
    stop_atr=None,
    target_atr=None,
    time_exit_hours=None,
    fee_pct=0.0,
    slippage_pct=0.0,
):
    trades = []

    position = False
    entry_price = None
    entry_time = None
    stop = None
    target = None
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

        # Entrada
        if not position:
            atr = previous[
                "ATR"
            ]

            if (
                bool(
                    previous[
                        "Buy_Signal"
                    ]
                )
                and
                pd.notna(atr)
                and
                atr > 0
            ):
                position = True
                entry_price = o
                entry_time = (
                    df.index[i]
                )
                bars_held = 0

                if stop_atr is not None:
                    stop = (
                        entry_price
                        -
                        stop_atr
                        * atr
                    )
                else:
                    stop = None

                if target_atr is not None:
                    target = (
                        entry_price
                        +
                        target_atr
                        * atr
                    )
                else:
                    target = None

        if not position:
            continue

        bars_held += 1

        exit_price = None
        reason = None

        # Gap-aware stop/target.
        if (
            stop is not None
            and
            o <= stop
        ):
            exit_price = o
            reason = "STOP_GAP"

        elif (
            target is not None
            and
            o >= target
        ):
            exit_price = o
            reason = "TARGET_GAP"

        else:
            stop_hit = (
                stop is not None
                and
                l <= stop
            )

            target_hit = (
                target is not None
                and
                h >= target
            )

            if stop_hit:
                exit_price = stop
                reason = "STOP"

            elif target_hit:
                exit_price = target
                reason = "TAKE_PROFIT"

            elif (
                time_exit_hours
                is not None
                and
                bars_held
                >= time_exit_hours
            ):
                exit_price = c
                reason = (
                    f"TIME_{time_exit_hours}H"
                )

        if exit_price is not None:
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
                "reason": reason,
                "pnl_pct": pnl,
                "bars_held": bars_held,
            })

            position = False

    return pd.DataFrame(
        trades
    )


# ============================================================
# METRICS
# ============================================================

def metrics(trades):
    if trades.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "pnl": 0.0,
            "pf": 0.0,
            "expectancy": 0.0,
            "dd": 0.0,
            "avg_bars": 0.0,
        }

    wins = trades[
        trades["pnl_pct"] > 0
    ]

    losses = trades[
        trades["pnl_pct"] < 0
    ]

    gp = (
        float(
            wins[
                "pnl_pct"
            ].sum()
        )
        if len(wins)
        else 0.0
    )

    gl = (
        abs(
            float(
                losses[
                    "pnl_pct"
                ].sum()
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
            trades
        ),

        "win_rate": (
            len(wins)
            /
            len(trades)
            *
            100.0
        ),

        "pnl": float(
            trades[
                "pnl_pct"
            ].sum()
        ),

        "pf": pf,

        "expectancy": float(
            trades[
                "pnl_pct"
            ].mean()
        ),

        "dd": float(
            dd.min()
        ),

        "avg_bars": float(
            trades[
                "bars_held"
            ].mean()
        ),
    }


def metric_line(
    label,
    trades,
):
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
        f"{label:<28} | "
        f"Trades {m['trades']:>3} | "
        f"WR {m['win_rate']:>6.2f}% | "
        f"PnL {m['pnl']:>8.2f}% | "
        f"PF {pf_text:>6} | "
        f"Exp {m['expectancy']:>8.4f}% | "
        f"DD {m['dd']:>7.2f}% | "
        f"AvgBars {m['avg_bars']:>6.1f}"
    )

    return m


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 152)
    print("C2G V1.11 - BUY EXIT PATH DIAGNOSTIC")
    print("=" * 152)

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

    binance = prepare_indicators(
        binance_raw.loc[
            warmup_start:
            overlap_end
        ].copy()
    ).loc[
        overlap_start:
        overlap_end
    ]

    bybit = prepare_indicators(
        bybit_raw.loc[
            warmup_start:
            overlap_end
        ].copy()
    ).loc[
        overlap_start:
        overlap_end
    ]

    print(
        f"Período comum: "
        f"{overlap_start} -> {overlap_end}"
    )

    variants = [
        "V1.5A_CURRENT",
        "V1.9B_ROBUST",
    ]

    market_map = {
        "BINANCE_SPOT": binance,
        "BYBIT_PERPETUAL": bybit,
    }

    exit_models = [
        {
            "name": "SL1.5_TP3.0",
            "stop": 1.5,
            "target": 3.0,
            "time": None,
        },
        {
            "name": "SL2.0_TP3.0",
            "stop": 2.0,
            "target": 3.0,
            "time": None,
        },
        {
            "name": "SL2.0_TP2.0",
            "stop": 2.0,
            "target": 2.0,
            "time": None,
        },
        {
            "name": "SL2.5_TP3.0",
            "stop": 2.5,
            "target": 3.0,
            "time": None,
        },
        {
            "name": "TIME24H",
            "stop": None,
            "target": None,
            "time": 24,
        },
        {
            "name": "TIME48H",
            "stop": None,
            "target": None,
            "time": 48,
        },
    ]

    ranking_rows = []
    touch_rows = []
    all_events = []

    for market_name, market_df in (
        market_map.items()
    ):
        for variant in variants:
            df = build_buy_signals(
                market_df,
                variant,
            )

            events = build_buy_events(
                df,
                market_name,
                variant,
                max_hours=72,
            )

            all_events.append(
                events
            )

            print()
            print("=" * 152)
            print(
                f"{market_name} | {variant}"
            )
            print("=" * 152)
            print(
                f"BUY signals: "
                f"{int(df['Buy_Signal'].sum())}"
            )

            # First-touch diagnostics.
            for stop_atr, target_atr in [
                (1.5, 3.0),
                (2.0, 3.0),
                (2.0, 2.0),
                (2.5, 3.0),
            ]:
                s = summarize_first_touch(
                    events,
                    stop_atr,
                    target_atr,
                )

                print(
                    f"First touch SL {stop_atr:.1f} / "
                    f"TP {target_atr:.1f} ATR | "
                    f"Target first {s['target_first_pct']:.2f}% | "
                    f"Stop first {s['stop_first_pct']:.2f}% | "
                    f"None {s['none_pct']:.2f}% | "
                    f"Med hrs target {s['median_hours_target']} | "
                    f"Med hrs stop {s['median_hours_stop']}"
                )

                touch_rows.append({
                    "market": market_name,
                    "variant": variant,
                    "stop_atr": stop_atr,
                    "target_atr": target_atr,
                    **s,
                })

            print()
            print("EXIT MODEL BACKTEST - GROSS")
            print("-" * 152)

            for model in exit_models:
                trades = (
                    run_buy_exit_backtest(
                        df,
                        stop_atr=model[
                            "stop"
                        ],
                        target_atr=model[
                            "target"
                        ],
                        time_exit_hours=model[
                            "time"
                        ],
                    )
                )

                m = metric_line(
                    model["name"],
                    trades,
                )

                ranking_rows.append({
                    "market": market_name,
                    "variant": variant,
                    "exit_model": model[
                        "name"
                    ],
                    **m,
                })

    # ========================================================
    # CROSS-MARKET EXIT MODEL RANKING
    # ========================================================

    ranking = pd.DataFrame(
        ranking_rows
    )

    cross_rows = []

    for (
        variant,
        exit_model
    ), group in ranking.groupby(
        [
            "variant",
            "exit_model",
        ]
    ):
        if len(group) < 2:
            continue

        min_pf = float(
            group["pf"].min()
        )

        min_exp = float(
            group[
                "expectancy"
            ].min()
        )

        min_trades = int(
            group[
                "trades"
            ].min()
        )

        both_positive = bool(
            (
                group["pf"] > 1.0
            ).all()
            and
            (
                group[
                    "expectancy"
                ] > 0
            ).all()
        )

        cross_rows.append({
            "variant": variant,
            "exit_model": exit_model,
            "min_pf_cross_market": min_pf,
            "min_exp_cross_market": min_exp,
            "min_trades_cross_market": min_trades,
            "both_markets_positive": both_positive,

            "binance_pf": float(
                group.loc[
                    group[
                        "market"
                    ]
                    == "BINANCE_SPOT",
                    "pf",
                ].iloc[0]
            ),

            "bybit_pf": float(
                group.loc[
                    group[
                        "market"
                    ]
                    == "BYBIT_PERPETUAL",
                    "pf",
                ].iloc[0]
            ),
        })

    cross = pd.DataFrame(
        cross_rows
    )

    cross = cross.sort_values(
        by=[
            "both_markets_positive",
            "min_pf_cross_market",
            "min_exp_cross_market",
            "min_trades_cross_market",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    print()
    print("=" * 152)
    print(
        "C2G V1.11 - BUY EXIT CROSS-MARKET RANKING"
    )
    print("=" * 152)

    print(
        cross.to_string(
            index=False,
            formatters={
                "min_pf_cross_market": "{:.3f}".format,
                "min_exp_cross_market": "{:.4f}%".format,
                "binance_pf": "{:.3f}".format,
                "bybit_pf": "{:.3f}".format,
            },
        )
    )

    # ========================================================
    # COST STRESS ON TOP CONFIG ONLY
    # ========================================================

    if not cross.empty:
        best_variant = (
            cross.iloc[0][
                "variant"
            ]
        )

        best_exit = (
            cross.iloc[0][
                "exit_model"
            ]
        )

        best_model = next(
            model
            for model in exit_models
            if model[
                "name"
            ] == best_exit
        )

        print()
        print("=" * 152)
        print(
            f"COST STRESS - {best_variant} | {best_exit}"
        )
        print("=" * 152)

        for market_name, market_df in (
            market_map.items()
        ):
            df = build_buy_signals(
                market_df,
                best_variant,
            )

            trades = (
                run_buy_exit_backtest(
                    df,
                    stop_atr=best_model[
                        "stop"
                    ],
                    target_atr=best_model[
                        "target"
                    ],
                    time_exit_hours=best_model[
                        "time"
                    ],
                    fee_pct=(
                        STRESS_FEE_PCT_PER_SIDE
                    ),
                    slippage_pct=(
                        STRESS_SLIPPAGE_PCT_PER_SIDE
                    ),
                )
            )

            metric_line(
                market_name,
                trades,
            )

    # ========================================================
    # SAVE
    # ========================================================

    ranking.to_csv(
        "c2g_v111_buy_exit_market_results.csv",
        index=False,
    )

    cross.to_csv(
        "c2g_v111_buy_exit_cross_market_ranking.csv",
        index=False,
    )

    pd.DataFrame(
        touch_rows
    ).to_csv(
        "c2g_v111_first_touch_diagnostic.csv",
        index=False,
    )

    pd.concat(
        all_events,
        ignore_index=True,
    ).to_csv(
        "c2g_v111_buy_path_events.csv",
        index=False,
    )

    print()
    print("=" * 152)
    print("ARQUIVOS GERADOS")
    print("=" * 152)
    print(
        "c2g_v111_buy_exit_market_results.csv"
    )
    print(
        "c2g_v111_buy_exit_cross_market_ranking.csv"
    )
    print(
        "c2g_v111_first_touch_diagnostic.csv"
    )
    print(
        "c2g_v111_buy_path_events.csv"
    )
    print("=" * 152)
