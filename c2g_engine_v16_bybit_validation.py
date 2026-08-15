import pandas as pd
import pandas_ta as ta
import numpy as np


# ============================================================
# C2G SYSTEM PRO - V1.6 BYBIT PERPETUAL VALIDATION
#
# NÃO otimiza parâmetros.
#
# Candidata principal congelada:
# V1.5B
#   BUY somente em BULL
#   SELL somente em BEAR
#   ATR Ratio >= 0.80 e < 1.20
#
# Benchmark:
# V1.5A
#   BUY somente em BULL
#   SELL somente em BEAR
#   sem filtro ATR Ratio
#
# Regras comuns:
# Supertrend 10/3
# ADX > 25 + Rising
# SL = 1.5 ATR
# TP = 3 ATR
#
# Rodamos:
# 1) resultado bruto
# 2) stress test de custos
#
# Stress test:
# fee 0.055% por lado
# slippage assumido 0.020% por lado
#
# Funding NÃO está incluído ainda.
# ============================================================


FILE_PATH = "btc_data_bybit_perp_1h.csv"

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

# Stress test definido ANTES de olhar o resultado Bybit.
STRESS_FEE_PCT_PER_SIDE = 0.055
STRESS_SLIPPAGE_PCT_PER_SIDE = 0.020


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


def calculate_pnl(
    side,
    entry_price,
    exit_price,
    fee_pct,
    slippage_pct,
):
    if side == "BUY":
        gross = (
            (exit_price - entry_price)
            / entry_price
        ) * 100.0
    else:
        gross = (
            (entry_price - exit_price)
            / entry_price
        ) * 100.0

    costs = (
        fee_pct * 2.0
        +
        slippage_pct * 2.0
    )

    return gross - costs


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
            "exp": 0.0,
            "dd": 0.0,
            "loss_streak": 0,
            "ending_equity": 100.0,
        }

    trades = trades.sort_values(
        "entry_time"
    )

    winners = trades[
        trades["pnl_pct"] > 0
    ]

    losers = trades[
        trades["pnl_pct"] < 0
    ]

    gp = (
        float(
            winners["pnl_pct"].sum()
        )
        if len(winners)
        else 0.0
    )

    gl = (
        abs(
            float(
                losers["pnl_pct"].sum()
            )
        )
        if len(losers)
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
            (1.0 + pnl / 100.0)
        )

    equity = pd.Series(
        equity,
        dtype=float,
    )

    peak = equity.cummax()

    dd = (
        (
            equity / peak
        ) - 1.0
    ) * 100.0

    return {
        "trades": len(trades),
        "wins": len(winners),
        "win_rate": (
            len(winners)
            / len(trades)
            * 100.0
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
        "loss_streak": (
            max_loss_streak(
                trades[
                    "pnl_pct"
                ].tolist()
            )
        ),
        "ending_equity": float(
            equity.iloc[-1]
        ),
    }


def line(label, trades):
    m = metrics(
        trades
    )

    pf = (
        "inf"
        if np.isinf(m["pf"])
        else f"{m['pf']:.3f}"
    )

    print(
        f"{label:<28} | "
        f"Trades {m['trades']:>4} | "
        f"WR {m['win_rate']:>6.2f}% | "
        f"PnL {m['pnl']:>8.2f}% | "
        f"PF {pf:>6} | "
        f"Exp {m['exp']:>8.4f}% | "
        f"DD {m['dd']:>7.2f}% | "
        f"LStreak {m['loss_streak']:>2}"
    )

    return m


def prepare_data():
    print()
    print("=" * 116)
    print("C2G V1.6 - BYBIT PERPETUAL VALIDATION")
    print("=" * 116)
    print("Carregando Bybit BTCUSDT Perpetual 1H...")

    df = pd.read_csv(
        FILE_PATH,
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
                f"Coluna {col} ausente."
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

    print(f"Candles:   {len(df)}")
    print(f"Primeiro:  {df.index.min()}")
    print(f"Último:    {df.index.max()}")

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

    return df


def build_signals(
    base,
    candidate,
):
    df = base.copy()

    buy = (
        df["Base_Buy"]
        &
        (
            df["Regime"]
            == "BULL"
        )
    )

    sell = (
        df["Base_Sell"]
        &
        (
            df["Regime"]
            == "BEAR"
        )
    )

    if candidate == "V1.5B":
        atr_ok = (
            (df["ATR_Ratio"] >= 0.80)
            &
            (df["ATR_Ratio"] < 1.20)
        )

        buy = (
            buy
            &
            atr_ok
        )

        sell = (
            sell
            &
            atr_ok
        )

    df["Buy_Signal"] = buy.astype(
        int
    )

    df["Sell_Signal"] = sell.astype(
        int
    )

    return df


def run_backtest(
    df,
    fee_pct,
    slippage_pct,
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
        # EXISTING POSITION: GAP AT CURRENT OPEN
        # ====================================================
        if position == "BUY":
            if o <= stop:
                exit_price = o

                pnl = calculate_pnl(
                    "BUY",
                    entry_price,
                    exit_price,
                    fee_pct,
                    slippage_pct,
                )

                trades.append({
                    "side": "BUY",
                    "entry_time": entry_time,
                    "exit_time": df.index[i],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "reason": "STOP_GAP",
                    "pnl_pct": pnl,
                })

                position = None

            elif o >= target:
                exit_price = o

                pnl = calculate_pnl(
                    "BUY",
                    entry_price,
                    exit_price,
                    fee_pct,
                    slippage_pct,
                )

                trades.append({
                    "side": "BUY",
                    "entry_time": entry_time,
                    "exit_time": df.index[i],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "reason": "TARGET_GAP",
                    "pnl_pct": pnl,
                })

                position = None

        elif position == "SELL":
            if o >= stop:
                exit_price = o

                pnl = calculate_pnl(
                    "SELL",
                    entry_price,
                    exit_price,
                    fee_pct,
                    slippage_pct,
                )

                trades.append({
                    "side": "SELL",
                    "entry_time": entry_time,
                    "exit_time": df.index[i],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "reason": "STOP_GAP",
                    "pnl_pct": pnl,
                })

                position = None

            elif o <= target:
                exit_price = o

                pnl = calculate_pnl(
                    "SELL",
                    entry_price,
                    exit_price,
                    fee_pct,
                    slippage_pct,
                )

                trades.append({
                    "side": "SELL",
                    "entry_time": entry_time,
                    "exit_time": df.index[i],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "reason": "TARGET_GAP",
                    "pnl_pct": pnl,
                })

                position = None

        # ====================================================
        # ENTRY AT CURRENT OPEN FROM PREVIOUS CLOSED CANDLE
        # ====================================================
        if position is None:
            atr = previous[
                "ATR"
            ]

            if (
                pd.notna(atr)
                and atr > 0
            ):
                if (
                    previous[
                        "Buy_Signal"
                    ]
                    == 1
                ):
                    position = "BUY"
                    entry_time = df.index[i]
                    entry_price = o

                    stop = (
                        entry_price
                        -
                        STOP_ATR * atr
                    )

                    target = (
                        entry_price
                        +
                        TARGET_ATR * atr
                    )

                elif (
                    previous[
                        "Sell_Signal"
                    ]
                    == 1
                ):
                    position = "SELL"
                    entry_time = df.index[i]
                    entry_price = o

                    stop = (
                        entry_price
                        +
                        STOP_ATR * atr
                    )

                    target = (
                        entry_price
                        -
                        TARGET_ATR * atr
                    )

        # ====================================================
        # INTRABAR MANAGEMENT
        # ====================================================
        if position == "BUY":
            stop_hit = (
                l <= stop
            )

            target_hit = (
                h >= target
            )

            exit_price = None
            reason = None

            # Conservador: STOP primeiro se ambos tocam.
            if stop_hit:
                exit_price = stop
                reason = "STOP"

            elif target_hit:
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
                pnl = calculate_pnl(
                    "BUY",
                    entry_price,
                    exit_price,
                    fee_pct,
                    slippage_pct,
                )

                trades.append({
                    "side": "BUY",
                    "entry_time": entry_time,
                    "exit_time": df.index[i],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "reason": reason,
                    "pnl_pct": pnl,
                })

                position = None

        elif position == "SELL":
            stop_hit = (
                h >= stop
            )

            target_hit = (
                l <= target
            )

            exit_price = None
            reason = None

            if stop_hit:
                exit_price = stop
                reason = "STOP"

            elif target_hit:
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
                pnl = calculate_pnl(
                    "SELL",
                    entry_price,
                    exit_price,
                    fee_pct,
                    slippage_pct,
                )

                trades.append({
                    "side": "SELL",
                    "entry_time": entry_time,
                    "exit_time": df.index[i],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "reason": reason,
                    "pnl_pct": pnl,
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

        trades[
            "year"
        ] = (
            trades[
                "entry_time"
            ].dt.year
        )

    return trades


def print_report(
    candidate,
    cost_name,
    trades,
):
    print()
    print("=" * 120)
    print(
        f"{candidate} | {cost_name}"
    )
    print("=" * 120)

    total = line(
        "FULL HISTORY",
        trades,
    )

    line(
        "BUY ONLY",
        trades[
            trades["side"]
            == "BUY"
        ],
    )

    line(
        "SELL ONLY",
        trades[
            trades["side"]
            == "SELL"
        ],
    )

    print()
    print("BY YEAR")
    print("-" * 120)

    for year, group in trades.groupby(
        "year"
    ):
        line(
            str(year),
            group,
        )

    return total


if __name__ == "__main__":
    base = prepare_data()

    candidates = [
        "V1.5A",
        "V1.5B",
    ]

    cost_scenarios = [
        {
            "name": "GROSS - NO COSTS",
            "fee": 0.0,
            "slippage": 0.0,
        },
        {
            "name": (
                "COST STRESS - "
                "0.055% FEE + "
                "0.020% SLIPPAGE / SIDE"
            ),
            "fee": (
                STRESS_FEE_PCT_PER_SIDE
            ),
            "slippage": (
                STRESS_SLIPPAGE_PCT_PER_SIDE
            ),
        },
    ]

    ranking = []

    for candidate in candidates:
        df = build_signals(
            base,
            candidate,
        )

        print()
        print(
            f"{candidate} | "
            f"BUY signals "
            f"{int(df['Buy_Signal'].sum())} | "
            f"SELL signals "
            f"{int(df['Sell_Signal'].sum())}"
        )

        for costs in cost_scenarios:
            trades = run_backtest(
                df,
                fee_pct=costs[
                    "fee"
                ],
                slippage_pct=costs[
                    "slippage"
                ],
            )

            total = print_report(
                candidate,
                costs["name"],
                trades,
            )

            ranking.append({
                "candidate": candidate,
                "cost_scenario": costs[
                    "name"
                ],
                **total,
            })

            safe_cost = (
                "gross"
                if costs["fee"] == 0
                else "cost_stress"
            )

            trades.to_csv(
                f"c2g_v16_"
                f"{candidate.lower()}_"
                f"{safe_cost}_trades.csv",
                index=False,
            )

    ranking = pd.DataFrame(
        ranking
    )

    print()
    print("=" * 132)
    print("C2G V1.6 - BYBIT VALIDATION SUMMARY")
    print("=" * 132)

    print(
        ranking[
            [
                "candidate",
                "cost_scenario",
                "trades",
                "win_rate",
                "pnl",
                "pf",
                "exp",
                "dd",
                "loss_streak",
                "ending_equity",
            ]
        ].to_string(
            index=False,
            formatters={
                "win_rate": "{:.2f}%".format,
                "pnl": "{:.2f}%".format,
                "pf": "{:.3f}".format,
                "exp": "{:.4f}%".format,
                "dd": "{:.2f}%".format,
                "ending_equity": "{:.2f}".format,
            },
        )
    )

    print("=" * 132)

    ranking.to_csv(
        "c2g_v16_bybit_validation_summary.csv",
        index=False,
    )

    print()
    print(
        "Resumo salvo em: "
        "c2g_v16_bybit_validation_summary.csv"
    )
    print(
        "Funding não está incluído nesta fase."
    )
