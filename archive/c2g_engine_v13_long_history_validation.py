import pandas as pd
import pandas_ta as ta
import numpy as np


# ============================================================
# C2G SYSTEM PRO - V1.3 LONG HISTORY VALIDATION
#
# IMPORTANT:
# This script DOES NOT optimize the strategy.
#
# It freezes the current best V1.2C rules:
# - Supertrend 10 / 3
# - ADX > 25
# - ADX must be rising
# - Stop = 1.5 ATR
# - Take Profit = 3.0 ATR
#
# The purpose is to test whether the same rules survive on
# older Binance BTC/USDT 1H data that was not part of the
# original ~4-year development sample.
# ============================================================

FILE_PATH = "btc_data_binance_full_1h.csv"

SUPERTREND_LENGTH = 10
SUPERTREND_MULTIPLIER = 3.0

ADX_LENGTH = 14
ADX_MIN = 25.0

ATR_LENGTH = 14
STOP_ATR = 1.5
TARGET_ATR = 3.0

FEE_PCT = 0.0
SLIPPAGE_PCT = 0.0

# Original research window started approximately here.
# This lets us separate the "older unseen history" from the
# period already used during strategy development.
ORIGINAL_SAMPLE_START = pd.Timestamp("2022-08-15 00:00:00")


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
    fee_pct=0.0,
    slippage_pct=0.0,
):
    if side == "BUY":
        gross_pct = (
            (exit_price - entry_price)
            / entry_price
        ) * 100.0
    else:
        gross_pct = (
            (entry_price - exit_price)
            / entry_price
        ) * 100.0

    total_cost = (
        fee_pct * 2.0
        +
        slippage_pct * 2.0
    )

    return gross_pct - total_cost


def build_equity_curve(
    pnl_values,
    initial_equity=100.0,
):
    equity = [initial_equity]

    for pnl_pct in pnl_values:
        equity.append(
            equity[-1]
            *
            (1.0 + pnl_pct / 100.0)
        )

    return pd.Series(
        equity,
        dtype=float,
    )


def max_consecutive_losses(pnl_values):
    best = 0
    current = 0

    for pnl in pnl_values:
        if pnl < 0:
            current += 1
            best = max(best, current)
        else:
            current = 0

    return best


def calculate_metrics(trades_df):
    if trades_df.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "pnl": 0.0,
            "expectancy": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "max_loss_streak": 0,
            "ending_equity": 100.0,
        }

    winners = trades_df[
        trades_df["pnl_pct"] > 0
    ]

    losers = trades_df[
        trades_df["pnl_pct"] < 0
    ]

    total = len(trades_df)
    wins = len(winners)
    losses = len(losers)

    gross_profit = (
        float(winners["pnl_pct"].sum())
        if wins
        else 0.0
    )

    gross_loss = (
        abs(float(losers["pnl_pct"].sum()))
        if losses
        else 0.0
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    equity = build_equity_curve(
        trades_df["pnl_pct"].tolist()
    )

    peak = equity.cummax()

    drawdown = (
        (equity / peak) - 1.0
    ) * 100.0

    return {
        "trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": (
            wins / total * 100.0
        ),
        "pnl": float(
            trades_df["pnl_pct"].sum()
        ),
        "expectancy": float(
            trades_df["pnl_pct"].mean()
        ),
        "avg_win": (
            float(winners["pnl_pct"].mean())
            if wins
            else 0.0
        ),
        "avg_loss": (
            float(losers["pnl_pct"].mean())
            if losses
            else 0.0
        ),
        "profit_factor": profit_factor,
        "max_drawdown": float(
            drawdown.min()
        ),
        "max_loss_streak": (
            max_consecutive_losses(
                trades_df["pnl_pct"].tolist()
            )
        ),
        "ending_equity": float(
            equity.iloc[-1]
        ),
    }


def prepare_data():
    print()
    print("=" * 84)
    print("C2G V1.3 - LONG HISTORY VALIDATION")
    print("=" * 84)
    print("Carregando histórico completo...")

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

    required = [
        "open",
        "high",
        "low",
        "close",
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Coluna '{col}' não encontrada."
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

    print(f"Candles:        {len(df)}")
    print(f"Primeiro:       {df.index.min()}")
    print(f"Último:         {df.index.max()}")

    # ---------------- SUPERTREND ----------------
    st = ta.supertrend(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        length=SUPERTREND_LENGTH,
        multiplier=SUPERTREND_MULTIPLIER,
    )

    st_dir = find_column(
        st.columns,
        "SUPERTd_",
    )

    st_line = find_column(
        st.columns,
        "SUPERT_",
    )

    df["Trend_Direction"] = st[st_dir]
    df["Trend_Line"] = st[st_line]
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

    adx_col = find_column(
        adx_df.columns,
        "ADX_",
    )

    df["ADX"] = adx_df[adx_col]

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

    # ---------------- FROZEN V1.2C SIGNALS ----------------
    df["Buy_Signal"] = (
        (df["Signal"] == 2)
        &
        (df["ADX"] > ADX_MIN)
        &
        (df["ADX_Rising"])
    ).astype(int)

    df["Sell_Signal"] = (
        (df["Signal"] == -2)
        &
        (df["ADX"] > ADX_MIN)
        &
        (df["ADX_Rising"])
    ).astype(int)

    print(
        f"BUY signals:    "
        f"{int(df['Buy_Signal'].sum())}"
    )

    print(
        f"SELL signals:   "
        f"{int(df['Sell_Signal'].sum())}"
    )

    return df


def run_backtest(df):
    trades = []

    position = None
    entry_price = None
    entry_time = None
    stop_loss = None
    take_profit = None

    for i in range(1, len(df)):
        previous = df.iloc[i - 1]
        current = df.iloc[i]

        current_open = float(
            current["open"]
        )

        current_high = float(
            current["high"]
        )

        current_low = float(
            current["low"]
        )

        current_close = float(
            current["close"]
        )

        # ====================================================
        # ENTRY AT CURRENT OPEN USING PREVIOUS CLOSED BAR SIGNAL
        # ====================================================
        if position is None:
            previous_atr = previous["ATR"]

            if (
                pd.notna(previous_atr)
                and previous_atr > 0
            ):
                if previous["Buy_Signal"] == 1:
                    position = "BUY"
                    entry_time = df.index[i]
                    entry_price = current_open

                    stop_loss = (
                        entry_price
                        -
                        STOP_ATR * previous_atr
                    )

                    take_profit = (
                        entry_price
                        +
                        TARGET_ATR * previous_atr
                    )

                elif previous["Sell_Signal"] == 1:
                    position = "SELL"
                    entry_time = df.index[i]
                    entry_price = current_open

                    stop_loss = (
                        entry_price
                        +
                        STOP_ATR * previous_atr
                    )

                    take_profit = (
                        entry_price
                        -
                        TARGET_ATR * previous_atr
                    )

        # ====================================================
        # MANAGE BUY
        # ====================================================
        if position == "BUY":
            exit_price = None
            reason = None

            stop_hit = (
                current_low <= stop_loss
            )

            target_hit = (
                current_high >= take_profit
            )

            # Conservative intrabar assumption:
            # if both are touched, STOP wins.
            if stop_hit:
                exit_price = stop_loss
                reason = "STOP"

            elif target_hit:
                exit_price = take_profit
                reason = "TAKE_PROFIT"

            elif (
                current["Trend_Direction"] == -1
            ):
                exit_price = current_close
                reason = "TREND_REVERSAL"

            if exit_price is not None:
                pnl = calculate_pnl(
                    "BUY",
                    entry_price,
                    exit_price,
                    FEE_PCT,
                    SLIPPAGE_PCT,
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

        # ====================================================
        # MANAGE SELL
        # ====================================================
        elif position == "SELL":
            exit_price = None
            reason = None

            stop_hit = (
                current_high >= stop_loss
            )

            target_hit = (
                current_low <= take_profit
            )

            if stop_hit:
                exit_price = stop_loss
                reason = "STOP"

            elif target_hit:
                exit_price = take_profit
                reason = "TAKE_PROFIT"

            elif (
                current["Trend_Direction"] == 1
            ):
                exit_price = current_close
                reason = "TREND_REVERSAL"

            if exit_price is not None:
                pnl = calculate_pnl(
                    "SELL",
                    entry_price,
                    exit_price,
                    FEE_PCT,
                    SLIPPAGE_PCT,
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

    if position is not None:
        final_price = float(
            df["close"].iloc[-1]
        )

        pnl = calculate_pnl(
            position,
            entry_price,
            final_price,
            FEE_PCT,
            SLIPPAGE_PCT,
        )

        trades.append({
            "side": position,
            "entry_time": entry_time,
            "exit_time": df.index[-1],
            "entry_price": entry_price,
            "exit_price": final_price,
            "reason": "END_OF_DATA",
            "pnl_pct": pnl,
        })

    return pd.DataFrame(trades)


def metric_line(label, trades_df):
    m = calculate_metrics(trades_df)

    pf = (
        "inf"
        if np.isinf(m["profit_factor"])
        else f"{m['profit_factor']:.3f}"
    )

    print(
        f"{label:<22} | "
        f"Trades {m['trades']:>4} | "
        f"WR {m['win_rate']:>6.2f}% | "
        f"PnL {m['pnl']:>8.2f}% | "
        f"PF {pf:>6} | "
        f"Exp {m['expectancy']:>8.4f}% | "
        f"DD {m['max_drawdown']:>7.2f}%"
    )

    return m


def report(full_df, trades):
    if trades.empty:
        print("Nenhum trade encontrado.")
        return

    trades = trades.copy()

    trades["entry_time"] = pd.to_datetime(
        trades["entry_time"]
    )

    trades["exit_time"] = pd.to_datetime(
        trades["exit_time"]
    )

    print()
    print("=" * 112)
    print("C2G V1.2C FROZEN RULES - FULL HISTORY RESULT")
    print("=" * 112)

    overall = metric_line(
        "FULL HISTORY",
        trades,
    )

    metric_line(
        "BUY ONLY",
        trades[
            trades["side"] == "BUY"
        ],
    )

    metric_line(
        "SELL ONLY",
        trades[
            trades["side"] == "SELL"
        ],
    )

    # ========================================================
    # KEY ROBUSTNESS TEST:
    # older data that was not used in the original 4y research
    # ========================================================
    older = trades[
        trades["entry_time"]
        <
        ORIGINAL_SAMPLE_START
    ]

    original_period = trades[
        trades["entry_time"]
        >=
        ORIGINAL_SAMPLE_START
    ]

    print()
    print("=" * 112)
    print("ROBUSTNESS SPLIT")
    print("=" * 112)

    metric_line(
        "OLDER HOLDOUT",
        older,
    )

    metric_line(
        "ORIGINAL PERIOD",
        original_period,
    )

    if not older.empty:
        metric_line(
            "HOLDOUT BUY",
            older[
                older["side"] == "BUY"
            ],
        )

        metric_line(
            "HOLDOUT SELL",
            older[
                older["side"] == "SELL"
            ],
        )

    print()
    print("=" * 112)
    print("RESULT BY YEAR")
    print("=" * 112)

    trades["year"] = (
        trades["entry_time"].dt.year
    )

    yearly_rows = []

    for year, group in trades.groupby("year"):
        m = metric_line(
            str(year),
            group,
        )

        yearly_rows.append({
            "year": year,
            **m,
        })

    print()
    print("=" * 112)
    print("SELL ONLY - RESULT BY YEAR")
    print("=" * 112)

    sells = trades[
        trades["side"] == "SELL"
    ].copy()

    if sells.empty:
        print("Sem operações SELL.")
    else:
        for year, group in sells.groupby(
            sells["entry_time"].dt.year
        ):
            metric_line(
                str(year),
                group,
            )

    print()
    print("=" * 112)
    print("FINAL SUMMARY")
    print("=" * 112)
    print(
        f"Rules frozen: Supertrend "
        f"{SUPERTREND_LENGTH}/"
        f"{SUPERTREND_MULTIPLIER}, "
        f"ADX > {ADX_MIN} + Rising, "
        f"SL {STOP_ATR} ATR, "
        f"TP {TARGET_ATR} ATR"
    )

    print(
        f"Full history candles: "
        f"{len(full_df)}"
    )

    print(
        f"Full history from: "
        f"{full_df.index.min()}"
    )

    print(
        f"Full history to:   "
        f"{full_df.index.max()}"
    )

    print(
        f"Overall PF:        "
        f"{overall['profit_factor']:.3f}"
    )

    print(
        f"Overall expectancy:"
        f" {overall['expectancy']:.4f}%"
    )

    print("=" * 112)

    trades.to_csv(
        "c2g_v13_full_history_trades.csv",
        index=False,
    )

    pd.DataFrame(
        yearly_rows
    ).to_csv(
        "c2g_v13_yearly_results.csv",
        index=False,
    )

    print()
    print(
        "Trades salvos em: "
        "c2g_v13_full_history_trades.csv"
    )

    print(
        "Resultados anuais salvos em: "
        "c2g_v13_yearly_results.csv"
    )


if __name__ == "__main__":
    df = prepare_data()

    trades = run_backtest(df)

    report(
        df,
        trades,
    )
    