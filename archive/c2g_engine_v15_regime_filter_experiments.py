import pandas as pd
import pandas_ta as ta
import numpy as np


# ============================================================
# C2G SYSTEM PRO - V1.5 REGIME FILTER EXPERIMENTS
#
# Base congelada:
# - Supertrend 10 / 3
# - ADX > 25
# - ADX Rising
# - Stop = 1.5 ATR
# - Take Profit = 3 ATR
#
# Experimentos:
# 0) Baseline V1.2C (sem filtro de regime)
# A) BUY apenas em BULL / SELL apenas em BEAR
# B) Regime + ATR Ratio entre 0.80 e 1.20
# C) Regime + ATR Ratio entre 1.00 e 1.20
#
# Regime:
# BULL:
#   Close > EMA200
#   EMA200 atual > EMA200 de 50 candles atrás
#
# BEAR:
#   Close < EMA200
#   EMA200 atual < EMA200 de 50 candles atrás
#
# NEUTRAL:
#   todo o restante
#
# IMPORTANTE:
# O filtro é aplicado no candle ANTERIOR fechado.
# A entrada ocorre no OPEN do candle seguinte.
# ============================================================


FILE_PATH = "btc_data_binance_full_1h.csv"

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

FEE_PCT = 0.0
SLIPPAGE_PCT = 0.0

REFERENCE_SPLIT_DATE = pd.Timestamp("2022-08-15 00:00:00")


# ============================================================
# AUXILIARES
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

    total_cost_pct = (
        fee_pct * 2.0
        +
        slippage_pct * 2.0
    )

    return gross_pct - total_cost_pct


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
            best = max(
                best,
                current,
            )
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

    ordered = trades_df.sort_values(
        "entry_time"
    ).copy()

    winners = ordered[
        ordered["pnl_pct"] > 0
    ]

    losers = ordered[
        ordered["pnl_pct"] < 0
    ]

    total = len(ordered)
    wins = len(winners)
    losses = len(losers)

    gross_profit = (
        float(
            winners["pnl_pct"].sum()
        )
        if wins
        else 0.0
    )

    gross_loss = (
        abs(
            float(
                losers["pnl_pct"].sum()
            )
        )
        if losses
        else 0.0
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    equity = build_equity_curve(
        ordered["pnl_pct"].tolist()
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
            ordered["pnl_pct"].sum()
        ),
        "expectancy": float(
            ordered["pnl_pct"].mean()
        ),
        "avg_win": (
            float(
                winners["pnl_pct"].mean()
            )
            if wins
            else 0.0
        ),
        "avg_loss": (
            float(
                losers["pnl_pct"].mean()
            )
            if losses
            else 0.0
        ),
        "profit_factor": profit_factor,
        "max_drawdown": float(
            drawdown.min()
        ),
        "max_loss_streak": (
            max_consecutive_losses(
                ordered["pnl_pct"].tolist()
            )
        ),
        "ending_equity": float(
            equity.iloc[-1]
        ),
    }


def pf_text(value):
    if np.isinf(value):
        return "inf"

    return f"{value:.3f}"


def metric_line(label, trades_df):
    m = calculate_metrics(
        trades_df
    )

    print(
        f"{label:<30} | "
        f"Trades {m['trades']:>4} | "
        f"WR {m['win_rate']:>6.2f}% | "
        f"PnL {m['pnl']:>8.2f}% | "
        f"PF {pf_text(m['profit_factor']):>6} | "
        f"Exp {m['expectancy']:>8.4f}% | "
        f"DD {m['max_drawdown']:>7.2f}% | "
        f"LStreak {m['max_loss_streak']:>2}"
    )

    return m


# ============================================================
# PREPARAR INDICADORES
# ============================================================

def prepare_data():
    print()
    print("=" * 108)
    print("C2G V1.5 - REGIME FILTER EXPERIMENTS")
    print("=" * 108)
    print("Carregando histórico...")

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

    print(f"Candles:   {len(df)}")
    print(f"Primeiro:  {df.index.min()}")
    print(f"Último:    {df.index.max()}")

    # ---------------- SUPERTREND ----------------
    st = ta.supertrend(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        length=SUPERTREND_LENGTH,
        multiplier=SUPERTREND_MULTIPLIER,
    )

    st_dir_col = find_column(
        st.columns,
        "SUPERTd_",
    )

    st_line_col = find_column(
        st.columns,
        "SUPERT_",
    )

    df["Trend_Direction"] = st[
        st_dir_col
    ]

    df["Trend_Line"] = st[
        st_line_col
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

    adx_col = find_column(
        adx_df.columns,
        "ADX_",
    )

    df["ADX"] = adx_df[
        adx_col
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
        .rolling(ATR_MA_LENGTH)
        .mean()
    )

    df["ATR_Ratio"] = (
        df["ATR"]
        /
        df["ATR_MA"]
    )

    # ---------------- EMA200 / REGIME ----------------
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

    df["Market_Regime"] = "NEUTRAL"

    df.loc[
        bull,
        "Market_Regime"
    ] = "BULL"

    df.loc[
        bear,
        "Market_Regime"
    ] = "BEAR"

    # ---------------- BASE SIGNAL V1.2C ----------------
    df["Base_Buy"] = (
        (df["Signal"] == 2)
        &
        (df["ADX"] > ADX_MIN)
        &
        (df["ADX_Rising"])
    )

    df["Base_Sell"] = (
        (df["Signal"] == -2)
        &
        (df["ADX"] > ADX_MIN)
        &
        (df["ADX_Rising"])
    )

    print(
        f"Base BUY signals:  "
        f"{int(df['Base_Buy'].sum())}"
    )

    print(
        f"Base SELL signals: "
        f"{int(df['Base_Sell'].sum())}"
    )

    return df


# ============================================================
# CONSTRUIR SINAIS DE CADA EXPERIMENTO
# ============================================================

def build_signals(
    base_df,
    experiment,
):
    df = base_df.copy()

    buy = df["Base_Buy"].copy()
    sell = df["Base_Sell"].copy()

    # --------------------------------------------------------
    # A partir de V1.5A:
    # BUY somente em BULL
    # SELL somente em BEAR
    # --------------------------------------------------------
    if experiment["use_regime_filter"]:
        buy = (
            buy
            &
            (
                df["Market_Regime"]
                == "BULL"
            )
        )

        sell = (
            sell
            &
            (
                df["Market_Regime"]
                == "BEAR"
            )
        )

    # --------------------------------------------------------
    # Filtro opcional ATR Ratio
    # --------------------------------------------------------
    atr_min = experiment.get(
        "atr_min"
    )

    atr_max = experiment.get(
        "atr_max"
    )

    if atr_min is not None:
        buy = (
            buy
            &
            (
                df["ATR_Ratio"]
                >= atr_min
            )
        )

        sell = (
            sell
            &
            (
                df["ATR_Ratio"]
                >= atr_min
            )
        )

    if atr_max is not None:
        buy = (
            buy
            &
            (
                df["ATR_Ratio"]
                < atr_max
            )
        )

        sell = (
            sell
            &
            (
                df["ATR_Ratio"]
                < atr_max
            )
        )

    df["Buy_Signal"] = (
        buy.astype(int)
    )

    df["Sell_Signal"] = (
        sell.astype(int)
    )

    return df


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(df):
    trades = []

    position = None

    entry_price = None
    entry_time = None

    stop_loss = None
    take_profit = None

    entry_regime = None
    entry_atr_ratio = None
    entry_adx = None

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
        # ENTRY
        # ====================================================
        if position is None:
            previous_atr = previous[
                "ATR"
            ]

            if (
                pd.notna(previous_atr)
                and previous_atr > 0
            ):
                side = None

                if (
                    previous["Buy_Signal"]
                    == 1
                ):
                    side = "BUY"

                elif (
                    previous["Sell_Signal"]
                    == 1
                ):
                    side = "SELL"

                if side is not None:
                    position = side

                    entry_time = (
                        df.index[i]
                    )

                    entry_price = (
                        current_open
                    )

                    entry_regime = (
                        previous[
                            "Market_Regime"
                        ]
                    )

                    entry_atr_ratio = (
                        float(
                            previous[
                                "ATR_Ratio"
                            ]
                        )
                        if pd.notna(
                            previous[
                                "ATR_Ratio"
                            ]
                        )
                        else np.nan
                    )

                    entry_adx = (
                        float(
                            previous["ADX"]
                        )
                        if pd.notna(
                            previous["ADX"]
                        )
                        else np.nan
                    )

                    if side == "BUY":
                        stop_loss = (
                            entry_price
                            -
                            STOP_ATR
                            * previous_atr
                        )

                        take_profit = (
                            entry_price
                            +
                            TARGET_ATR
                            * previous_atr
                        )

                    else:
                        stop_loss = (
                            entry_price
                            +
                            STOP_ATR
                            * previous_atr
                        )

                        take_profit = (
                            entry_price
                            -
                            TARGET_ATR
                            * previous_atr
                        )

        # ====================================================
        # MANAGE BUY
        # ====================================================
        if position == "BUY":
            exit_price = None
            reason = None

            stop_hit = (
                current_low
                <= stop_loss
            )

            target_hit = (
                current_high
                >= take_profit
            )

            # Conservador:
            # se SL e TP aparecem no mesmo candle,
            # considera STOP primeiro.
            if stop_hit:
                exit_price = (
                    stop_loss
                )

                reason = "STOP"

            elif target_hit:
                exit_price = (
                    take_profit
                )

                reason = (
                    "TAKE_PROFIT"
                )

            elif (
                current[
                    "Trend_Direction"
                ]
                == -1
            ):
                exit_price = (
                    current_close
                )

                reason = (
                    "TREND_REVERSAL"
                )

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
                    "entry_regime": entry_regime,
                    "entry_atr_ratio": entry_atr_ratio,
                    "entry_adx": entry_adx,
                })

                position = None

        # ====================================================
        # MANAGE SELL
        # ====================================================
        elif position == "SELL":
            exit_price = None
            reason = None

            stop_hit = (
                current_high
                >= stop_loss
            )

            target_hit = (
                current_low
                <= take_profit
            )

            if stop_hit:
                exit_price = (
                    stop_loss
                )

                reason = "STOP"

            elif target_hit:
                exit_price = (
                    take_profit
                )

                reason = (
                    "TAKE_PROFIT"
                )

            elif (
                current[
                    "Trend_Direction"
                ]
                == 1
            ):
                exit_price = (
                    current_close
                )

                reason = (
                    "TREND_REVERSAL"
                )

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
                    "entry_regime": entry_regime,
                    "entry_atr_ratio": entry_atr_ratio,
                    "entry_adx": entry_adx,
                })

                position = None

    # Fecha posição restante no fim dos dados.
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
            "entry_regime": entry_regime,
            "entry_atr_ratio": entry_atr_ratio,
            "entry_adx": entry_adx,
        })

    trades_df = pd.DataFrame(
        trades
    )

    if not trades_df.empty:
        trades_df[
            "entry_time"
        ] = pd.to_datetime(
            trades_df["entry_time"]
        )

        trades_df[
            "exit_time"
        ] = pd.to_datetime(
            trades_df["exit_time"]
        )

        trades_df[
            "year"
        ] = (
            trades_df[
                "entry_time"
            ].dt.year
        )

    return trades_df


# ============================================================
# RELATÓRIO DE EXPERIMENTO
# ============================================================

def print_experiment_report(
    name,
    trades,
):
    print()
    print("=" * 118)
    print(name)
    print("=" * 118)

    total_metrics = metric_line(
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

    older = trades[
        trades["entry_time"]
        <
        REFERENCE_SPLIT_DATE
    ]

    recent = trades[
        trades["entry_time"]
        >=
        REFERENCE_SPLIT_DATE
    ]

    metric_line(
        "2017 -> 2022-08",
        older,
    )

    metric_line(
        "2022-08 -> 2026",
        recent,
    )

    print()
    print("RESULTADO POR ANO")
    print("-" * 118)

    yearly_rows = []

    for year, group in trades.groupby(
        "year"
    ):
        m = metric_line(
            str(year),
            group,
        )

        yearly_rows.append({
            "year": year,
            **m,
        })

    return (
        total_metrics,
        yearly_rows,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    base_df = prepare_data()

    experiments = [
        {
            "name": (
                "V1.2C BASELINE - "
                "ADX25 + Rising"
            ),
            "use_regime_filter": False,
            "atr_min": None,
            "atr_max": None,
        },

        {
            "name": (
                "V1.5A - REGIME DIRECTION"
            ),
            "use_regime_filter": True,
            "atr_min": None,
            "atr_max": None,
        },

        {
            "name": (
                "V1.5B - REGIME + "
                "ATR 0.80-1.20"
            ),
            "use_regime_filter": True,
            "atr_min": 0.80,
            "atr_max": 1.20,
        },

        {
            "name": (
                "V1.5C - REGIME + "
                "ATR 1.00-1.20"
            ),
            "use_regime_filter": True,
            "atr_min": 1.00,
            "atr_max": 1.20,
        },
    ]

    ranking_rows = []
    saved = {}

    for exp in experiments:
        test_df = build_signals(
            base_df,
            exp,
        )

        print()
        print(
            f"{exp['name']} | "
            f"BUY signals: "
            f"{int(test_df['Buy_Signal'].sum())} | "
            f"SELL signals: "
            f"{int(test_df['Sell_Signal'].sum())}"
        )

        trades = run_backtest(
            test_df
        )

        metrics, yearly_rows = (
            print_experiment_report(
                exp["name"],
                trades,
            )
        )

        ranking_rows.append({
            "strategy": exp["name"],
            **metrics,
        })

        saved[
            exp["name"]
        ] = {
            "trades": trades,
            "yearly": yearly_rows,
        }

    # ========================================================
    # RANKING
    # ========================================================

    ranking = pd.DataFrame(
        ranking_rows
    )

    ranking = ranking.sort_values(
        by=[
            "profit_factor",
            "expectancy",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    print()
    print("=" * 128)
    print(
        "RANKING FINAL C2G V1.5 "
        "- ORDENADO POR PROFIT FACTOR"
    )
    print("=" * 128)

    columns = [
        "strategy",
        "trades",
        "win_rate",
        "pnl",
        "profit_factor",
        "expectancy",
        "max_drawdown",
        "max_loss_streak",
        "ending_equity",
    ]

    print(
        ranking[
            columns
        ].to_string(
            index=False,
            formatters={
                "win_rate": (
                    "{:.2f}%".format
                ),
                "pnl": (
                    "{:.2f}%".format
                ),
                "profit_factor": (
                    "{:.3f}".format
                ),
                "expectancy": (
                    "{:.4f}%".format
                ),
                "max_drawdown": (
                    "{:.2f}%".format
                ),
                "ending_equity": (
                    "{:.2f}".format
                ),
            },
        )
    )

    print("=" * 128)

    # ========================================================
    # SALVAR MELHOR
    # ========================================================

    best_name = ranking.iloc[
        0
    ]["strategy"]

    best_trades = saved[
        best_name
    ]["trades"]

    ranking.to_csv(
        "c2g_v15_ranking.csv",
        index=False,
    )

    best_trades.to_csv(
        "c2g_v15_best_trades.csv",
        index=False,
    )

    pd.DataFrame(
        saved[
            best_name
        ]["yearly"]
    ).to_csv(
        "c2g_v15_best_yearly.csv",
        index=False,
    )

    print()
    print(
        f"Melhor versão: {best_name}"
    )

    print(
        "Ranking salvo em: "
        "c2g_v15_ranking.csv"
    )

    print(
        "Trades da melhor versão: "
        "c2g_v15_best_trades.csv"
    )

    print(
        "Resultado anual da melhor versão: "
        "c2g_v15_best_yearly.csv"
    )

    print()
    print("=" * 128)
    print("LEMBRETE DE PESQUISA")
    print("=" * 128)
    print(
        "Os filtros de regime/ATR foram escolhidos "
        "depois de analisar este mesmo histórico."
    )
    print(
        "Portanto, uma melhora aqui é uma hipótese "
        "promissora, não uma validação fora da amostra."
    )
    print(
        "A próxima fase deve congelar a vencedora "
        "e validar em dados/mercado não usados para "
        "escolher estes filtros."
    )
    print("=" * 128)
    