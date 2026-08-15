import pandas as pd
import pandas_ta as ta
import numpy as np


# ============================================================
# C2G SYSTEM PRO - V1.4 REGIME DIAGNOSTIC
#
# Objetivo:
# NÃO otimizar a estratégia ainda.
#
# Mantém congeladas as regras da V1.2C:
# - Supertrend 10 / 3
# - ADX > 25
# - ADX Rising
# - Stop = 1.5 ATR
# - Take Profit = 3 ATR
#
# E classifica cada entrada por:
# - BUY / SELL
# - Ano
# - Regime: BULL / BEAR / NEUTRAL
# - Faixa de ADX
# - Faixa de volatilidade ATR Ratio
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
# Caso contrário:
#   NEUTRAL
#
# IMPORTANTE:
# Todas as informações usadas para classificar a entrada vêm
# do candle ANTERIOR já fechado, evitando lookahead.
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


def build_equity_curve(pnl_values, initial_equity=100.0):
    equity = [initial_equity]

    for pnl_pct in pnl_values:
        equity.append(
            equity[-1]
            *
            (1.0 + pnl_pct / 100.0)
        )

    return pd.Series(equity, dtype=float)


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
        "max_loss_streak": max_consecutive_losses(
            trades_df["pnl_pct"].tolist()
        ),
        "ending_equity": float(
            equity.iloc[-1]
        ),
    }


def metric_line(label, trades_df):
    m = calculate_metrics(trades_df)

    pf_text = (
        "inf"
        if np.isinf(m["profit_factor"])
        else f"{m['profit_factor']:.3f}"
    )

    print(
        f"{label:<28} | "
        f"Trades {m['trades']:>4} | "
        f"WR {m['win_rate']:>6.2f}% | "
        f"PnL {m['pnl']:>8.2f}% | "
        f"PF {pf_text:>6} | "
        f"Exp {m['expectancy']:>8.4f}% | "
        f"DD {m['max_drawdown']:>7.2f}%"
    )

    return m


# ============================================================
# CLASSIFICAÇÕES
# ============================================================

def classify_regime(row):
    if (
        pd.isna(row["EMA200"])
        or pd.isna(row["EMA200_Lag"])
    ):
        return "UNKNOWN"

    if (
        row["close"] > row["EMA200"]
        and row["EMA200"] > row["EMA200_Lag"]
    ):
        return "BULL"

    if (
        row["close"] < row["EMA200"]
        and row["EMA200"] < row["EMA200_Lag"]
    ):
        return "BEAR"

    return "NEUTRAL"


def classify_adx(adx):
    if pd.isna(adx):
        return "UNKNOWN"

    if adx < 25:
        return "<25"

    if adx < 30:
        return "25-30"

    if adx < 35:
        return "30-35"

    if adx < 40:
        return "35-40"

    return "40+"


def classify_atr_ratio(ratio):
    if pd.isna(ratio):
        return "UNKNOWN"

    if ratio < 0.80:
        return "<0.80"

    if ratio < 1.00:
        return "0.80-1.00"

    if ratio < 1.20:
        return "1.00-1.20"

    return "1.20+"


# ============================================================
# PREPARAR DADOS
# ============================================================

def prepare_data():
    print()
    print("=" * 96)
    print("C2G V1.4 - REGIME DIAGNOSTIC")
    print("=" * 96)
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

    # ---------------- EMA 200 ----------------
    df["EMA200"] = ta.ema(
        df["close"],
        length=EMA_LENGTH,
    )

    df["EMA200_Lag"] = (
        df["EMA200"]
        .shift(EMA_SLOPE_LOOKBACK)
    )

    # Regime calculado no próprio candle.
    # Na entrada usaremos sempre o regime do candle anterior.
    df["Market_Regime"] = df.apply(
        classify_regime,
        axis=1,
    )

    # ---------------- V1.2C FROZEN ----------------
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


# ============================================================
# BACKTEST COM METADADOS DO REGIME
# ============================================================

def run_backtest(df):
    trades = []

    position = None

    entry_price = None
    entry_time = None

    stop_loss = None
    take_profit = None

    entry_metadata = None

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
        # ENTRADA
        # ====================================================
        if position is None:
            previous_atr = previous["ATR"]

            if (
                pd.notna(previous_atr)
                and previous_atr > 0
            ):

                side = None

                if previous["Buy_Signal"] == 1:
                    side = "BUY"

                elif previous["Sell_Signal"] == 1:
                    side = "SELL"

                if side is not None:
                    position = side
                    entry_time = df.index[i]
                    entry_price = current_open

                    if side == "BUY":
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

                    else:
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

                    # Todos estes dados vêm do candle anterior fechado.
                    entry_metadata = {
                        "signal_time": df.index[i - 1],

                        "entry_year": df.index[i].year,

                        "entry_regime": (
                            previous["Market_Regime"]
                        ),

                        "entry_adx": float(
                            previous["ADX"]
                        ) if pd.notna(
                            previous["ADX"]
                        ) else np.nan,

                        "adx_band": classify_adx(
                            previous["ADX"]
                        ),

                        "entry_atr": float(
                            previous["ATR"]
                        ) if pd.notna(
                            previous["ATR"]
                        ) else np.nan,

                        "atr_ratio": float(
                            previous["ATR_Ratio"]
                        ) if pd.notna(
                            previous["ATR_Ratio"]
                        ) else np.nan,

                        "atr_band": classify_atr_ratio(
                            previous["ATR_Ratio"]
                        ),

                        "signal_close": float(
                            previous["close"]
                        ),

                        "signal_ema200": float(
                            previous["EMA200"]
                        ) if pd.notna(
                            previous["EMA200"]
                        ) else np.nan,

                        "signal_ema200_lag": float(
                            previous["EMA200_Lag"]
                        ) if pd.notna(
                            previous["EMA200_Lag"]
                        ) else np.nan,
                    }

        # ====================================================
        # GERENCIAR BUY
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

            # Conservador:
            # se SL e TP forem tocados no mesmo candle,
            # assume STOP primeiro.
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

                trade = {
                    "side": "BUY",
                    "entry_time": entry_time,
                    "exit_time": df.index[i],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "reason": reason,
                    "pnl_pct": pnl,
                }

                trade.update(
                    entry_metadata
                )

                trades.append(
                    trade
                )

                position = None
                entry_metadata = None

        # ====================================================
        # GERENCIAR SELL
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

                trade = {
                    "side": "SELL",
                    "entry_time": entry_time,
                    "exit_time": df.index[i],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "reason": reason,
                    "pnl_pct": pnl,
                }

                trade.update(
                    entry_metadata
                )

                trades.append(
                    trade
                )

                position = None
                entry_metadata = None

    # Fecha eventual posição restante.
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

        trade = {
            "side": position,
            "entry_time": entry_time,
            "exit_time": df.index[-1],
            "entry_price": entry_price,
            "exit_price": final_price,
            "reason": "END_OF_DATA",
            "pnl_pct": pnl,
        }

        trade.update(
            entry_metadata
        )

        trades.append(
            trade
        )

    return pd.DataFrame(
        trades
    )


# ============================================================
# RELATÓRIOS
# ============================================================

def print_group_report(
    trades,
    group_col,
    title,
):
    print()
    print("=" * 116)
    print(title)
    print("=" * 116)

    if trades.empty:
        print("Sem trades.")
        return []

    rows = []

    for group_name, group in trades.groupby(
        group_col,
        dropna=False,
    ):
        metrics = metric_line(
            str(group_name),
            group,
        )

        rows.append({
            group_col: group_name,
            **metrics,
        })

    return rows


def print_side_regime_report(trades):
    print()
    print("=" * 116)
    print("BUY / SELL x MARKET REGIME")
    print("=" * 116)

    rows = []

    for side in ["BUY", "SELL"]:
        for regime in [
            "BULL",
            "BEAR",
            "NEUTRAL",
            "UNKNOWN",
        ]:
            group = trades[
                (trades["side"] == side)
                &
                (
                    trades["entry_regime"]
                    == regime
                )
            ]

            if group.empty:
                continue

            label = (
                f"{side} in {regime}"
            )

            metrics = metric_line(
                label,
                group,
            )

            rows.append({
                "side": side,
                "regime": regime,
                **metrics,
            })

    return rows


def print_year_side_report(trades):
    print()
    print("=" * 116)
    print("BUY ONLY - RESULT BY YEAR")
    print("=" * 116)

    buy_rows = []

    buys = trades[
        trades["side"] == "BUY"
    ].copy()

    for year, group in buys.groupby(
        "entry_year"
    ):
        metrics = metric_line(
            str(year),
            group,
        )

        buy_rows.append({
            "year": year,
            "side": "BUY",
            **metrics,
        })

    print()
    print("=" * 116)
    print("SELL ONLY - RESULT BY YEAR")
    print("=" * 116)

    sell_rows = []

    sells = trades[
        trades["side"] == "SELL"
    ].copy()

    for year, group in sells.groupby(
        "entry_year"
    ):
        metrics = metric_line(
            str(year),
            group,
        )

        sell_rows.append({
            "year": year,
            "side": "SELL",
            **metrics,
        })

    return buy_rows + sell_rows


def final_report(df, trades):
    print()
    print("=" * 116)
    print("C2G V1.4 - BASELINE FROZEN RESULT")
    print("=" * 116)

    metric_line(
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

    # Ano
    year_rows = print_group_report(
        trades,
        "entry_year",
        "RESULT BY YEAR",
    )

    # BUY por ano / SELL por ano
    year_side_rows = print_year_side_report(
        trades
    )

    # Regime
    regime_rows = print_group_report(
        trades,
        "entry_regime",
        "RESULT BY MARKET REGIME",
    )

    # BUY/SELL x Regime
    side_regime_rows = (
        print_side_regime_report(
            trades
        )
    )

    # Faixa ADX
    adx_rows = print_group_report(
        trades,
        "adx_band",
        "RESULT BY ADX BAND",
    )

    # BUY por faixa ADX
    print_group_report(
        trades[
            trades["side"] == "BUY"
        ],
        "adx_band",
        "BUY ONLY - RESULT BY ADX BAND",
    )

    # SELL por faixa ADX
    print_group_report(
        trades[
            trades["side"] == "SELL"
        ],
        "adx_band",
        "SELL ONLY - RESULT BY ADX BAND",
    )

    # ATR Ratio
    atr_rows = print_group_report(
        trades,
        "atr_band",
        "RESULT BY ATR RATIO BAND",
    )

    # BUY x ATR
    print_group_report(
        trades[
            trades["side"] == "BUY"
        ],
        "atr_band",
        "BUY ONLY - RESULT BY ATR BAND",
    )

    # SELL x ATR
    print_group_report(
        trades[
            trades["side"] == "SELL"
        ],
        "atr_band",
        "SELL ONLY - RESULT BY ATR BAND",
    )

    # ========================================================
    # SALVAR
    # ========================================================

    trades.to_csv(
        "c2g_v14_regime_trades.csv",
        index=False,
    )

    pd.DataFrame(
        year_rows
    ).to_csv(
        "c2g_v14_year_results.csv",
        index=False,
    )

    pd.DataFrame(
        year_side_rows
    ).to_csv(
        "c2g_v14_year_side_results.csv",
        index=False,
    )

    pd.DataFrame(
        regime_rows
    ).to_csv(
        "c2g_v14_regime_results.csv",
        index=False,
    )

    pd.DataFrame(
        side_regime_rows
    ).to_csv(
        "c2g_v14_side_regime_results.csv",
        index=False,
    )

    pd.DataFrame(
        adx_rows
    ).to_csv(
        "c2g_v14_adx_results.csv",
        index=False,
    )

    pd.DataFrame(
        atr_rows
    ).to_csv(
        "c2g_v14_atr_results.csv",
        index=False,
    )

    print()
    print("=" * 116)
    print("ARQUIVOS GERADOS")
    print("=" * 116)
    print("c2g_v14_regime_trades.csv")
    print("c2g_v14_year_results.csv")
    print("c2g_v14_year_side_results.csv")
    print("c2g_v14_regime_results.csv")
    print("c2g_v14_side_regime_results.csv")
    print("c2g_v14_adx_results.csv")
    print("c2g_v14_atr_results.csv")

    print()
    print("=" * 116)
    print("REGRAS CONGELADAS")
    print("=" * 116)
    print(
        f"Supertrend {SUPERTREND_LENGTH}/"
        f"{SUPERTREND_MULTIPLIER}"
    )
    print(
        f"ADX > {ADX_MIN} + Rising"
    )
    print(
        f"Stop = {STOP_ATR} ATR"
    )
    print(
        f"Take Profit = {TARGET_ATR} ATR"
    )
    print(
        f"Regime = EMA{EMA_LENGTH} + "
        f"slope {EMA_SLOPE_LOOKBACK} candles"
    )
    print(
        f"ATR Ratio = ATR / média ATR "
        f"{ATR_MA_LENGTH}"
    )
    print("=" * 116)


if __name__ == "__main__":
    df = prepare_data()

    trades = run_backtest(
        df
    )

    final_report(
        df,
        trades
    )
    