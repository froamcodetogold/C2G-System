import pandas as pd
import pandas_ta as ta
import numpy as np


# ============================================================
# C2G SYSTEM PRO - V1.9 ROBUST TRIGGER RESEARCH
#
# Objetivo:
# Reduzir sensibilidade entre Binance Spot e Bybit Perpetual
# sem otimizar dezenas de parâmetros.
#
# A V1.8 mostrou:
# - regime BULL/BEAR é muito estável entre feeds;
# - sinais BUY ficam mais parecidos com tolerância de 1 candle;
# - sinais SELL continuam mais frágeis;
# - ADX Rising de 1 candle é sensível;
# - ATR Ratio com limites rígidos piorou a robustez.
#
# Por isso V1.9:
# - REMOVE o filtro ATR Ratio;
# - mantém o filtro de regime;
# - testa formas mais robustas de força do ADX;
# - testa uma janela curta de confirmação após o flip do Supertrend.
#
# Experimentos:
#
# BENCHMARK
# V1.5A_CURRENT
#   Flip exato
#   ADX > 25
#   ADX atual > ADX anterior
#   BUY BULL / SELL BEAR
#
# V1.9A_ADX3_SLOPE
#   Flip exato
#   ADX > 25
#   ADX atual > ADX de 3 candles atrás
#   BUY BULL / SELL BEAR
#
# V1.9B_FLIP2_ADX3
#   Permite confirmação no candle do flip OU no próximo candle
#   ADX > 25
#   ADX atual > ADX de 3 candles atrás
#   BUY BULL / SELL BEAR
#
# V1.9C_FLIP2_ADX_2OF3
#   Permite confirmação no candle do flip OU no próximo candle
#   ADX > 25
#   Pelo menos 2 das últimas 3 variações do ADX são positivas
#   BUY BULL / SELL BEAR
#
# Ranking:
# O script NÃO escolhe pelo maior PF de um único mercado.
# Ele ordena pelo PIOR Profit Factor entre Binance e Bybit
# no MESMO período. Isso favorece robustez cross-market.
#
# Custos:
# O ranking principal usa resultado bruto.
# Depois roda um stress test fixo de custos na melhor versão.
#
# IMPORTANTE:
# Como já usamos resultados da Bybit para desenhar esta V1.9,
# a Bybit deixa de ser um holdout puro a partir daqui.
# Uma futura versão congelada deverá ser validada em outra fonte
# ou em forward test.
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

# Stress assumptions fixed before seeing V1.9 results.
STRESS_FEE_PCT_PER_SIDE = 0.055
STRESS_SLIPPAGE_PCT_PER_SIDE = 0.020


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
                f"{path}: coluna '{col}' não encontrada."
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

    rising_flags = (
        df["ADX_Diff1"] > 0
    ).astype(int)

    df["ADX_Rising_Count_3"] = (
        rising_flags
        .rolling(
            window=3,
            min_periods=3,
        )
        .sum()
    )

    df["ADX_2of3"] = (
        df["ADX_Rising_Count_3"]
        >= 2
    )

    # ---------------- ATR ----------------
    df["ATR"] = ta.atr(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        length=ATR_LENGTH,
    )

    # ---------------- EMA200 REGIME ----------------
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
# EVENT SIGNAL BUILDER
# ============================================================

def first_confirmation_after_flip(
    df,
    flip_col,
    condition,
    window_bars,
):
    """
    Para cada flip, cria NO MÁXIMO um sinal.

    window_bars=1:
      apenas candle do flip.

    window_bars=2:
      candle do flip ou candle imediatamente seguinte.

    O sinal é emitido no PRIMEIRO candle que satisfaz condition.
    Não usa informação futura para antecipar sinal:
    se a confirmação só aparece no candle seguinte, o sinal
    só existe naquele candle seguinte.
    """

    signal = pd.Series(
        False,
        index=df.index,
    )

    flip_positions = np.flatnonzero(
        df[
            flip_col
        ].fillna(False).to_numpy()
    )

    cond_values = (
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
            if cond_values[j]:
                signal.iloc[j] = True
                break

    return signal


def build_variant(
    base,
    variant,
):
    df = base.copy()

    buy_regime = (
        df["Regime"] == "BULL"
    )

    sell_regime = (
        df["Regime"] == "BEAR"
    )

    adx_base = (
        df["ADX"] > ADX_MIN
    )

    if variant == "V1.5A_CURRENT":
        adx_condition = (
            adx_base
            &
            df["ADX_Rising_1"]
        )

        df["Buy_Signal"] = (
            df["ST_Buy_Flip"]
            &
            adx_condition
            &
            buy_regime
        )

        df["Sell_Signal"] = (
            df["ST_Sell_Flip"]
            &
            adx_condition
            &
            sell_regime
        )

    elif variant == "V1.9A_ADX3_SLOPE":
        adx_condition = (
            adx_base
            &
            df["ADX_Slope_3"]
        )

        df["Buy_Signal"] = (
            df["ST_Buy_Flip"]
            &
            adx_condition
            &
            buy_regime
        )

        df["Sell_Signal"] = (
            df["ST_Sell_Flip"]
            &
            adx_condition
            &
            sell_regime
        )

    elif variant == "V1.9B_FLIP2_ADX3":
        buy_condition = (
            adx_base
            &
            df["ADX_Slope_3"]
            &
            buy_regime
            &
            (
                df["Trend_Direction"] == 1
            )
        )

        sell_condition = (
            adx_base
            &
            df["ADX_Slope_3"]
            &
            sell_regime
            &
            (
                df["Trend_Direction"] == -1
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

    elif variant == "V1.9C_FLIP2_ADX_2OF3":
        buy_condition = (
            adx_base
            &
            df["ADX_2of3"]
            &
            buy_regime
            &
            (
                df["Trend_Direction"] == 1
            )
        )

        sell_condition = (
            adx_base
            &
            df["ADX_2of3"]
            &
            sell_regime
            &
            (
                df["Trend_Direction"] == -1
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
            f"Variant desconhecida: {variant}"
        )

    df["Buy_Signal"] = (
        df["Buy_Signal"]
        .fillna(False)
        .astype(int)
    )

    df["Sell_Signal"] = (
        df["Sell_Signal"]
        .fillna(False)
        .astype(int)
    )

    return df


# ============================================================
# BACKTEST
# ============================================================

def calculate_pnl(
    side,
    entry_price,
    exit_price,
    fee_pct=0.0,
    slippage_pct=0.0,
):
    if side == "BUY":
        gross = (
            (
                exit_price
                -
                entry_price
            )
            /
            entry_price
        ) * 100.0
    else:
        gross = (
            (
                entry_price
                -
                exit_price
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


def run_backtest(
    df,
    fee_pct=0.0,
    slippage_pct=0.0,
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
        # GAP MANAGEMENT
        # ====================================================
        if position == "BUY":
            if o <= stop:
                pnl = calculate_pnl(
                    "BUY",
                    entry_price,
                    o,
                    fee_pct,
                    slippage_pct,
                )

                trades.append({
                    "side": "BUY",
                    "entry_time": entry_time,
                    "exit_time": df.index[i],
                    "entry_price": entry_price,
                    "exit_price": o,
                    "reason": "STOP_GAP",
                    "pnl_pct": pnl,
                })

                position = None

            elif o >= target:
                pnl = calculate_pnl(
                    "BUY",
                    entry_price,
                    o,
                    fee_pct,
                    slippage_pct,
                )

                trades.append({
                    "side": "BUY",
                    "entry_time": entry_time,
                    "exit_time": df.index[i],
                    "entry_price": entry_price,
                    "exit_price": o,
                    "reason": "TARGET_GAP",
                    "pnl_pct": pnl,
                })

                position = None

        elif position == "SELL":
            if o >= stop:
                pnl = calculate_pnl(
                    "SELL",
                    entry_price,
                    o,
                    fee_pct,
                    slippage_pct,
                )

                trades.append({
                    "side": "SELL",
                    "entry_time": entry_time,
                    "exit_time": df.index[i],
                    "entry_price": entry_price,
                    "exit_price": o,
                    "reason": "STOP_GAP",
                    "pnl_pct": pnl,
                })

                position = None

            elif o <= target:
                pnl = calculate_pnl(
                    "SELL",
                    entry_price,
                    o,
                    fee_pct,
                    slippage_pct,
                )

                trades.append({
                    "side": "SELL",
                    "entry_time": entry_time,
                    "exit_time": df.index[i],
                    "entry_price": entry_price,
                    "exit_price": o,
                    "reason": "TARGET_GAP",
                    "pnl_pct": pnl,
                })

                position = None

        # ====================================================
        # ENTRY FROM PREVIOUS CLOSED BAR
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
                    ] == 1
                ):
                    position = "BUY"

                    entry_time = (
                        df.index[i]
                    )

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

                elif (
                    previous[
                        "Sell_Signal"
                    ] == 1
                ):
                    position = "SELL"

                    entry_time = (
                        df.index[i]
                    )

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
        # INTRABAR MANAGEMENT
        # ====================================================
        if position == "BUY":
            exit_price = None
            reason = None

            if l <= stop:
                exit_price = stop
                reason = "STOP"

            elif h >= target:
                exit_price = target
                reason = (
                    "TAKE_PROFIT"
                )

            elif (
                current[
                    "Trend_Direction"
                ] == -1
            ):
                exit_price = c
                reason = (
                    "TREND_REVERSAL"
                )

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
            exit_price = None
            reason = None

            if h >= stop:
                exit_price = stop
                reason = "STOP"

            elif l <= target:
                exit_price = target
                reason = (
                    "TAKE_PROFIT"
                )

            elif (
                current[
                    "Trend_Direction"
                ] == 1
            ):
                exit_price = c
                reason = (
                    "TREND_REVERSAL"
                )

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
            trades[
                "entry_time"
            ]
        )

        trades[
            "exit_time"
        ] = pd.to_datetime(
            trades[
                "exit_time"
            ]
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
            "win_rate": 0.0,
            "pnl": 0.0,
            "pf": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
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

    gross_profit = (
        float(
            wins["pnl_pct"].sum()
        )
        if len(wins)
        else 0.0
    )

    gross_loss = (
        abs(
            float(
                losses["pnl_pct"].sum()
            )
        )
        if len(losses)
        else 0.0
    )

    pf = (
        gross_profit
        /
        gross_loss
        if gross_loss > 0
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
        "max_drawdown": float(
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
        f"DD {m['max_drawdown']:>7.2f}% | "
        f"LStreak {m['loss_streak']:>2}"
    )

    return m


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 132)
    print("C2G V1.9 - ROBUST TRIGGER RESEARCH")
    print("=" * 132)

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

    # Same-period datasets.
    binance_same_calc = (
        binance_raw.loc[
            warmup_start:overlap_end
        ].copy()
    )

    bybit_same_calc = (
        bybit_raw.loc[
            warmup_start:overlap_end
        ].copy()
    )

    binance_same_calc = (
        prepare_indicators(
            binance_same_calc
        )
    )

    bybit_same_calc = (
        prepare_indicators(
            bybit_same_calc
        )
    )

    binance_same = (
        binance_same_calc.loc[
            overlap_start:overlap_end
        ].copy()
    )

    bybit_same = (
        bybit_same_calc.loc[
            overlap_start:overlap_end
        ].copy()
    )

    # Binance full history only for context.
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
        f"Binance same-period candles: "
        f"{len(binance_same)}"
    )

    print(
        f"Bybit same-period candles:   "
        f"{len(bybit_same)}"
    )

    print(
        f"Binance full candles:         "
        f"{len(binance_full)}"
    )

    variants = [
        "V1.5A_CURRENT",
        "V1.9A_ADX3_SLOPE",
        "V1.9B_FLIP2_ADX3",
        "V1.9C_FLIP2_ADX_2OF3",
    ]

    rows = []
    saved = {}

    for variant in variants:

        b_same_df = build_variant(
            binance_same,
            variant,
        )

        y_same_df = build_variant(
            bybit_same,
            variant,
        )

        b_full_df = build_variant(
            binance_full,
            variant,
        )

        b_same_trades = run_backtest(
            b_same_df
        )

        y_same_trades = run_backtest(
            y_same_df
        )

        b_full_trades = run_backtest(
            b_full_df
        )

        print()
        print("=" * 132)
        print(variant)
        print("=" * 132)

        print(
            f"Signals Binance same: "
            f"BUY {int(b_same_df['Buy_Signal'].sum())} | "
            f"SELL {int(b_same_df['Sell_Signal'].sum())}"
        )

        print(
            f"Signals Bybit same:   "
            f"BUY {int(y_same_df['Buy_Signal'].sum())} | "
            f"SELL {int(y_same_df['Sell_Signal'].sum())}"
        )

        m_binance = metric_line(
            "BINANCE SAME PERIOD",
            b_same_trades,
        )

        m_bybit = metric_line(
            "BYBIT SAME PERIOD",
            y_same_trades,
        )

        m_full = metric_line(
            "BINANCE FULL HISTORY",
            b_full_trades,
        )

        print()
        metric_line(
            "BINANCE SAME BUY",
            b_same_trades[
                b_same_trades[
                    "side"
                ] == "BUY"
            ],
        )

        metric_line(
            "BINANCE SAME SELL",
            b_same_trades[
                b_same_trades[
                    "side"
                ] == "SELL"
            ],
        )

        metric_line(
            "BYBIT SAME BUY",
            y_same_trades[
                y_same_trades[
                    "side"
                ] == "BUY"
            ],
        )

        metric_line(
            "BYBIT SAME SELL",
            y_same_trades[
                y_same_trades[
                    "side"
                ] == "SELL"
            ],
        )

        min_pf = min(
            m_binance["pf"],
            m_bybit["pf"],
        )

        min_exp = min(
            m_binance["expectancy"],
            m_bybit["expectancy"],
        )

        both_profitable = (
            m_binance["pf"] > 1.0
            and
            m_bybit["pf"] > 1.0
            and
            m_binance["expectancy"] > 0
            and
            m_bybit["expectancy"] > 0
        )

        min_trades = min(
            m_binance["trades"],
            m_bybit["trades"],
        )

        rows.append({
            "variant": variant,

            "binance_trades": (
                m_binance[
                    "trades"
                ]
            ),
            "binance_pf": (
                m_binance[
                    "pf"
                ]
            ),
            "binance_exp": (
                m_binance[
                    "expectancy"
                ]
            ),
            "binance_pnl": (
                m_binance[
                    "pnl"
                ]
            ),
            "binance_dd": (
                m_binance[
                    "max_drawdown"
                ]
            ),

            "bybit_trades": (
                m_bybit[
                    "trades"
                ]
            ),
            "bybit_pf": (
                m_bybit[
                    "pf"
                ]
            ),
            "bybit_exp": (
                m_bybit[
                    "expectancy"
                ]
            ),
            "bybit_pnl": (
                m_bybit[
                    "pnl"
                ]
            ),
            "bybit_dd": (
                m_bybit[
                    "max_drawdown"
                ]
            ),

            "binance_full_pf": (
                m_full[
                    "pf"
                ]
            ),
            "binance_full_exp": (
                m_full[
                    "expectancy"
                ]
            ),

            "min_pf_cross_market": min_pf,
            "min_exp_cross_market": min_exp,
            "min_trades_cross_market": min_trades,
            "both_markets_positive": both_profitable,
        })

        saved[
            variant
        ] = {
            "binance_same": b_same_trades,
            "bybit_same": y_same_trades,
            "binance_full": b_full_trades,
            "binance_df": b_same_df,
            "bybit_df": y_same_df,
        }

    ranking = pd.DataFrame(
        rows
    )

    ranking = ranking.sort_values(
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
    print("=" * 156)
    print(
        "C2G V1.9 - CROSS-MARKET ROBUSTNESS RANKING"
    )
    print("=" * 156)

    display_cols = [
        "variant",
        "binance_trades",
        "binance_pf",
        "binance_exp",
        "bybit_trades",
        "bybit_pf",
        "bybit_exp",
        "min_pf_cross_market",
        "min_exp_cross_market",
        "min_trades_cross_market",
        "both_markets_positive",
        "binance_full_pf",
    ]

    print(
        ranking[
            display_cols
        ].to_string(
            index=False,
            formatters={
                "binance_pf": "{:.3f}".format,
                "binance_exp": "{:.4f}%".format,
                "bybit_pf": "{:.3f}".format,
                "bybit_exp": "{:.4f}%".format,
                "min_pf_cross_market": "{:.3f}".format,
                "min_exp_cross_market": "{:.4f}%".format,
                "binance_full_pf": "{:.3f}".format,
            },
        )
    )

    print("=" * 156)

    # ========================================================
    # STRESS TEST ONLY THE TOP-RANKED VARIANT
    # ========================================================

    best_variant = ranking.iloc[
        0
    ]["variant"]

    best_binance_df = saved[
        best_variant
    ]["binance_df"]

    best_bybit_df = saved[
        best_variant
    ]["bybit_df"]

    print()
    print("=" * 132)
    print(
        f"STRESS TEST DE CUSTOS - {best_variant}"
    )
    print("=" * 132)

    best_binance_cost = run_backtest(
        best_binance_df,
        fee_pct=STRESS_FEE_PCT_PER_SIDE,
        slippage_pct=STRESS_SLIPPAGE_PCT_PER_SIDE,
    )

    best_bybit_cost = run_backtest(
        best_bybit_df,
        fee_pct=STRESS_FEE_PCT_PER_SIDE,
        slippage_pct=STRESS_SLIPPAGE_PCT_PER_SIDE,
    )

    metric_line(
        "BINANCE COST STRESS",
        best_binance_cost,
    )

    metric_line(
        "BYBIT COST STRESS",
        best_bybit_cost,
    )

    # ========================================================
    # SAVE
    # ========================================================

    ranking.to_csv(
        "c2g_v19_cross_market_ranking.csv",
        index=False,
    )

    saved[
        best_variant
    ]["binance_same"].to_csv(
        "c2g_v19_best_binance_gross_trades.csv",
        index=False,
    )

    saved[
        best_variant
    ]["bybit_same"].to_csv(
        "c2g_v19_best_bybit_gross_trades.csv",
        index=False,
    )

    best_binance_cost.to_csv(
        "c2g_v19_best_binance_cost_trades.csv",
        index=False,
    )

    best_bybit_cost.to_csv(
        "c2g_v19_best_bybit_cost_trades.csv",
        index=False,
    )

    print()
    print("=" * 132)
    print("ARQUIVOS GERADOS")
    print("=" * 132)
    print(
        "c2g_v19_cross_market_ranking.csv"
    )
    print(
        "c2g_v19_best_binance_gross_trades.csv"
    )
    print(
        "c2g_v19_best_bybit_gross_trades.csv"
    )
    print(
        "c2g_v19_best_binance_cost_trades.csv"
    )
    print(
        "c2g_v19_best_bybit_cost_trades.csv"
    )

    print()
    print("=" * 132)
    print("IMPORTANTE")
    print("=" * 132)
    print(
        "A Bybit foi usada para diagnosticar e agora também para "
        "comparar variantes. Portanto ela NÃO é mais um holdout puro."
    )
    print(
        "Se uma V1.9 sobreviver nos dois feeds, o próximo passo deve "
        "ser congelar as regras e validar em outra fonte ou forward test."
    )
    print("=" * 132)
