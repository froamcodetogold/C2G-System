import pandas as pd
import pandas_ta as ta
import numpy as np


# ============================================================
# C2G SYSTEM PRO - V1.13 EXTERNAL FEED VALIDATION
#
# REGRA COMPLETAMENTE CONGELADA:
#
# BUY ONLY
# - Supertrend 10 / 3 flip BUY
# - ADX > 25
# - ADX atual > ADX anterior
# - Regime BULL:
#     Close > EMA200
#     EMA200 atual > EMA200 de 50 candles atrás
#
# Entrada:
# - OPEN do candle seguinte
#
# Saída:
# - CLOSE após 24 candles de 1H
#
# Sem SELL.
# Sem SL.
# Sem TP.
#
# Mercados:
# - Binance Spot
# - Bybit Perpetual
# - OKX Perpetual
#
# O ranking principal usa o MESMO período comum aos 3 feeds.
#
# IMPORTANTE:
# OKX é um novo feed/venue, mas o período histórico ainda se
# sobrepõe ao período já estudado. Portanto isto é validação
# externa de feed, NÃO um OOS temporal puro.
# ============================================================


BINANCE_FILE = "btc_data_binance_full_1h.csv"
BYBIT_FILE = "btc_data_bybit_perp_1h.csv"
OKX_FILE = "btc_data_okx_perp_1h.csv"

SUPERTREND_LENGTH = 10
SUPERTREND_MULTIPLIER = 3.0

ADX_LENGTH = 14
ADX_MIN = 25.0

EMA_LENGTH = 200
EMA_SLOPE_LOOKBACK = 50

TIME_EXIT_BARS = 24

STRESS_FEE_PCT_PER_SIDE = 0.055
STRESS_SLIPPAGE_PCT_PER_SIDE = 0.020


# ============================================================
# DATA
# ============================================================

def find_column(columns, prefix):
    matches = [
        col for col in columns
        if str(col).startswith(
            prefix
        )
    ]

    if not matches:
        raise KeyError(
            f"Nenhuma coluna começando "
            f"com '{prefix}'."
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
                f"{path}: coluna "
                f"'{col}' ausente."
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

    st = ta.supertrend(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        length=SUPERTREND_LENGTH,
        multiplier=SUPERTREND_MULTIPLIER,
    )

    df[
        "Trend_Direction"
    ] = st[
        find_column(
            st.columns,
            "SUPERTd_",
        )
    ]

    df["Signal"] = (
        df[
            "Trend_Direction"
        ].diff()
    )

    df[
        "ST_Buy_Flip"
    ] = (
        df["Signal"]
        == 2
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

    df[
        "ADX_Rising"
    ] = (
        df["ADX"]
        >
        df["ADX"].shift(1)
    )

    df[
        "EMA200"
    ] = ta.ema(
        df["close"],
        length=EMA_LENGTH,
    )

    df[
        "EMA200_Lag"
    ] = (
        df["EMA200"]
        .shift(
            EMA_SLOPE_LOOKBACK
        )
    )

    df[
        "Bull_Regime"
    ] = (
        (
            df["close"]
            >
            df["EMA200"]
        )
        &
        (
            df["EMA200"]
            >
            df["EMA200_Lag"]
        )
    )

    df[
        "Buy_Signal"
    ] = (
        df[
            "ST_Buy_Flip"
        ]
        &
        (
            df["ADX"]
            >
            ADX_MIN
        )
        &
        df[
            "ADX_Rising"
        ]
        &
        df[
            "Bull_Regime"
        ]
    )

    return df


# ============================================================
# BACKTEST
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


def run_backtest(
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

            entry_time = (
                df.index[i]
            )

            bars_held = 0

        if not position:
            continue

        bars_held += 1

        if (
            bars_held
            >=
            TIME_EXIT_BARS
        ):
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
                "entry_time": (
                    entry_time
                ),
                "exit_time": (
                    df.index[i]
                ),
                "entry_price": (
                    entry_price
                ),
                "exit_price": (
                    exit_price
                ),
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

def metrics(trades):
    if trades.empty:
        return {
            "trades": 0,
            "wr": 0.0,
            "pnl": 0.0,
            "pf": 0.0,
            "exp": 0.0,
            "median": 0.0,
            "dd": 0.0,
            "equity": 100.0,
        }

    wins = trades[
        trades["pnl_pct"]
        > 0
    ]

    losses = trades[
        trades["pnl_pct"]
        < 0
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
        "wr": (
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
        "exp": float(
            trades[
                "pnl_pct"
            ].mean()
        ),
        "median": float(
            trades[
                "pnl_pct"
            ].median()
        ),
        "dd": float(
            dd.min()
        ),
        "equity": float(
            equity.iloc[-1]
        ),
    }


def line(label, trades):
    m = metrics(
        trades
    )

    pf = (
        "inf"
        if np.isinf(
            m["pf"]
        )
        else
        f"{m['pf']:.3f}"
    )

    print(
        f"{label:<22} | "
        f"Trades {m['trades']:>3} | "
        f"WR {m['wr']:>6.2f}% | "
        f"PnL {m['pnl']:>8.2f}% | "
        f"PF {pf:>6} | "
        f"Exp {m['exp']:>8.4f}% | "
        f"Median {m['median']:>8.4f}% | "
        f"DD {m['dd']:>7.2f}%"
    )

    return m


# ============================================================
# SIGNAL MATCHING
# ============================================================

def signal_times(df):
    return list(
        df.index[
            df[
                "Buy_Signal"
            ].astype(bool)
        ]
    )


def greedy_match(
    times_a,
    times_b,
    tolerance_hours=1,
):
    a = sorted(
        pd.Timestamp(x)
        for x in times_a
    )

    b = sorted(
        pd.Timestamp(x)
        for x in times_b
    )

    unused = set(
        b
    )

    pairs = []

    tolerance = (
        pd.Timedelta(
            hours=tolerance_hours
        )
    )

    for ta in a:
        candidates = [
            tb
            for tb in unused
            if abs(
                tb - ta
            )
            <= tolerance
        ]

        if not candidates:
            continue

        tb = min(
            candidates,
            key=lambda x: abs(
                x - ta
            ),
        )

        unused.remove(
            tb
        )

        pairs.append(
            (
                ta,
                tb,
            )
        )

    return pairs


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 142)
    print(
        "C2G V1.13 - EXTERNAL FEED VALIDATION"
    )
    print("=" * 142)

    raw = {
        "BINANCE": load_ohlcv(
            BINANCE_FILE
        ),
        "BYBIT": load_ohlcv(
            BYBIT_FILE
        ),
        "OKX": load_ohlcv(
            OKX_FILE
        ),
    }

    overlap_start = max(
        df.index.min()
        for df in raw.values()
    )

    overlap_end = min(
        df.index.max()
        for df in raw.values()
    )

    warmup_start = (
        overlap_start
        -
        pd.Timedelta(
            hours=1000
        )
    )

    prepared = {}

    for name, df in raw.items():
        calc = prepare_indicators(
            df.loc[
                warmup_start:
                overlap_end
            ].copy()
        )

        prepared[
            name
        ] = calc.loc[
            overlap_start:
            overlap_end
        ].copy()

    print(
        f"Common 3-market period: "
        f"{overlap_start} -> "
        f"{overlap_end}"
    )

    for name, df in (
        prepared.items()
    ):
        print(
            f"{name:<8} candles: "
            f"{len(df)} | "
            f"BUY signals: "
            f"{int(df['Buy_Signal'].sum())}"
        )

    gross_rows = []
    cost_rows = []
    gross_trades = {}

    print()
    print("=" * 142)
    print("SAME PERIOD - GROSS")
    print("=" * 142)

    for name, df in (
        prepared.items()
    ):
        trades = run_backtest(
            df
        )

        gross_trades[
            name
        ] = trades

        m = line(
            name,
            trades,
        )

        gross_rows.append({
            "market": name,
            **m,
        })

    print()
    print("=" * 142)
    print("SAME PERIOD - COST STRESS")
    print("=" * 142)

    for name, df in (
        prepared.items()
    ):
        trades = run_backtest(
            df,
            fee_pct=(
                STRESS_FEE_PCT_PER_SIDE
            ),
            slippage_pct=(
                STRESS_SLIPPAGE_PCT_PER_SIDE
            ),
        )

        m = line(
            name,
            trades,
        )

        cost_rows.append({
            "market": name,
            **m,
        })

    print()
    print("=" * 142)
    print("RESULT BY YEAR - GROSS")
    print("=" * 142)

    yearly_rows = []

    for name, trades in (
        gross_trades.items()
    ):
        print()
        print(name)
        print(
            "-" * 142
        )

        for year, group in (
            trades.groupby(
                "year"
            )
        ):
            m = line(
                str(
                    int(
                        year
                    )
                ),
                group,
            )

            yearly_rows.append({
                "market": name,
                "year": int(
                    year
                ),
                **m,
            })

    # Pairwise signal matching.
    print()
    print("=" * 142)
    print("BUY SIGNAL AGREEMENT +/- 1H")
    print("=" * 142)

    pairs_to_check = [
        (
            "BINANCE",
            "BYBIT",
        ),
        (
            "BINANCE",
            "OKX",
        ),
        (
            "BYBIT",
            "OKX",
        ),
    ]

    match_rows = []

    for a_name, b_name in (
        pairs_to_check
    ):
        a_times = signal_times(
            prepared[
                a_name
            ]
        )

        b_times = signal_times(
            prepared[
                b_name
            ]
        )

        pairs = greedy_match(
            a_times,
            b_times,
            tolerance_hours=1,
        )

        denominator = max(
            len(
                a_times
            ),
            len(
                b_times
            ),
            1,
        )

        coverage = (
            len(pairs)
            /
            denominator
            *
            100.0
        )

        print(
            f"{a_name:<8} vs "
            f"{b_name:<8} | "
            f"A {len(a_times):>3} | "
            f"B {len(b_times):>3} | "
            f"Matched {len(pairs):>3} | "
            f"Coverage {coverage:>6.2f}%"
        )

        match_rows.append({
            "market_a": a_name,
            "market_b": b_name,
            "signals_a": len(
                a_times
            ),
            "signals_b": len(
                b_times
            ),
            "matched_1h": len(
                pairs
            ),
            "coverage_pct": (
                coverage
            ),
        })

    gross_df = pd.DataFrame(
        gross_rows
    )

    cost_df = pd.DataFrame(
        cost_rows
    )

    print()
    print("=" * 142)
    print("THREE-MARKET SUMMARY")
    print("=" * 142)

    min_pf_gross = float(
        gross_df[
            "pf"
        ].min()
    )

    min_exp_gross = float(
        gross_df[
            "exp"
        ].min()
    )

    min_pf_cost = float(
        cost_df[
            "pf"
        ].min()
    )

    min_exp_cost = float(
        cost_df[
            "exp"
        ].min()
    )

    all_gross_positive = bool(
        (
            gross_df[
                "pf"
            ] > 1.0
        ).all()
        and
        (
            gross_df[
                "exp"
            ] > 0
        ).all()
    )

    all_cost_positive = bool(
        (
            cost_df[
                "pf"
            ] > 1.0
        ).all()
        and
        (
            cost_df[
                "exp"
            ] > 0
        ).all()
    )

    print(
        f"Worst gross PF:       "
        f"{min_pf_gross:.3f}"
    )

    print(
        f"Worst gross Exp:      "
        f"{min_exp_gross:.4f}%"
    )

    print(
        f"All gross positive:   "
        f"{all_gross_positive}"
    )

    print(
        f"Worst cost PF:        "
        f"{min_pf_cost:.3f}"
    )

    print(
        f"Worst cost Exp:       "
        f"{min_exp_cost:.4f}%"
    )

    print(
        f"All cost positive:    "
        f"{all_cost_positive}"
    )

    gross_df.to_csv(
        "c2g_v113_three_market_gross.csv",
        index=False,
    )

    cost_df.to_csv(
        "c2g_v113_three_market_cost.csv",
        index=False,
    )

    pd.DataFrame(
        yearly_rows
    ).to_csv(
        "c2g_v113_three_market_yearly.csv",
        index=False,
    )

    pd.DataFrame(
        match_rows
    ).to_csv(
        "c2g_v113_signal_agreement.csv",
        index=False,
    )

    gross_trades[
        "OKX"
    ].to_csv(
        "c2g_v113_okx_gross_trades.csv",
        index=False,
    )

    print()
    print("=" * 142)
    print("ARQUIVOS GERADOS")
    print("=" * 142)
    print(
        "c2g_v113_three_market_gross.csv"
    )
    print(
        "c2g_v113_three_market_cost.csv"
    )
    print(
        "c2g_v113_three_market_yearly.csv"
    )
    print(
        "c2g_v113_signal_agreement.csv"
    )
    print(
        "c2g_v113_okx_gross_trades.csv"
    )

    print()
    print("=" * 142)
    print("FROZEN RULE")
    print("=" * 142)
    print(
        "BUY only | Supertrend 10/3 | ADX > 25 + 1-bar Rising | "
        "BULL regime | Entry next open | Exit after 24 x 1H candles"
    )
    print(
        "Nenhum parâmetro foi otimizado nesta V1.13."
    )
    print("=" * 142)
