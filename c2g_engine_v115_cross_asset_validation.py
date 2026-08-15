import pandas as pd
import pandas_ta as ta
import numpy as np
from pathlib import Path


# ============================================================
# C2G SYSTEM PRO - V1.15 CROSS-ASSET GENERALIZATION TEST
#
# PERGUNTA:
# A regra congelada do BTC também mostra edge em outros
# criptoativos líquidos, SEM mudar nenhum parâmetro?
#
# REGRA CONGELADA:
# BUY ONLY
# Supertrend 10 / 3
# ADX(14) > 25
# ADX atual > ADX anterior
# Regime BULL:
#   Close > EMA200
#   EMA200 atual > EMA200 de 50 candles atrás
# Entrada:
#   OPEN do candle seguinte
# Saída:
#   CLOSE após 24 candles 1H
#
# SEM:
# - retuning por ativo
# - SELL
# - SL
# - TP
#
# Assets:
# BTC, ETH, SOL, XRP, BNB
# Bybit USDT Perpetual
#
# Dois relatórios:
# 1) FULL AVAILABLE PER ASSET
# 2) COMMON PERIOD entre todos os ativos disponíveis
#
# Stress de custos:
# 0.055% fee por lado
# 0.020% slippage por lado
#
# IMPORTANTE:
# A estratégia NÃO precisa funcionar em todo ativo para ser
# válida no BTC. Este teste mede GENERALIZAÇÃO do conceito.
# ============================================================


FILES = {
    "BTC": "btc_data_bybit_perp_1h.csv",
    "ETH": "eth_data_bybit_perp_1h.csv",
    "SOL": "sol_data_bybit_perp_1h.csv",
    "XRP": "xrp_data_bybit_perp_1h.csv",
    "BNB": "bnb_data_bybit_perp_1h.csv",
}

SUPERTREND_LENGTH = 10
SUPERTREND_MULTIPLIER = 3.0

ADX_LENGTH = 14
ADX_MIN = 25.0

EMA_LENGTH = 200
EMA_SLOPE_LOOKBACK = 50

TIME_EXIT_BARS = 24

STRESS_FEE_PCT_PER_SIDE = 0.055
STRESS_SLIPPAGE_PCT_PER_SIDE = 0.020


def find_column(columns, prefix):
    matches = [
        col
        for col in columns
        if str(col).startswith(
            prefix
        )
    ]

    if not matches:
        raise KeyError(
            f"Nenhuma coluna começando com '{prefix}'."
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

    df["Bull_Regime"] = (
        (df["close"] > df["EMA200"])
        &
        (
            df["EMA200"]
            >
            df["EMA200_Lag"]
        )
    )

    df["Buy_Signal"] = (
        df["ST_Buy_Flip"]
        &
        (df["ADX"] > ADX_MIN)
        &
        df["ADX_Rising"]
        &
        df["Bull_Regime"]
    )

    return df


def calculate_return(
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

            entry_time = df.index[
                i
            ]

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

            pnl = calculate_return(
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

        trades["year"] = (
            trades[
                "entry_time"
            ].dt.year
        )

    return trades


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
        }

    wins = trades[
        trades[
            "pnl_pct"
        ] > 0
    ]

    losses = trades[
        trades[
            "pnl_pct"
        ] < 0
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
    }


def print_line(
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
        f"{label:<8} | "
        f"Trades {m['trades']:>3} | "
        f"WR {m['wr']:>6.2f}% | "
        f"PnL {m['pnl']:>8.2f}% | "
        f"PF {pf_text:>6} | "
        f"Exp {m['exp']:>8.4f}% | "
        f"Median {m['median']:>8.4f}% | "
        f"DD {m['dd']:>7.2f}%"
    )

    return m


if __name__ == "__main__":

    print()
    print("=" * 146)
    print(
        "C2G V1.15 - CROSS-ASSET GENERALIZATION TEST"
    )
    print("=" * 146)

    raw = {}

    for asset, path in FILES.items():
        if not Path(
            path
        ).exists():
            print(
                f"{asset}: arquivo não encontrado -> {path}"
            )
            continue

        raw[
            asset
        ] = load_ohlcv(
            path
        )

    if len(raw) < 2:
        raise RuntimeError(
            "Poucos arquivos disponíveis para comparação."
        )

    prepared_full = {
        asset: prepare_indicators(
            df.copy()
        )
        for asset, df in raw.items()
    }

    print()
    print("ARQUIVOS DISPONÍVEIS")
    print("-" * 146)

    for asset, df in raw.items():
        print(
            f"{asset:<5} | "
            f"{len(df):>6} candles | "
            f"{df.index.min()} -> {df.index.max()}"
        )

    # ========================================================
    # FULL AVAILABLE PER ASSET
    # ========================================================

    full_rows = []
    full_trades = {}

    print()
    print("=" * 146)
    print("FULL AVAILABLE PER ASSET - GROSS")
    print("=" * 146)

    for asset, df in prepared_full.items():
        trades = run_backtest(
            df
        )

        full_trades[
            asset
        ] = trades

        m = print_line(
            asset,
            trades,
        )

        full_rows.append({
            "asset": asset,
            "scenario": "FULL_GROSS",
            **m,
        })

    print()
    print("=" * 146)
    print("FULL AVAILABLE PER ASSET - COST STRESS")
    print("=" * 146)

    for asset, df in prepared_full.items():
        trades = run_backtest(
            df,
            fee_pct=(
                STRESS_FEE_PCT_PER_SIDE
            ),
            slippage_pct=(
                STRESS_SLIPPAGE_PCT_PER_SIDE
            ),
        )

        m = print_line(
            asset,
            trades,
        )

        full_rows.append({
            "asset": asset,
            "scenario": "FULL_COST",
            **m,
        })

    # ========================================================
    # COMMON PERIOD
    # ========================================================

    common_start = max(
        df.index.min()
        for df in raw.values()
    )

    common_end = min(
        df.index.max()
        for df in raw.values()
    )

    warmup_start = (
        common_start
        -
        pd.Timedelta(
            hours=1000
        )
    )

    common_prepared = {}

    for asset, df in raw.items():
        calc = prepare_indicators(
            df.loc[
                warmup_start:
                common_end
            ].copy()
        )

        common_prepared[
            asset
        ] = calc.loc[
            common_start:
            common_end
        ].copy()

    print()
    print("=" * 146)
    print(
        "COMMON PERIOD"
    )
    print("=" * 146)

    print(
        f"{common_start} -> {common_end}"
    )

    common_rows = []

    print()
    print(
        "COMMON PERIOD - GROSS"
    )
    print(
        "-" * 146
    )

    for asset, df in common_prepared.items():
        trades = run_backtest(
            df
        )

        m = print_line(
            asset,
            trades,
        )

        common_rows.append({
            "asset": asset,
            "scenario": (
                "COMMON_GROSS"
            ),
            **m,
        })

    print()
    print(
        "COMMON PERIOD - COST STRESS"
    )
    print(
        "-" * 146
    )

    for asset, df in common_prepared.items():
        trades = run_backtest(
            df,
            fee_pct=(
                STRESS_FEE_PCT_PER_SIDE
            ),
            slippage_pct=(
                STRESS_SLIPPAGE_PCT_PER_SIDE
            ),
        )

        m = print_line(
            asset,
            trades,
        )

        common_rows.append({
            "asset": asset,
            "scenario": (
                "COMMON_COST"
            ),
            **m,
        })

    # ========================================================
    # GENERALIZATION SCORE
    # ========================================================

    all_rows = pd.DataFrame(
        full_rows
        +
        common_rows
    )

    common_cost = all_rows[
        all_rows[
            "scenario"
        ]
        ==
        "COMMON_COST"
    ].copy()

    n_assets = len(
        common_cost
    )

    positive_assets = int(
        (
            (
                common_cost[
                    "pf"
                ] > 1.0
            )
            &
            (
                common_cost[
                    "exp"
                ] > 0
            )
        ).sum()
    )

    robust_assets = int(
        (
            (
                common_cost[
                    "pf"
                ] >= 1.20
            )
            &
            (
                common_cost[
                    "exp"
                ] > 0
            )
        ).sum()
    )

    median_pf = float(
        common_cost[
            "pf"
        ].median()
    )

    worst_pf = float(
        common_cost[
            "pf"
        ].min()
    )

    median_exp = float(
        common_cost[
            "exp"
        ].median()
    )

    print()
    print("=" * 146)
    print(
        "C2G V1.15 - CROSS-ASSET GENERALIZATION SUMMARY"
    )
    print("=" * 146)

    print(
        f"Assets testados no período comum: "
        f"{n_assets}"
    )

    print(
        f"PF > 1 e Exp > 0 após custos: "
        f"{positive_assets}/{n_assets}"
    )

    print(
        f"PF >= 1.20 e Exp > 0 após custos: "
        f"{robust_assets}/{n_assets}"
    )

    print(
        f"Median PF após custos: "
        f"{median_pf:.3f}"
    )

    print(
        f"Worst PF após custos:  "
        f"{worst_pf:.3f}"
    )

    print(
        f"Median Exp após custos:"
        f" {median_exp:.4f}%"
    )

    print()
    print(
        "INTERPRETAÇÃO:"
    )

    print(
        "- A estratégia NÃO precisa ganhar em todos os ativos "
        "para continuar válida no BTC."
    )

    print(
        "- Se vários ativos diferentes também tiverem edge sem "
        "retuning, aumenta a evidência de que a lógica captura "
        "um comportamento de tendência mais geral."
    )

    print(
        "- Se somente BTC funcionar, trate o C2G como uma "
        "estratégia específica de BTC e não force parâmetros "
        "nos outros ativos."
    )

    all_rows.to_csv(
        "c2g_v115_cross_asset_results.csv",
        index=False,
    )

    print()
    print(
        "Resultado salvo em: "
        "c2g_v115_cross_asset_results.csv"
    )

    print("=" * 146)
