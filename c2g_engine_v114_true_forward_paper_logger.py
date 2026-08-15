import pandas as pd
import pandas_ta as ta
import numpy as np
from pathlib import Path


# ============================================================
# C2G SYSTEM PRO - V1.14 TRUE FORWARD PAPER LOGGER
#
# REGRA CONGELADA EM:
# 2026-08-15 19:00:00 UTC
#
# A partir desse timestamp, este script apenas REGISTRA o que
# acontecer. Ele NÃO otimiza, NÃO altera parâmetros e NÃO envia
# ordens para corretora.
#
# REGRA:
# BUY ONLY
# Supertrend 10/3
# ADX > 25
# ADX atual > ADX anterior
# BULL:
#   Close > EMA200
#   EMA200 atual > EMA200 de 50 candles atrás
# Entrada:
#   OPEN do candle seguinte ao sinal
# Saída:
#   CLOSE após 24 candles de 1H
# SELL OFF
# SL OFF
# TP OFF
#
# Para atualizar o forward test:
# 1) atualize os CSVs de mercado;
# 2) rode este script novamente;
# 3) ele reconstrói apenas os sinais APÓS o freeze timestamp.
#
# Arquivos:
# btc_data_binance_full_1h.csv
# btc_data_bybit_perp_1h.csv
# btc_data_okx_perp_1h.csv
#
# Saídas:
# c2g_v114_forward_ledger.csv
# c2g_v114_forward_summary.csv
# c2g_v114_forward_manifest.txt
# ============================================================


FREEZE_TIME = pd.Timestamp(
    "2026-08-15 19:00:00"
)

MARKETS = {
    "BINANCE_SPOT": (
        "btc_data_binance_full_1h.csv"
    ),
    "BYBIT_PERPETUAL": (
        "btc_data_bybit_perp_1h.csv"
    ),
    "OKX_PERPETUAL": (
        "btc_data_okx_perp_1h.csv"
    ),
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

LEDGER_FILE = (
    "c2g_v114_forward_ledger.csv"
)

SUMMARY_FILE = (
    "c2g_v114_forward_summary.csv"
)

MANIFEST_FILE = (
    "c2g_v114_forward_manifest.txt"
)


# ============================================================
# DATA / INDICATORS
# ============================================================

def find_column(
    columns,
    prefix,
):
    matches = [
        col
        for col in columns
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


def load_ohlcv(
    path,
):
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


def prepare_indicators(
    df,
):
    df = df.copy()

    st = ta.supertrend(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        length=(
            SUPERTREND_LENGTH
        ),
        multiplier=(
            SUPERTREND_MULTIPLIER
        ),
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
        df["ADX"].shift(
            1
        )
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
            df[
                "EMA200_Lag"
            ]
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
# FORWARD EVENT RECONSTRUCTION
# ============================================================

def build_forward_events(
    market_name,
    df,
):
    rows = []

    signal_positions = (
        np.flatnonzero(
            df[
                "Buy_Signal"
            ]
            .fillna(False)
            .to_numpy()
        )
    )

    for signal_i in (
        signal_positions
    ):
        signal_time = (
            df.index[
                signal_i
            ]
        )

        if (
            signal_time
            <
            FREEZE_TIME
        ):
            continue

        entry_i = (
            signal_i + 1
        )

        if entry_i >= len(
            df
        ):
            rows.append({
                "market": (
                    market_name
                ),
                "signal_time": (
                    signal_time
                ),
                "status": (
                    "WAITING_ENTRY"
                ),
                "entry_time": (
                    pd.NaT
                ),
                "entry_price": (
                    np.nan
                ),
                "planned_exit_time": (
                    pd.NaT
                ),
                "exit_time": (
                    pd.NaT
                ),
                "exit_price": (
                    np.nan
                ),
                "gross_pnl_pct": (
                    np.nan
                ),
                "cost_stress_pnl_pct": (
                    np.nan
                ),
                "mfe_pct": np.nan,
                "mae_pct": np.nan,
                "signal_adx": (
                    float(
                        df[
                            "ADX"
                        ].iloc[
                            signal_i
                        ]
                    )
                ),
            })

            continue

        entry_time = (
            df.index[
                entry_i
            ]
        )

        entry_price = float(
            df["open"].iloc[
                entry_i
            ]
        )

        exit_i = (
            entry_i
            +
            TIME_EXIT_BARS
            -
            1
        )

        # Ainda não completou 24 candles.
        if exit_i >= len(
            df
        ):
            available = df.iloc[
                entry_i:
            ]

            if not available.empty:
                mfe_pct = (
                    (
                        float(
                            available[
                                "high"
                            ].max()
                        )
                        -
                        entry_price
                    )
                    /
                    entry_price
                    *
                    100.0
                )

                mae_pct = (
                    (
                        entry_price
                        -
                        float(
                            available[
                                "low"
                            ].min()
                        )
                    )
                    /
                    entry_price
                    *
                    100.0
                )
            else:
                mfe_pct = np.nan
                mae_pct = np.nan

            planned_exit_time = (
                entry_time
                +
                pd.Timedelta(
                    hours=24
                )
            )

            rows.append({
                "market": (
                    market_name
                ),
                "signal_time": (
                    signal_time
                ),
                "status": "OPEN",
                "entry_time": (
                    entry_time
                ),
                "entry_price": (
                    entry_price
                ),
                "planned_exit_time": (
                    planned_exit_time
                ),
                "exit_time": (
                    pd.NaT
                ),
                "exit_price": (
                    np.nan
                ),
                "gross_pnl_pct": (
                    np.nan
                ),
                "cost_stress_pnl_pct": (
                    np.nan
                ),
                "mfe_pct": (
                    mfe_pct
                ),
                "mae_pct": (
                    mae_pct
                ),
                "signal_adx": (
                    float(
                        df[
                            "ADX"
                        ].iloc[
                            signal_i
                        ]
                    )
                ),
            })

            continue

        exit_time = (
            df.index[
                exit_i
            ]
        )

        exit_price = float(
            df["close"].iloc[
                exit_i
            ]
        )

        gross_pnl = (
            (
                exit_price
                -
                entry_price
            )
            /
            entry_price
            *
            100.0
        )

        round_trip_cost = (
            STRESS_FEE_PCT_PER_SIDE
            * 2.0
            +
            STRESS_SLIPPAGE_PCT_PER_SIDE
            * 2.0
        )

        cost_pnl = (
            gross_pnl
            -
            round_trip_cost
        )

        trade_window = (
            df.iloc[
                entry_i:
                exit_i + 1
            ]
        )

        mfe_pct = (
            (
                float(
                    trade_window[
                        "high"
                    ].max()
                )
                -
                entry_price
            )
            /
            entry_price
            *
            100.0
        )

        mae_pct = (
            (
                entry_price
                -
                float(
                    trade_window[
                        "low"
                    ].min()
                )
            )
            /
            entry_price
            *
            100.0
        )

        rows.append({
            "market": (
                market_name
            ),
            "signal_time": (
                signal_time
            ),
            "status": (
                "CLOSED"
            ),
            "entry_time": (
                entry_time
            ),
            "entry_price": (
                entry_price
            ),
            "planned_exit_time": (
                entry_time
                +
                pd.Timedelta(
                    hours=24
                )
            ),
            "exit_time": (
                exit_time
            ),
            "exit_price": (
                exit_price
            ),
            "gross_pnl_pct": (
                gross_pnl
            ),
            "cost_stress_pnl_pct": (
                cost_pnl
            ),
            "mfe_pct": (
                mfe_pct
            ),
            "mae_pct": (
                mae_pct
            ),
            "signal_adx": (
                float(
                    df[
                        "ADX"
                    ].iloc[
                        signal_i
                    ]
                )
            ),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# SUMMARY
# ============================================================

def calc_pf(
    values,
):
    values = (
        pd.Series(
            values,
            dtype=float,
        )
        .dropna()
    )

    gp = values[
        values > 0
    ].sum()

    gl = abs(
        values[
            values < 0
        ].sum()
    )

    if gl <= 0:
        return (
            np.inf
            if gp > 0
            else 0.0
        )

    return float(
        gp / gl
    )


def build_summary(
    ledger,
):
    rows = []

    for market in MARKETS:
        subset = ledger[
            ledger[
                "market"
            ]
            ==
            market
        ]

        closed = subset[
            subset[
                "status"
            ]
            ==
            "CLOSED"
        ]

        gross = (
            closed[
                "gross_pnl_pct"
            ]
            .dropna()
        )

        cost = (
            closed[
                "cost_stress_pnl_pct"
            ]
            .dropna()
        )

        rows.append({
            "market": market,

            "signals_after_freeze": (
                len(
                    subset
                )
            ),

            "open_trades": int(
                (
                    subset[
                        "status"
                    ]
                    ==
                    "OPEN"
                ).sum()
            ),

            "waiting_entry": int(
                (
                    subset[
                        "status"
                    ]
                    ==
                    "WAITING_ENTRY"
                ).sum()
            ),

            "closed_trades": (
                len(
                    closed
                )
            ),

            "gross_win_rate": (
                (
                    gross > 0
                ).mean()
                * 100.0
                if len(
                    gross
                )
                else np.nan
            ),

            "gross_pnl": (
                float(
                    gross.sum()
                )
                if len(
                    gross
                )
                else 0.0
            ),

            "gross_expectancy": (
                float(
                    gross.mean()
                )
                if len(
                    gross
                )
                else np.nan
            ),

            "gross_pf": (
                calc_pf(
                    gross
                )
                if len(
                    gross
                )
                else np.nan
            ),

            "cost_pnl": (
                float(
                    cost.sum()
                )
                if len(
                    cost
                )
                else 0.0
            ),

            "cost_expectancy": (
                float(
                    cost.mean()
                )
                if len(
                    cost
                )
                else np.nan
            ),

            "cost_pf": (
                calc_pf(
                    cost
                )
                if len(
                    cost
                )
                else np.nan
            ),

            "median_mfe_pct": (
                float(
                    closed[
                        "mfe_pct"
                    ].median()
                )
                if len(
                    closed
                )
                else np.nan
            ),

            "median_mae_pct": (
                float(
                    closed[
                        "mae_pct"
                    ].median()
                )
                if len(
                    closed
                )
                else np.nan
            ),
        })

    return pd.DataFrame(
        rows
    )


def write_manifest():
    text = f"""C2G SYSTEM PRO - V1.14 TRUE FORWARD PAPER TEST

FREEZE TIMESTAMP (UTC)
{FREEZE_TIME}

RULE
BUY ONLY
Supertrend: 10 / 3.0
ADX length: 14
ADX minimum: > 25
ADX Rising: current ADX > previous candle ADX
Bull regime:
- Close > EMA200
- EMA200 current > EMA200 50 candles ago

ENTRY
Next candle OPEN after confirmed signal.

EXIT
Close after 24 x 1H candles.

DISABLED
SELL
Stop Loss
Take Profit

COST STRESS
Fee assumption per side: {STRESS_FEE_PCT_PER_SIDE:.3f}%
Slippage assumption per side: {STRESS_SLIPPAGE_PCT_PER_SIDE:.3f}%

RESEARCH RULE
No parameter changes are allowed while collecting this forward sample.
Any change requires a new strategy version and a new freeze timestamp.

IMPORTANT
This script is paper/research only. It does not send exchange orders.
Funding is not included.
"""

    Path(
        MANIFEST_FILE
    ).write_text(
        text,
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 140)
    print(
        "C2G V1.14 - TRUE FORWARD PAPER LOGGER"
    )
    print("=" * 140)

    print(
        f"Freeze UTC: "
        f"{FREEZE_TIME}"
    )

    all_events = []

    for market_name, path in (
        MARKETS.items()
    ):
        if not Path(
            path
        ).exists():
            print(
                f"{market_name}: "
                f"arquivo ausente -> {path}"
            )
            continue

        raw = load_ohlcv(
            path
        )

        prepared = (
            prepare_indicators(
                raw
            )
        )

        events = (
            build_forward_events(
                market_name,
                prepared,
            )
        )

        all_events.append(
            events
        )

        last_candle = (
            prepared.index.max()
        )

        print()
        print(
            f"{market_name}"
        )

        print(
            f"Último candle: "
            f"{last_candle}"
        )

        print(
            f"Sinais após freeze: "
            f"{len(events)}"
        )

        if not events.empty:
            counts = (
                events[
                    "status"
                ]
                .value_counts()
                .to_dict()
            )

            print(
                f"Status: {counts}"
            )

    if all_events:
        ledger = pd.concat(
            all_events,
            ignore_index=True,
        )
    else:
        ledger = pd.DataFrame()

    if not ledger.empty:
        ledger = ledger.sort_values(
            [
                "signal_time",
                "market",
            ]
        ).reset_index(
            drop=True
        )

    ledger.to_csv(
        LEDGER_FILE,
        index=False,
    )

    summary = (
        build_summary(
            ledger
        )
        if not ledger.empty
        else pd.DataFrame()
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    write_manifest()

    print()
    print("=" * 140)
    print(
        "FORWARD SUMMARY"
    )
    print("=" * 140)

    if summary.empty:
        print(
            "Ainda não existem sinais "
            "após o freeze timestamp."
        )
    else:
        print(
            summary.to_string(
                index=False,
                formatters={
                    "gross_win_rate": (
                        lambda x:
                        (
                            f"{x:.2f}%"
                            if pd.notna(x)
                            else "n/a"
                        )
                    ),
                    "gross_pnl": (
                        "{:.4f}%".format
                    ),
                    "gross_expectancy": (
                        lambda x:
                        (
                            f"{x:.4f}%"
                            if pd.notna(x)
                            else "n/a"
                        )
                    ),
                    "gross_pf": (
                        lambda x:
                        (
                            f"{x:.3f}"
                            if pd.notna(x)
                            else "n/a"
                        )
                    ),
                    "cost_pnl": (
                        "{:.4f}%".format
                    ),
                    "cost_expectancy": (
                        lambda x:
                        (
                            f"{x:.4f}%"
                            if pd.notna(x)
                            else "n/a"
                        )
                    ),
                    "cost_pf": (
                        lambda x:
                        (
                            f"{x:.3f}"
                            if pd.notna(x)
                            else "n/a"
                        )
                    ),
                    "median_mfe_pct": (
                        lambda x:
                        (
                            f"{x:.4f}%"
                            if pd.notna(x)
                            else "n/a"
                        )
                    ),
                    "median_mae_pct": (
                        lambda x:
                        (
                            f"{x:.4f}%"
                            if pd.notna(x)
                            else "n/a"
                        )
                    ),
                },
            )
        )

    print()
    print("=" * 140)
    print(
        "ARQUIVOS"
    )
    print("=" * 140)
    print(
        LEDGER_FILE
    )
    print(
        SUMMARY_FILE
    )
    print(
        MANIFEST_FILE
    )
    print()
    print(
        "PAPER TEST ONLY - "
        "nenhuma ordem é enviada."
    )
    print("=" * 140)
