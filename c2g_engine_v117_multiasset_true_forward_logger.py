import pandas as pd
import pandas_ta as ta
import numpy as np
from pathlib import Path


# ============================================================
# C2G SYSTEM PRO - V1.17 MULTI-ASSET TRUE FORWARD LOGGER
#
# OBJETIVO:
# Forward paper test da MESMA regra congelada em:
# BTC, ETH e SOL (Bybit Perpetual 1H)
#
# IMPORTANTE:
# - Isto NÃO substitui a V1.14 do BTC.
# - A V1.17 é um forward multiativo separado.
# - Nenhum parâmetro pode ser alterado durante a coleta.
#
# FREEZE:
# 2026-08-15 20:00:00 UTC
#
# Esse horário é posterior aos candles já usados na pesquisa
# de BTC / ETH / SOL até 2026-08-15 19:00 UTC.
#
# REGRA CONGELADA:
# BUY ONLY
# Supertrend 10 / 3
# ADX(14) > 25
# ADX atual > ADX anterior
# Bull regime:
#   Close > EMA200
#   EMA200 atual > EMA200 de 50 candles atrás
#
# Entrada:
#   OPEN do candle seguinte ao sinal
#
# Saída:
#   CLOSE após 24 candles de 1H
#
# SELL OFF
# SL OFF
# TP OFF
#
# Stress de custos:
# Fee 0.055% por lado
# Slippage 0.020% por lado
#
# Arquivos de mercado:
# btc_data_bybit_perp_1h.csv
# eth_data_bybit_perp_1h.csv
# sol_data_bybit_perp_1h.csv
#
# Saídas:
# c2g_v117_forward_ledger.csv
# c2g_v117_forward_summary.csv
# c2g_v117_forward_manifest.txt
# ============================================================


FREEZE_TIME = pd.Timestamp(
    "2026-08-15 20:00:00"
)

FILES = {
    "BTC": "btc_data_bybit_perp_1h.csv",
    "ETH": "eth_data_bybit_perp_1h.csv",
    "SOL": "sol_data_bybit_perp_1h.csv",
}

SUPERTREND_LENGTH = 10
SUPERTREND_MULTIPLIER = 3.0

ADX_LENGTH = 14
ADX_MIN = 25.0

EMA_LENGTH = 200
EMA_SLOPE_LOOKBACK = 50

TIME_EXIT_BARS = 24

FEE_PCT_PER_SIDE = 0.055
SLIPPAGE_PCT_PER_SIDE = 0.020

LEDGER_FILE = "c2g_v117_forward_ledger.csv"
SUMMARY_FILE = "c2g_v117_forward_summary.csv"
MANIFEST_FILE = "c2g_v117_forward_manifest.txt"


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


def build_forward_events(
    asset,
    df,
):
    rows = []

    signal_positions = np.flatnonzero(
        df["Buy_Signal"]
        .fillna(False)
        .to_numpy()
    )

    for signal_i in signal_positions:
        signal_time = df.index[
            signal_i
        ]

        if signal_time < FREEZE_TIME:
            continue

        signal_adx = float(
            df["ADX"].iloc[
                signal_i
            ]
        )

        entry_i = signal_i + 1

        if entry_i >= len(df):
            rows.append({
                "asset": asset,
                "signal_time": signal_time,
                "status": "WAITING_ENTRY",
                "entry_time": pd.NaT,
                "entry_price": np.nan,
                "planned_exit_time": pd.NaT,
                "exit_time": pd.NaT,
                "exit_price": np.nan,
                "gross_pnl_pct": np.nan,
                "cost_pnl_pct": np.nan,
                "mfe_pct": np.nan,
                "mae_pct": np.nan,
                "signal_adx": signal_adx,
            })

            continue

        entry_time = df.index[
            entry_i
        ]

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

        planned_exit_time = (
            entry_time
            +
            pd.Timedelta(
                hours=24
            )
        )

        # Trade ainda aberto.
        if exit_i >= len(df):
            available = df.iloc[
                entry_i:
            ]

            if available.empty:
                mfe_pct = np.nan
                mae_pct = np.nan

            else:
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

            rows.append({
                "asset": asset,
                "signal_time": signal_time,
                "status": "OPEN",
                "entry_time": entry_time,
                "entry_price": entry_price,
                "planned_exit_time": planned_exit_time,
                "exit_time": pd.NaT,
                "exit_price": np.nan,
                "gross_pnl_pct": np.nan,
                "cost_pnl_pct": np.nan,
                "mfe_pct": mfe_pct,
                "mae_pct": mae_pct,
                "signal_adx": signal_adx,
            })

            continue

        exit_time = df.index[
            exit_i
        ]

        exit_price = float(
            df["close"].iloc[
                exit_i
            ]
        )

        gross = (
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

        costs = (
            FEE_PCT_PER_SIDE * 2.0
            +
            SLIPPAGE_PCT_PER_SIDE * 2.0
        )

        cost_pnl = gross - costs

        window = df.iloc[
            entry_i:
            exit_i + 1
        ]

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
            *
            100.0
        )

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
            *
            100.0
        )

        rows.append({
            "asset": asset,
            "signal_time": signal_time,
            "status": "CLOSED",
            "entry_time": entry_time,
            "entry_price": entry_price,
            "planned_exit_time": planned_exit_time,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "gross_pnl_pct": gross,
            "cost_pnl_pct": cost_pnl,
            "mfe_pct": mfe_pct,
            "mae_pct": mae_pct,
            "signal_adx": signal_adx,
        })

    return pd.DataFrame(
        rows
    )


def calc_pf(values):
    values = (
        pd.Series(
            values,
            dtype=float,
        )
        .dropna()
    )

    if values.empty:
        return np.nan

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


def summary_for_group(
    asset,
    group,
):
    closed = group[
        group["status"]
        ==
        "CLOSED"
    ]

    gross = closed[
        "gross_pnl_pct"
    ].dropna()

    cost = closed[
        "cost_pnl_pct"
    ].dropna()

    return {
        "asset": asset,

        "signals_after_freeze": (
            len(group)
        ),

        "waiting_entry": int(
            (
                group["status"]
                ==
                "WAITING_ENTRY"
            ).sum()
        ),

        "open_trades": int(
            (
                group["status"]
                ==
                "OPEN"
            ).sum()
        ),

        "closed_trades": (
            len(closed)
        ),

        "gross_wr": (
            (
                gross > 0
            ).mean()
            * 100.0
            if len(gross)
            else np.nan
        ),

        "gross_pnl": (
            float(
                gross.sum()
            )
            if len(gross)
            else 0.0
        ),

        "gross_exp": (
            float(
                gross.mean()
            )
            if len(gross)
            else np.nan
        ),

        "gross_pf": (
            calc_pf(
                gross
            )
        ),

        "cost_pnl": (
            float(
                cost.sum()
            )
            if len(cost)
            else 0.0
        ),

        "cost_exp": (
            float(
                cost.mean()
            )
            if len(cost)
            else np.nan
        ),

        "cost_pf": (
            calc_pf(
                cost
            )
        ),

        "median_mfe_pct": (
            float(
                closed[
                    "mfe_pct"
                ].median()
            )
            if len(closed)
            else np.nan
        ),

        "median_mae_pct": (
            float(
                closed[
                    "mae_pct"
                ].median()
            )
            if len(closed)
            else np.nan
        ),
    }


def write_manifest():
    text = f"""C2G SYSTEM PRO - V1.17 MULTI-ASSET FORWARD MANIFEST

FREEZE UTC
{FREEZE_TIME}

ASSETS
BTC/USDT Perpetual - Bybit
ETH/USDT Perpetual - Bybit
SOL/USDT Perpetual - Bybit

FROZEN RULE
BUY ONLY
Supertrend 10 / 3.0
ADX(14) > 25
ADX current > ADX previous candle
Bull regime:
- Close > EMA200
- EMA200 current > EMA200 50 candles ago

ENTRY
Next candle OPEN after confirmed signal.

EXIT
CLOSE after 24 x 1H candles.

DISABLED
SELL
Stop Loss
Take Profit

COST STRESS
Fee per side: {FEE_PCT_PER_SIDE:.3f}%
Slippage per side: {SLIPPAGE_PCT_PER_SIDE:.3f}%

RESEARCH CONSTRAINT
No parameter changes while collecting the forward sample.
Any change requires a new version and a new freeze timestamp.

NOTE
BTC/ETH/SOL were selected after V1.15 showed positive historical
results. Therefore only trades after this freeze timestamp should
be treated as the new forward evidence for this multi-asset set.

PAPER TEST ONLY.
No exchange orders are sent.
Funding is not included.
"""

    Path(
        MANIFEST_FILE
    ).write_text(
        text,
        encoding="utf-8",
    )


if __name__ == "__main__":

    print()
    print("=" * 142)
    print(
        "C2G V1.17 - MULTI-ASSET TRUE FORWARD LOGGER"
    )
    print("=" * 142)

    print(
        f"Freeze UTC: {FREEZE_TIME}"
    )

    event_frames = []

    for asset, path in FILES.items():
        if not Path(path).exists():
            print(
                f"{asset}: arquivo ausente -> {path}"
            )
            continue

        raw = load_ohlcv(
            path
        )

        prepared = prepare_indicators(
            raw
        )

        events = build_forward_events(
            asset,
            prepared,
        )

        event_frames.append(
            events
        )

        print()
        print(asset)
        print(
            f"Último candle: "
            f"{prepared.index.max()}"
        )
        print(
            f"Sinais após freeze: "
            f"{len(events)}"
        )

        if not events.empty:
            print(
                f"Status: "
                f"{events['status'].value_counts().to_dict()}"
            )

    if event_frames:
        ledger = pd.concat(
            event_frames,
            ignore_index=True,
        )
    else:
        ledger = pd.DataFrame()

    if not ledger.empty:
        ledger = ledger.sort_values(
            [
                "signal_time",
                "asset",
            ]
        ).reset_index(
            drop=True
        )

    ledger.to_csv(
        LEDGER_FILE,
        index=False,
    )

    summary_rows = []

    for asset in FILES:
        if ledger.empty:
            group = pd.DataFrame()
        else:
            group = ledger[
                ledger[
                    "asset"
                ]
                ==
                asset
            ]

        if group.empty:
            summary_rows.append({
                "asset": asset,
                "signals_after_freeze": 0,
                "waiting_entry": 0,
                "open_trades": 0,
                "closed_trades": 0,
                "gross_wr": np.nan,
                "gross_pnl": 0.0,
                "gross_exp": np.nan,
                "gross_pf": np.nan,
                "cost_pnl": 0.0,
                "cost_exp": np.nan,
                "cost_pf": np.nan,
                "median_mfe_pct": np.nan,
                "median_mae_pct": np.nan,
            })
        else:
            summary_rows.append(
                summary_for_group(
                    asset,
                    group,
                )
            )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    write_manifest()

    print()
    print("=" * 142)
    print("FORWARD SUMMARY")
    print("=" * 142)

    print(
        summary.to_string(
            index=False,
            formatters={
                "gross_wr": (
                    lambda x:
                    f"{x:.2f}%"
                    if pd.notna(x)
                    else "n/a"
                ),
                "gross_pnl": (
                    "{:.4f}%".format
                ),
                "gross_exp": (
                    lambda x:
                    f"{x:.4f}%"
                    if pd.notna(x)
                    else "n/a"
                ),
                "gross_pf": (
                    lambda x:
                    f"{x:.3f}"
                    if pd.notna(x)
                    else "n/a"
                ),
                "cost_pnl": (
                    "{:.4f}%".format
                ),
                "cost_exp": (
                    lambda x:
                    f"{x:.4f}%"
                    if pd.notna(x)
                    else "n/a"
                ),
                "cost_pf": (
                    lambda x:
                    f"{x:.3f}"
                    if pd.notna(x)
                    else "n/a"
                ),
            },
        )
    )

    print()
    print("=" * 142)
    print("ARQUIVOS")
    print("=" * 142)
    print(LEDGER_FILE)
    print(SUMMARY_FILE)
    print(MANIFEST_FILE)
    print()
    print(
        "PAPER TEST ONLY - nenhuma ordem é enviada."
    )
    print("=" * 142)
