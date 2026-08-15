import ccxt
import pandas as pd
import time


# ============================================================
# C2G SYSTEM PRO - V1.15 MULTI-ASSET BYBIT PERP 1H LOADER
#
# Objetivo:
# Baixar histórico 1H de outros contratos perpétuos USDT
# da Bybit para testar a regra BTC congelada SEM retuning.
#
# Assets:
# ETH, SOL, XRP, BNB
#
# Outputs:
# btc_data_bybit_perp_1h.csv   (mantém se já existir)
# eth_data_bybit_perp_1h.csv
# sol_data_bybit_perp_1h.csv
# xrp_data_bybit_perp_1h.csv
# bnb_data_bybit_perp_1h.csv
#
# O script encontra automaticamente o símbolo swap USDT
# disponível na Bybit via CCXT.
# ============================================================


TIMEFRAME = "1h"
START_DATE = "2018-01-01T00:00:00Z"
LIMIT = 1000

ASSETS = [
    "ETH",
    "SOL",
    "XRP",
    "BNB",
]

MAX_RETRIES = 8
RETRY_SECONDS = 5


def find_usdt_perpetual(exchange, base):
    exchange.load_markets()

    preferred = f"{base}/USDT:USDT"

    if preferred in exchange.markets:
        market = exchange.markets[preferred]

        if (
            market.get("swap")
            and
            market.get("active", True)
        ):
            return preferred

    candidates = []

    for symbol, market in exchange.markets.items():
        if (
            market.get("base") == base
            and
            market.get("quote") == "USDT"
            and
            market.get("swap") is True
            and
            market.get("active", True)
        ):
            candidates.append(symbol)

    if not candidates:
        return None

    return candidates[0]


def download_asset(exchange, base):
    symbol = find_usdt_perpetual(
        exchange,
        base,
    )

    if symbol is None:
        print(
            f"{base}: contrato perpétuo USDT não encontrado."
        )
        return None

    output_file = (
        f"{base.lower()}_data_bybit_perp_1h.csv"
    )

    print()
    print("=" * 92)
    print(
        f"DOWNLOAD {base} | {symbol} | BYBIT PERPETUAL 1H"
    )
    print("=" * 92)

    since = exchange.parse8601(
        START_DATE
    )

    end_ms = exchange.milliseconds()

    timeframe_ms = (
        exchange.parse_timeframe(
            TIMEFRAME
        )
        * 1000
    )

    rows = []
    request_number = 0

    while since < end_ms:
        batch = None

        for attempt in range(
            1,
            MAX_RETRIES + 1
        ):
            try:
                batch = exchange.fetch_ohlcv(
                    symbol,
                    timeframe=TIMEFRAME,
                    since=since,
                    limit=LIMIT,
                    params={
                        "category": "linear",
                    },
                )
                break

            except Exception as exc:
                print(
                    f"{base} req {request_number + 1}, "
                    f"tentativa {attempt}/{MAX_RETRIES}: "
                    f"{exc}"
                )

                if attempt == MAX_RETRIES:
                    raise

                time.sleep(
                    RETRY_SECONDS
                )

        if not batch:
            break

        batch = [
            row
            for row in batch
            if row[0] <= end_ms
        ]

        if not batch:
            break

        batch.sort(
            key=lambda row: row[0]
        )

        first_ts = batch[0][0]
        last_ts = batch[-1][0]

        rows.extend(
            batch
        )

        request_number += 1

        print(
            f"{base} | Req {request_number:03d} | "
            f"+{len(batch):4d} | "
            f"{pd.to_datetime(first_ts, unit='ms', utc=True).strftime('%Y-%m-%d')} "
            f"-> "
            f"{pd.to_datetime(last_ts, unit='ms', utc=True).strftime('%Y-%m-%d')} | "
            f"bruto {len(rows)}"
        )

        next_since = (
            last_ts
            +
            timeframe_ms
        )

        if next_since <= since:
            break

        since = next_since

        if last_ts >= (
            end_ms
            -
            timeframe_ms
        ):
            break

    if not rows:
        print(
            f"{base}: nenhum candle baixado."
        )
        return None

    df = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ],
    )

    raw_count = len(df)

    df.drop_duplicates(
        subset=["timestamp"],
        keep="last",
        inplace=True,
    )

    df.sort_values(
        "timestamp",
        inplace=True,
    )

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
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

    df["timestamp"] = (
        pd.to_datetime(
            df["timestamp"],
            unit="ms",
            utc=True,
        )
        .dt.tz_localize(None)
    )

    df.set_index(
        "timestamp",
        inplace=True,
    )

    df.to_csv(
        output_file
    )

    first = df.index.min()
    last = df.index.max()

    expected = (
        (
            last - first
        ).total_seconds()
        / 3600.0
        + 1.0
    )

    coverage = (
        len(df)
        /
        expected
        *
        100.0
        if expected > 0
        else 0.0
    )

    print()
    print(
        f"{base} FINALIZADO | "
        f"Candles {len(df)} | "
        f"{first} -> {last} | "
        f"Cobertura {coverage:.2f}% | "
        f"{output_file}"
    )

    return {
        "asset": base,
        "symbol": symbol,
        "file": output_file,
        "raw": raw_count,
        "candles": len(df),
        "first": first,
        "last": last,
        "coverage": coverage,
    }


def main():
    exchange = ccxt.bybit({
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",
        },
    })

    print()
    print("=" * 92)
    print(
        "C2G V1.15 - MULTI-ASSET BYBIT PERPETUAL HISTORY"
    )
    print("=" * 92)

    results = []

    for base in ASSETS:
        result = download_asset(
            exchange,
            base,
        )

        if result:
            results.append(
                result
            )

    print()
    print("=" * 120)
    print("RESUMO FINAL")
    print("=" * 120)

    if results:
        summary = pd.DataFrame(
            results
        )

        print(
            summary.to_string(
                index=False
            )
        )

        summary.to_csv(
            "c2g_v115_multiasset_download_summary.csv",
            index=False,
        )

        print()
        print(
            "Resumo salvo em: "
            "c2g_v115_multiasset_download_summary.csv"
        )

    else:
        print(
            "Nenhum ativo foi baixado."
        )

    print("=" * 120)


if __name__ == "__main__":
    main()
