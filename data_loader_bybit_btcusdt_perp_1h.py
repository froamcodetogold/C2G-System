import ccxt
import pandas as pd
import time


# ============================================================
# C2G SYSTEM - BYBIT BTCUSDT USDT PERPETUAL 1H HISTORY
#
# Objetivo:
# Baixar o histórico disponível do BTCUSDT Perpetual da Bybit
# para validação independente da estratégia criada com dados
# spot da Binance.
#
# Output:
# btc_data_bybit_perp_1h.csv
# ============================================================

TIMEFRAME = "1h"
START_DATE = "2018-01-01T00:00:00Z"
LIMIT = 1000
OUTPUT_FILE = "btc_data_bybit_perp_1h.csv"

MAX_RETRIES = 8
RETRY_SECONDS = 5


def find_btc_usdt_perpetual(exchange):
    exchange.load_markets()

    preferred = "BTC/USDT:USDT"

    if preferred in exchange.markets:
        market = exchange.markets[preferred]

        if market.get("swap"):
            return preferred

    candidates = []

    for symbol, market in exchange.markets.items():
        if (
            market.get("base") == "BTC"
            and market.get("quote") == "USDT"
            and market.get("swap") is True
        ):
            candidates.append(symbol)

    if not candidates:
        raise RuntimeError(
            "Não encontrei o contrato perpétuo BTC/USDT na Bybit."
        )

    print(
        "Símbolo perpétuo encontrado automaticamente:",
        candidates[0]
    )

    return candidates[0]


def fetch_history():
    exchange = ccxt.bybit({
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",
        },
    })

    print("=" * 82)
    print("C2G - DOWNLOAD BYBIT BTCUSDT PERPETUAL 1H")
    print("=" * 82)

    symbol = find_btc_usdt_perpetual(
        exchange
    )

    print(f"Símbolo CCXT: {symbol}")
    print(f"Timeframe:    {TIMEFRAME}")
    print(f"Início pedido:{START_DATE}")
    print(f"Saída:        {OUTPUT_FILE}")
    print()

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

    all_rows = []
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
                    f"Erro req {request_number + 1}, "
                    f"tentativa {attempt}/{MAX_RETRIES}: "
                    f"{exc}"
                )

                if attempt == MAX_RETRIES:
                    raise

                time.sleep(
                    RETRY_SECONDS
                )

        if not batch:
            print(
                "A Bybit não retornou mais candles."
            )
            break

        batch = [
            row
            for row in batch
            if row[0] <= end_ms
        ]

        if not batch:
            break

        batch = sorted(
            batch,
            key=lambda row: row[0]
        )

        first_ts = batch[0][0]
        last_ts = batch[-1][0]

        if last_ts < since:
            print(
                "Sem avanço no timestamp. "
                "Encerrando por segurança."
            )
            break

        all_rows.extend(
            batch
        )

        request_number += 1

        first_dt = pd.to_datetime(
            first_ts,
            unit="ms",
            utc=True,
        )

        last_dt = pd.to_datetime(
            last_ts,
            unit="ms",
            utc=True,
        )

        print(
            f"Req {request_number:03d} | "
            f"+{len(batch):4d} | "
            f"{first_dt.strftime('%Y-%m-%d')} -> "
            f"{last_dt.strftime('%Y-%m-%d')} | "
            f"bruto {len(all_rows)}"
        )

        next_since = (
            last_ts
            +
            timeframe_ms
        )

        if next_since <= since:
            print(
                "Timestamp não avançou."
            )
            break

        since = next_since

        if last_ts >= (
            end_ms - timeframe_ms
        ):
            break

    if not all_rows:
        raise RuntimeError(
            "Nenhum candle foi baixado da Bybit."
        )

    print()
    print("Limpando dados...")

    df = pd.DataFrame(
        all_rows,
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
        OUTPUT_FILE
    )

    first = df.index.min()
    last = df.index.max()

    expected_hours = (
        (
            last - first
        ).total_seconds()
        / 3600
    ) + 1

    coverage = (
        len(df)
        / expected_hours
        * 100
        if expected_hours > 0
        else 0
    )

    print()
    print("=" * 82)
    print("DOWNLOAD BYBIT FINALIZADO")
    print("=" * 82)
    print(f"Candles brutos:       {raw_count}")
    print(f"Candles finais:       {len(df)}")
    print(f"Primeiro candle:      {first}")
    print(f"Último candle:        {last}")
    print(f"Cobertura aproximada: {coverage:.2f}%")
    print(f"Arquivo salvo:        {OUTPUT_FILE}")
    print("=" * 82)


if __name__ == "__main__":
    fetch_history()
