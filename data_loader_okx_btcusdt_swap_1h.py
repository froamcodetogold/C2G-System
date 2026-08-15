import requests
import pandas as pd
import time
from datetime import datetime, timezone


# ============================================================
# C2G SYSTEM PRO - OKX BTC-USDT-SWAP 1H HISTORY LOADER
#
# Public endpoint:
# GET /api/v5/market/history-candles
#
# Instrument:
# BTC-USDT-SWAP
#
# Timeframe:
# 1H
#
# Output:
# btc_data_okx_perp_1h.csv
#
# Não usa API key.
# Não sobrescreve os arquivos Binance/Bybit.
# ============================================================


BASE_URL = "https://www.okx.com"
ENDPOINT = "/api/v5/market/history-candles"

INST_ID = "BTC-USDT-SWAP"
BAR = "1H"
LIMIT = 100

# Pede histórico até 2020. A API pode simplesmente parar antes
# caso o instrumento / endpoint não tenha candles tão antigos.
START_DATE = "2020-01-01T00:00:00Z"

OUTPUT_FILE = "btc_data_okx_perp_1h.csv"

MAX_RETRIES = 8
REQUEST_SLEEP = 0.15
RETRY_SLEEP = 3.0


def iso_to_ms(value):
    dt = datetime.fromisoformat(
        value.replace(
            "Z",
            "+00:00",
        )
    )

    return int(
        dt.timestamp()
        * 1000
    )


def request_page(after=None):
    params = {
        "instId": INST_ID,
        "bar": BAR,
        "limit": str(LIMIT),
    }

    if after is not None:
        params["after"] = str(
            after
        )

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):
        try:
            response = requests.get(
                BASE_URL + ENDPOINT,
                params=params,
                timeout=20,
            )

            response.raise_for_status()

            payload = response.json()

            if payload.get(
                "code"
            ) != "0":
                raise RuntimeError(
                    f"OKX code={payload.get('code')} "
                    f"msg={payload.get('msg')}"
                )

            return payload.get(
                "data",
                []
            )

        except Exception as exc:
            last_error = exc

            print(
                f"Erro tentativa "
                f"{attempt}/{MAX_RETRIES}: "
                f"{exc}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(
                    RETRY_SLEEP
                )

    raise RuntimeError(
        f"Falha ao consultar OKX: "
        f"{last_error}"
    )


def main():
    print()
    print("=" * 88)
    print(
        "C2G - DOWNLOAD OKX BTC-USDT-SWAP 1H"
    )
    print("=" * 88)

    start_ms = iso_to_ms(
        START_DATE
    )

    rows = []
    after = None
    page_number = 0

    previous_oldest = None

    while True:
        page = request_page(
            after=after
        )

        if not page:
            print(
                "API não retornou mais candles."
            )
            break

        parsed = []

        for item in page:
            # OKX history candle:
            # [ts,o,h,l,c,vol,volCcy,volCcyQuote,confirm]
            if len(item) < 6:
                continue

            ts = int(
                item[0]
            )

            confirm = (
                str(
                    item[8]
                )
                if len(item) > 8
                else "1"
            )

            # Ignora candle ainda não confirmado.
            if confirm != "1":
                continue

            parsed.append([
                ts,
                float(
                    item[1]
                ),
                float(
                    item[2]
                ),
                float(
                    item[3]
                ),
                float(
                    item[4]
                ),
                float(
                    item[5]
                ),
            ])

        if not parsed:
            print(
                "Página sem candles confirmados."
            )
            break

        parsed.sort(
            key=lambda x: x[0]
        )

        oldest = parsed[0][0]
        newest = parsed[-1][0]

        rows.extend(
            parsed
        )

        page_number += 1

        oldest_dt = pd.to_datetime(
            oldest,
            unit="ms",
            utc=True,
        )

        newest_dt = pd.to_datetime(
            newest,
            unit="ms",
            utc=True,
        )

        print(
            f"Req {page_number:04d} | "
            f"+{len(parsed):3d} | "
            f"{oldest_dt.strftime('%Y-%m-%d %H:%M')} "
            f"-> "
            f"{newest_dt.strftime('%Y-%m-%d %H:%M')} | "
            f"bruto {len(rows)}"
        )

        if oldest <= start_ms:
            print(
                "Data inicial desejada alcançada."
            )
            break

        if (
            previous_oldest is not None
            and
            oldest >= previous_oldest
        ):
            print(
                "Paginação não avançou. "
                "Encerrando por segurança."
            )
            break

        previous_oldest = oldest

        # 'after' pede registros anteriores ao timestamp.
        after = oldest

        time.sleep(
            REQUEST_SLEEP
        )

    if not rows:
        raise RuntimeError(
            "Nenhum candle OKX foi baixado."
        )

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

    raw_count = len(
        df
    )

    df.drop_duplicates(
        subset=[
            "timestamp"
        ],
        keep="last",
        inplace=True,
    )

    df.sort_values(
        "timestamp",
        inplace=True,
    )

    # Corta apenas se conseguimos baixar além da data pedida.
    df = df[
        df["timestamp"]
        >= start_ms
    ].copy()

    df[
        "timestamp"
    ] = (
        pd.to_datetime(
            df["timestamp"],
            unit="ms",
            utc=True,
        )
        .dt.tz_localize(
            None
        )
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

    expected = (
        (
            last
            -
            first
        ).total_seconds()
        /
        3600.0
        +
        1.0
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
    print("=" * 88)
    print("DOWNLOAD OKX FINALIZADO")
    print("=" * 88)
    print(
        f"Candles brutos:       "
        f"{raw_count}"
    )
    print(
        f"Candles finais:       "
        f"{len(df)}"
    )
    print(
        f"Primeiro candle:      "
        f"{first}"
    )
    print(
        f"Último candle:        "
        f"{last}"
    )
    print(
        f"Cobertura aproximada: "
        f"{coverage:.2f}%"
    )
    print(
        f"Arquivo salvo:        "
        f"{OUTPUT_FILE}"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
