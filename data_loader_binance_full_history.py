import ccxt
import pandas as pd
import time


# ============================================================
# C2G SYSTEM - BINANCE BTC/USDT FULL AVAILABLE HISTORY
#
# Requested research horizon: ~10 years.
# Binance launched in 2017, so BTC/USDT cannot provide a full
# 10-year Binance history. This loader requests data from
# 2017-07-01 onward and saves whatever Binance actually has.
#
# Timeframe: 1H
# Output: btc_data_binance_full_1h.csv
# ============================================================

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
START_DATE = "2017-07-01T00:00:00Z"
LIMIT_PER_REQUEST = 1000
OUTPUT_FILE = "btc_data_binance_full_1h.csv"

MAX_RETRIES = 8
RETRY_SECONDS = 5


def fetch_full_history():
    exchange = ccxt.binance({
        "enableRateLimit": True,
    })

    print("=" * 76)
    print("C2G SYSTEM - DOWNLOAD FULL BINANCE BTC/USDT 1H HISTORY")
    print("=" * 76)
    print("Carregando mercados da Binance...")

    exchange.load_markets()

    if SYMBOL not in exchange.markets:
        raise RuntimeError(f"{SYMBOL} não encontrado na Binance.")

    since = exchange.parse8601(START_DATE)
    timeframe_ms = exchange.parse_timeframe(TIMEFRAME) * 1000

    end_ms = exchange.milliseconds()

    all_rows = []
    request_number = 0
    last_printed_month = None

    print(f"Símbolo:       {SYMBOL}")
    print(f"Timeframe:     {TIMEFRAME}")
    print(f"Início pedido: {START_DATE}")
    print(f"Arquivo final: {OUTPUT_FILE}")
    print()
    print("Baixando histórico...")
    print()

    while since < end_ms:
        batch = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                batch = exchange.fetch_ohlcv(
                    SYMBOL,
                    timeframe=TIMEFRAME,
                    since=since,
                    limit=LIMIT_PER_REQUEST,
                )
                break

            except Exception as exc:
                print(
                    f"Erro na requisição {request_number + 1} "
                    f"(tentativa {attempt}/{MAX_RETRIES}): {exc}"
                )

                if attempt == MAX_RETRIES:
                    raise

                time.sleep(RETRY_SECONDS)

        if not batch:
            print("A Binance não retornou mais candles.")
            break

        # Mantém somente candles até o momento atual.
        batch = [row for row in batch if row[0] <= end_ms]

        if not batch:
            break

        first_ts = batch[0][0]
        last_ts = batch[-1][0]

        # Proteção contra loop sem progresso.
        if last_ts < since:
            print("Sem progresso no timestamp. Encerrando por segurança.")
            break

        all_rows.extend(batch)
        request_number += 1

        last_dt = pd.to_datetime(last_ts, unit="ms", utc=True)

        # Mostra progresso por lote e evita terminal silencioso.
        print(
            f"Req {request_number:03d} | "
            f"+{len(batch):4d} candles | "
            f"até {last_dt.strftime('%Y-%m-%d %H:%M UTC')} | "
            f"acumulado bruto: {len(all_rows)}"
        )

        next_since = last_ts + timeframe_ms

        if next_since <= since:
            print("Timestamp não avançou. Encerrando.")
            break

        since = next_since

        # Último lote geralmente possui menos de 1000 candles.
        if len(batch) < LIMIT_PER_REQUEST:
            # Não encerramos imediatamente caso a exchange tenha devolvido
            # um lote parcial histórico; tentamos avançar mais uma vez.
            if since >= end_ms:
                break

    if not all_rows:
        raise RuntimeError("Nenhum candle foi baixado.")

    print()
    print("Processando e limpando dados...")

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

    before_dedup = len(df)

    df.drop_duplicates(
        subset=["timestamp"],
        keep="last",
        inplace=True,
    )

    df.sort_values(
        "timestamp",
        inplace=True,
    )

    df = df[df["timestamp"] <= end_ms].copy()

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df.dropna(
        subset=["open", "high", "low", "close"],
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

    df.to_csv(OUTPUT_FILE)

    first_candle = df.index.min()
    last_candle = df.index.max()

    expected_hours = (
        (last_candle - first_candle).total_seconds() / 3600
    ) + 1

    coverage_pct = (
        len(df) / expected_hours * 100
        if expected_hours > 0
        else 0
    )

    print()
    print("=" * 76)
    print("DOWNLOAD FINALIZADO")
    print("=" * 76)
    print(f"Candles brutos:        {before_dedup}")
    print(f"Candles finais:        {len(df)}")
    print(f"Primeiro candle:       {first_candle}")
    print(f"Último candle:         {last_candle}")
    print(f"Cobertura aproximada:  {coverage_pct:.2f}% das horas do período")
    print(f"Arquivo salvo:         {OUTPUT_FILE}")
    print("=" * 76)

    return df


if __name__ == "__main__":
    fetch_full_history()
    