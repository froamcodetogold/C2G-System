import ccxt
import pandas as pd
import time

# ============================================================
# CONFIGURAÇÃO
# ============================================================

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
YEARS = 4
LIMIT_PER_REQUEST = 1000
OUTPUT_FILE = "btc_data.csv"


def fetch_btc_history(
    symbol=SYMBOL,
    timeframe=TIMEFRAME,
    years=YEARS,
    limit_per_request=LIMIT_PER_REQUEST,
    output_file=OUTPUT_FILE,
):
    """
    Baixa histórico OHLCV paginado da Binance via CCXT.

    - Busca os últimos `years` anos.
    - Faz paginação usando o timestamp do último candle recebido.
    - Remove duplicatas.
    - Salva em btc_data.csv no formato esperado pelo c2g_engine.py.
    """

    exchange = ccxt.binance({
        "enableRateLimit": True,
    })

    # Carrega mercados e valida o símbolo/timeframe.
    exchange.load_markets()

    if symbol not in exchange.markets:
        raise ValueError(f"Símbolo não encontrado na Binance: {symbol}")

    if timeframe not in exchange.timeframes:
        raise ValueError(
            f"Timeframe '{timeframe}' não suportado. "
            f"Disponíveis: {list(exchange.timeframes.keys())}"
        )

    # Data inicial = exatamente 4 anos antes do momento atual.
    end_ts = pd.Timestamp.now(tz="UTC")
    start_ts = end_ts - pd.DateOffset(years=years)

    since = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)

    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000

    print("=" * 70)
    print("C2G SYSTEM - DOWNLOAD HISTÓRICO BTC")
    print("=" * 70)
    print(f"Exchange:   Binance")
    print(f"Símbolo:    {symbol}")
    print(f"Timeframe:  {timeframe}")
    print(f"Início UTC: {start_ts}")
    print(f"Fim UTC:    {end_ts}")
    print(f"Arquivo:    {output_file}")
    print("=" * 70)

    all_rows = []
    request_count = 0
    last_progress_ts = None

    while since < end_ms:
        request_count += 1

        try:
            batch = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit_per_request,
            )
        except Exception as exc:
            print(f"\nErro na requisição {request_count}: {exc}")
            print("Aguardando 5 segundos e tentando novamente...")
            time.sleep(5)
            continue

        if not batch:
            print("\nNenhum candle adicional retornado. Encerrando.")
            break

        # Remove qualquer candle além do momento final desejado.
        batch = [row for row in batch if row[0] <= end_ms]

        if not batch:
            break

        all_rows.extend(batch)

        first_dt = pd.to_datetime(batch[0][0], unit="ms", utc=True)
        last_dt = pd.to_datetime(batch[-1][0], unit="ms", utc=True)

        print(
            f"Req {request_count:03d} | "
            f"+{len(batch):4d} candles | "
            f"{first_dt} -> {last_dt} | "
            f"Total bruto: {len(all_rows)}"
        )

        last_timestamp = batch[-1][0]

        # Proteção contra loop infinito caso a exchange devolva
        # repetidamente o mesmo último timestamp.
        if last_progress_ts is not None and last_timestamp <= last_progress_ts:
            print("Sem avanço de timestamp. Encerrando para evitar loop infinito.")
            break

        last_progress_ts = last_timestamp

        # Próximo candle após o último recebido.
        since = last_timestamp + timeframe_ms

        # Se o lote veio menor do que o limite, normalmente chegamos
        # ao trecho mais recente disponível.
        if len(batch) < limit_per_request and since >= end_ms:
            break

    if not all_rows:
        raise RuntimeError("Nenhum candle foi baixado.")

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

    # Remove duplicatas e ordena.
    df.drop_duplicates(subset="timestamp", keep="last", inplace=True)
    df.sort_values("timestamp", inplace=True)

    # Timestamp UTC sem timezone no CSV para compatibilidade simples
    # com pandas/mplfinance no c2g_engine.
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        .dt.tz_localize(None)
    )

    # Garante valores numéricos.
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["open", "high", "low", "close"], inplace=True)

    # Mantém apenas o intervalo pedido.
    start_naive = start_ts.tz_localize(None)
    end_naive = end_ts.tz_localize(None)

    df = df[
        (df["timestamp"] >= start_naive)
        & (df["timestamp"] <= end_naive)
    ].copy()

    df.set_index("timestamp", inplace=True)

    df.to_csv(output_file)

    print("\n" + "=" * 70)
    print("DOWNLOAD CONCLUÍDO")
    print("=" * 70)
    print(f"Candles finais: {len(df)}")
    print(f"Primeiro candle: {df.index.min()}")
    print(f"Último candle:   {df.index.max()}")
    print(f"Arquivo salvo:   {output_file}")
    print("=" * 70)

    return df


if __name__ == "__main__":
    fetch_btc_history()
    