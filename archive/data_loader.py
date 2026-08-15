import ccxt
import pandas as pd

def fetch_btc_data(symbol='BTC/USDT', timeframe='1h', limit=500):
    exchange = ccxt.binance()
    print(f"Buscando {limit} candles de {symbol} no timeframe {timeframe}...")
    
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        print("Dados baixados com sucesso!")
        return df
    except Exception as e:
        print(f"Erro ao buscar dados: {e}")
        return None

if __name__ == "__main__":
    df = fetch_btc_data(symbol='BTC/USDT', timeframe='1h', limit=500)
    if df is not None:
        df.to_csv('btc_data.csv')
        print(df.head())
        print("\nArquivo 'btc_data.csv' gerado com sucesso!")
        