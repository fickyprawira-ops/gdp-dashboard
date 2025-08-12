import yfinance as yf
import pandas as pd
import numpy as np

def volume_change_5d(volume):
    """
    Menghitung persentase perubahan volume dari 20 hari yang lalu ke 5 hari yang lalu.
    """
    if len(volume) < 20:
        return np.nan  # Data tidak cukup
    prev20 = volume.iloc[-20]
    if prev20 == 0 or np.isnan(prev20):
        return np.nan
    return (volume.iloc[-5] - prev20) / prev20

def analyze_ticker(ticker):
    """
    Mengambil data dari Yahoo Finance dan menghitung sinyal swing trading.
    """
    try:
        df = yf.download(ticker, period="6mo", interval="1d")
        if df.empty:
            return None

        # Moving Average
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['EMA5'] = df['Close'].ewm(span=5).mean()

        # Volume Change 5D
        vol_change = volume_change_5d(df['Volume'])

        # Sinyal Buy Swing
        last_close = df['Close'].iloc[-1]
        last_ma20 = df['MA20'].iloc[-1]
        last_ema5 = df['EMA5'].iloc[-1]

        signal = (last_ema5 > last_ma20) and (vol_change > 0.3)

        return {
            "Ticker": ticker,
            "Last Close": last_close,
            "EMA5": last_ema5,
            "MA20": last_ma20,
            "Vol Change 5D": vol_change,
            "Swing Signal": signal
        }

    except Exception as e:
        print(f"Error {ticker}: {e}")
        return None

# Contoh penggunaan
tickers = ["BBCA.JK", "BMRI.JK", "TLKM.JK"]
results = [analyze_ticker(t) for t in tickers]
df_results = pd.DataFrame([r for r in results if r is not None])
print(df_results)
