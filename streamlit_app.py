import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fungsi menghitung perubahan volume 5 hari vs 20 hari
def volume_change_5d(volume):
    if len(volume) < 20:
        return np.nan
    prev20 = volume.iloc[-20]
    if prev20 == 0 or np.isnan(prev20):
        return np.nan
    return (volume.iloc[-5] - prev20) / prev20

# Fungsi analisis per saham
def analyze_ticker(ticker):
    try:
        df = yf.download(ticker, period="6mo", interval="1d")
        if df.empty:
            return None

        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['EMA5'] = df['Close'].ewm(span=5).mean()

        vol_change = volume_change_5d(df['Volume'])

        last_close = df['Close'].iloc[-1]
        last_ma20 = df['MA20'].iloc[-1]
        last_ema5 = df['EMA5'].iloc[-1]

        signal = (last_ema5 > last_ma20) and (vol_change > 0.3)

        return {
            "Ticker": ticker,
            "Data": df,
            "Last Close": round(last_close, 2),
            "EMA5": round(last_ema5, 2),
            "MA20": round(last_ma20, 2),
            "Vol Change 5D (%)": round(vol_change * 100, 2) if pd.notna(vol_change) else np.nan,
            "Swing Signal": "✅ BUY" if signal else "❌ HOLD"
        }

    except Exception as e:
        st.error(f"Error {ticker}: {e}")
        return None

# Fungsi untuk plot chart
def plot_chart(df, ticker):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['Close'], label='Close Price', color='blue', linewidth=1)
    ax.plot(df.index, df['EMA5'], label='EMA5', color='orange', linewidth=1.2)
    ax.plot(df.index, df['MA20'], label='MA20', color='green', linewidth=1.2)
    ax.set_title(f"{ticker} - EMA5 & MA20", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

# UI Streamlit
st.title("📊 Swing Trading Screener + Chart - Weekly (yfinance)")

default_tickers = "BBCA.JK, BMRI.JK, TLKM.JK, BBRI.JK, ASII.JK"
ticker_input = st.text_area("Masukkan daftar saham (pisahkan dengan koma)", default_tickers)

if st.button("Jalankan Screener"):
    tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]
    results = [analyze_ticker(t) for t in tickers]
    results = [r for r in results if r is not None]
    
    if results:
        df_results = pd.DataFrame([{k: v for k, v in r.items() if k != "Data"} for r in results])
        st.dataframe(df_results, use_container_width=True)
        
        # Tampilkan chart per saham
        for r in results:
            st.subheader(f"📈 Chart {r['Ticker']}")
            plot_chart(r["Data"], r["Ticker"])
    else:
        st.warning("Tidak ada data yang sesuai kriteria.")
