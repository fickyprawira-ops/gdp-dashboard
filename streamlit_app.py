import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime

st.title("📊 Screener Swing Trading Mingguan")

# Input ticker saham
tickers = st.text_area(
    "Masukkan daftar kode saham (pisahkan dengan koma)",
    "BBCA.JK,BBRI.JK,TLKM.JK,ASII.JK,UNVR.JK"
).upper().split(",")

# Parameter EMA
ema_short = st.number_input("EMA Pendek (hari)", min_value=3, max_value=20, value=5)
ema_long = st.number_input("EMA Panjang (hari)", min_value=10, max_value=50, value=20)

# Ambil data 3 bulan terakhir (supaya cukup data untuk MA)
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=90)

data_list = []

for ticker in tickers:
    ticker = ticker.strip()
    try:
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
        if df.empty:
            continue

        # Hitung EMA
        df["EMA_Short"] = df["Close"].ewm(span=ema_short, adjust=False).mean()
        df["EMA_Long"] = df["Close"].ewm(span=ema_long, adjust=False).mean()

        # Sinyal swing trading: EMA pendek memotong EMA panjang dari bawah
        # Cek hanya baris terakhir
        ema_cross_up = (
            df["EMA_Short"].iloc[-1] > df["EMA_Long"].iloc[-1] and
            df["EMA_Short"].iloc[-2] <= df["EMA_Long"].iloc[-2]
        )

        if ema_cross_up:
            data_list.append({
                "Ticker": ticker,
                "Close": round(df["Close"].iloc[-1], 2),
                f"EMA_{ema_short}": round(df["EMA_Short"].iloc[-1], 2),
                f"EMA_{ema_long}": round(df["EMA_Long"].iloc[-1], 2),
                "Sinyal": "Buy"
            })
    except Exception as e:
        st.write(f"Error {ticker}: {e}")

# Tampilkan hasil
if data_list:
    result_df = pd.DataFrame(data_list)
    st.dataframe(result_df)
else:
    st.warning("Tidak ada saham yang memenuhi kriteria.")

