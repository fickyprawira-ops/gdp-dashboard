import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Screener IHSG Swing Trading", layout="wide")

# 1. Ambil daftar saham BEI langsung dari IDX
@st.cache_data
def get_list_saham():
    url = "https://www.idx.co.id/data-pasar/data-saham/daftar-saham/"
    tables = pd.read_html(url)
    df_saham = tables[0]
    df_saham.columns = df_saham.columns.str.strip()
    tickers = [kode + ".JK" for kode in df_saham["Kode"]]
    return tickers, df_saham

tickers, df_listed = get_list_saham()

# 2. Fungsi ambil data dari yfinance & analisis crossover EMA5 > MA20
def analyze_ticker(ticker):
    try:
        df = yf.download(ticker, period="3mo", interval="1d", progress=False)
        if df.empty or len(df) < 20:
            return None
        df["EMA5"] = df["Close"].ewm(span=5, adjust=False).mean()
        df["MA20"] = df["Close"].rolling(window=20).mean()
        last, prev = df.iloc[-1], df.iloc[-2]
        signal = (last["EMA5"] > last["MA20"]) and (prev["EMA5"] <= prev["MA20"])
        return {
            "Ticker": ticker,
            "Signal": "BUY" if signal else "HOLD",
            "df": df
        }
    except:
        return None

st.title("📊 Screener Swing Trading IHSG (EMA5 vs MA20)")

if st.button("🔍 Jalankan Screener"):
    results = []
    progress = st.progress(0)
    for i, t in enumerate(tickers):
        res = analyze_ticker(t)
        if res:
            results.append(res)
        progress.progress((i + 1) / len(tickers))
    
    # Filter BUY saja
    buy_list = [r for r in results if r["Signal"] == "BUY"]

    if buy_list:
        st.subheader(f"📈 {len(buy_list)} Saham Sinyal BUY Minggu Ini")
        df_res = pd.DataFrame([{"Ticker": r["Ticker"], "Signal": r["Signal"]} for r in buy_list])
        st.dataframe(df_res)

        for r in buy_list:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(r["df"].index, r["df"]["Close"], label="Close", color="blue")
            ax.plot(r["df"].index, r["df"]["EMA5"], label="EMA5", color="green")
            ax.plot(r["df"].index, r["df"]["MA20"], label="MA20", color="red")
            ax.set_title(r["Ticker"])
            ax.legend()
            st.pyplot(fig)
    else:
        st.warning("Tidak ada saham yang memenuhi sinyal BUY minggu ini.")

st.caption("Data diambil real-time dari Yahoo Finance (.JK) & daftar saham IDX")
