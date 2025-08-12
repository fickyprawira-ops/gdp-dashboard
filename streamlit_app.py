"""
swing_screener_with_rsi.py
Streamlit app: Swing Trading 1-Week Screener (Opsi 1) + Chart Harga + MA/EMA + RSI(14)

Dependencies:
    pip install streamlit yfinance pandas numpy matplotlib
Run:
    streamlit run swing_screener_with_rsi.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# -------------------------
# Indikator
# -------------------------
def sma(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder smoothing
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series.fillna(50)

def price_change_5d(close):
    if len(close) < 6: return np.nan
    return (close.iloc[-1] / close.iloc[-6] - 1) * 100

def volume_change_5d(volume):
    if len(volume) < 6: return np.nan
    last5 = volume.iloc[-5:].mean()
    prev20 = volume.iloc[-25:-5].mean() if len(volume) >= 25 else volume.iloc[:-5].mean()
    if prev20 == 0 or np.isnan(prev20): return np.nan
    return (last5 / prev20 - 1) * 100

def average_daily_value(close, volume, window=30):
    return (close * volume).rolling(window=window, min_periods=1).mean().iloc[-1]

# -------------------------
# Analisa per ticker
# -------------------------
def analyze_ticker(ticker, start, end, adv_threshold=5_000_000_000, price_min=200, price_max=5000):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty or len(df) < 50:
        return None, None

    close = df["Close"]
    volume = df["Volume"]

    df["MA5"] = sma(close, 5)
    df["MA20"] = sma(close, 20)
    df["MA50"] = sma(close, 50)
    df["EMA12"] = ema(close, 12)
    df["EMA26"] = ema(close, 26)
    df["RSI14"] = rsi(close, 14)

    rsi14 = df["RSI14"].iloc[-1]
    pc5 = price_change_5d(close)
    vc5 = volume_change_5d(volume)
    adv30 = average_daily_value(close, volume, window=30)
    last_price = close.iloc[-1]

    # Kriteria Opsi 1
    cond_ma = df["MA5"].iloc[-1] > df["MA20"].iloc[-1] > df["MA50"].iloc[-1]
    cond_ema = df["EMA12"].iloc[-1] > df["EMA26"].iloc[-1]
    cond_rsi = 50 <= rsi14 <= 70
    cond_pc5 = (not np.isnan(pc5)) and (pc5 > 3)
    cond_vc5 = (not np.isnan(vc5)) and (vc5 > 20)
    cond_adv = adv30 >= adv_threshold
    cond_price = price_min <= last_price <= price_max

    passes_all = all([cond_ma, cond_ema, cond_rsi, cond_pc5, cond_vc5, cond_adv, cond_price])

    result = {
        "Ticker": ticker,
        "Last Price": float(last_price),
        "Price Chg 5D %": float(pc5) if not np.isnan(pc5) else None,
        "Vol Chg 5D %": float(vc5) if not np.isnan(vc5) else None,
        "RSI14": float(rsi14),
        "ADV30": float(adv30),
        "Passes All": passes_all
    }
    return result, df

# -------------------------
# Plot: Price + MA/EMA + RSI + Volume
# -------------------------
def plot_price_rsi(df, ticker):
    # create two-row layout: price (top) and RSI (bottom)
    fig, (ax_price, ax_rsi) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                           gridspec_kw={"height_ratios": [3, 1]})

    # Price + MA + EMA
    ax_price.plot(df.index, df["Close"], label="Close", linewidth=1.5)
    ax_price.plot(df.index, df["MA5"], label="MA5", linestyle="--")
    ax_price.plot(df.index, df["MA20"], label="MA20", linestyle="--")
    ax_price.plot(df.index, df["EMA12"], label="EMA12")
    ax_price.plot(df.index, df["EMA26"], label="EMA26")
    ax_price.set_title(f"{ticker} — Price, MA & EMA")
    ax_price.set_ylabel("Price")
    ax_price.legend(loc="upper left", fontsize="small")

    # Volume as bar on price axis (semi-transparent)
    ax_vol = ax_price.twinx()
    ax_vol.bar(df.index, df["Volume"], alpha=0.12, label="Volume")
    ax_vol.set_ylabel("Volume")
    ax_vol.get_yaxis().set_visible(False)  # hide secondary y ticks for cleaner look

    # RSI
    ax_rsi.plot(df.index, df["RSI14"], label="RSI(14)", linewidth=1)
    ax_rsi.axhline(70, linestyle="--", linewidth=0.7)
    ax_rsi.axhline(30, linestyle="--", linewidth=0.7)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.legend(loc="upper left", fontsize="small")

    fig.tight_layout()
    return fig

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Swing Screener + RSI", layout="wide")
st.title("📊 Swing Trading Screener 1 Mingguan — Opsi 1 (EMA & Momentum) + RSI(14)")

with st.sidebar:
    st.header("Input & Thresholds")
    tickers_input = st.text_area("Kode saham (pisah koma)", value="BBRI.JK, BBCA.JK, TLKM.JK")
    period_days = st.number_input("History (hari)", min_value=60, max_value=1000, value=240)
    adv_threshold = st.number_input("Avg Daily Value threshold (Rp)", value=5_000_000_000, step=1_000_000)
    price_min = st.number_input("Min last price (Rp)", value=200)
    price_max = st.number_input("Max last price (Rp)", value=5000)
    run_button = st.button("Run Screener")

if run_button:
    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
    if not tickers:
        st.warning("Masukkan minimal satu ticker.")
        st.stop()

    end = datetime.now()
    start = end - timedelta(days=period_days)

    st.info(f"Menjalankan screener untuk {len(tickers)} ticker (mengambil data via yfinance)...")
    results = []
    charts = []

    progress = st.progress(0)
    for i, tk in enumerate(tickers, start=1):
        res, df = analyze_ticker(tk, start, end, adv_threshold=adv_threshold, price_min=price_min, price_max=price_max)
        if res:
            results.append(res)
            if res["Passes All"] and df is not None:
                charts.append((tk, df))
        progress.progress(i / len(tickers))

    if not results:
        st.warning("Tidak ada hasil (mungkin data tidak cukup untuk ticker yang dimasukkan).")
    else:
        df_results = pd.DataFrame(results).sort_values(by=["Passes All", "Price Chg 5D %"], ascending=[False, False])
        st.subheader("Hasil Screener (semua ticker)")
        st.dataframe(df_results.reset_index(drop=True))

        passed = df_results[df_results["Passes All"] == True]
        st.subheader("✅ Saham yang memenuhi semua kriteria")
        if not passed.empty:
            st.table(passed.reset_index(drop=True))
        else:
            st.info("Tidak ada ticker yang memenuhi semua kriteria. Coba longgarkan threshold atau tambahkan ticker lain.")

        # Tampilkan chart untuk saham yang lolos
        if charts:
            st.markdown("---")
            st.subheader("📈 Chart untuk ticker yang lolos")
            for tk, df in charts:
                st.markdown(f"**{tk}**")
                fig = plot_price_rsi(df, tk)
                st.pyplot(fig)
        else:
            st.info("Tidak ada chart karena tidak ada ticker yang lolos kriteria.")
else:
    st.info("Atur ticker dan threshold di sidebar lalu klik **Run Screener**.")
