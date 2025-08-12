"""
swing_screener.py
Streamlit app: Swing Trading 1-Week Screener (Opsi 1 - EMA & Momentum)
Dependencies:
    pip install streamlit yfinance pandas numpy
Run:
    streamlit run swing_screener.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# -------------------------
# Utility / Indicator funcs
# -------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    # Wilder RSI smoothing fallback if avg_loss==0
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # neutral where not enough data
    return rsi

def price_change_5d(close: pd.Series) -> float:
    # percent change over last 5 trading days (compare close[-1] to close[-6])
    if len(close) < 6:
        return np.nan
    return (close.iloc[-1] / close.iloc[-6] - 1) * 100

def volume_change_5d(volume: pd.Series) -> float:
    # compares avg volume last 5 days to avg volume previous 20 days (if available)
    if len(volume) < 6:
        return np.nan
    last5 = volume.iloc[-5:].mean()
    prev20 = volume.iloc[-25:-5]  # previous 20 trading days before the last5
    if len(prev20) >= 5:
        prev20_mean = prev20.mean()
        if prev20_mean == 0:
            return np.nan
        return (last5 / prev20_mean - 1) * 100
    else:
        # fallback to comparing last5 to overall mean excluding last5
        prev_mean = volume.iloc[:-5].mean() if len(volume) > 5 else volume.mean()
        if prev_mean == 0:
            return np.nan
        return (last5 / prev_mean - 1) * 100

def average_daily_value(close: pd.Series, volume: pd.Series, window: int = 30) -> float:
    adv = (close * volume).rolling(window=window, min_periods=1).mean()
    return adv.iloc[-1]

# -------------------------
# Screener logic per ticker
# -------------------------
def analyze_ticker(ticker: str, period_days: int = 180):
    """
    Download data and compute screener metrics.
    Return dict with metrics and flags.
    """
    try:
        # fetch last period_days + padding
        end = datetime.now()
        start = end - timedelta(days=period_days)
        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False)
        if df.empty or len(df) < 30:
            return {"ticker": ticker, "error": "not enough data"}

        close = df['Close']
        vol = df['Volume']

        # moving averages
        ma5 = sma(close, 5).iloc[-1]
        ma20 = sma(close, 20).iloc[-1]
        ma50 = sma(close, 50).iloc[-1]

        # EMAs
        ema12 = ema(close, 12).iloc[-1]
        ema26 = ema(close, 26).iloc[-1]

        # RSI
        rsi14 = rsi(close, 14).iloc[-1]

        # price change 5d
        pc5 = price_change_5d(close)

        # volume change 5d
        vc5 = volume_change_5d(vol)

        # avg daily value 30
        adv30 = average_daily_value(close, vol, window=30)

        last_price = close.iloc[-1]

        # boolean checks (criteria from Opsi 1)
        cond_ma = (ma5 > ma20) and (ma20 > ma50)
        cond_ema = (ema12 > ema26)  # momentum positive
        cond_rsi = (rsi14 >= 50) and (rsi14 <= 70)
        cond_pc5 = (pc5 is not np.nan) and (pc5 > 3)
        cond_vc5 = (vc5 is not np.nan) and (vc5 > 20)
        cond_adv = (adv30 >= 5_000_000_000)  # avg daily value threshold
        cond_price_range = (last_price >= 200) and (last_price <= 5000)

        result = {
            "ticker": ticker,
            "last_price": float(last_price),
            "ma5": float(ma5),
            "ma20": float(ma20),
            "ma50": float(ma50),
            "ema12": float(ema12),
            "ema26": float(ema26),
            "rsi14": float(rsi14),
            "price_change_5d": float(pc5) if not np.isnan(pc5) else None,
            "volume_change_5d": float(vc5) if not np.isnan(vc5) else None,
            "avg_daily_value_30": float(adv30),
            "cond_ma": cond_ma,
            "cond_ema": cond_ema,
            "cond_rsi": cond_rsi,
            "cond_pc5": cond_pc5,
            "cond_vc5": cond_vc5,
            "cond_adv": cond_adv,
            "cond_price_range": cond_price_range,
            "passes_all": all([cond_ma, cond_ema, cond_rsi, cond_pc5, cond_vc5, cond_adv, cond_price_range])
        }
        return result
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Swing Screener 1W (Opsi 1)", layout="wide")
st.title("Screener Swing Trading 1 Mingguan — Opsi 1 (EMA & Momentum)")

st.markdown(
    """
    **Cara pakai**:
    - Masukkan daftar ticker (comma-separated) atau upload file CSV (kolom `ticker`).
    - Klik **Run Screener**. App akan memanggil data historis (yfinance) dan menghitung indikator.
    - Hasil: tabel saham yang lulus semua kriteria beserta metriknya.
    """
)

# Sidebar inputs
st.sidebar.header("Input & Thresholds")
tickers_input = st.sidebar.text_area("Tickers (comma-separated)", value="BBCA.JK, BBRI.JK, TLKM.JK")
uploaded_file = st.sidebar.file_uploader("Atau upload CSV berisi kolom 'ticker'", type=["csv"])
period_days = st.sidebar.number_input("History (days)", value=240, min_value=60, max_value=2000)
adv_threshold = st.sidebar.number_input("Avg Daily Value threshold (Rp)", value=5_000_000_000, step=1_000_000)
price_min = st.sidebar.number_input("Min last price (Rp)", value=200)
price_max = st.sidebar.number_input("Max last price (Rp)", value=5000)
run_button = st.sidebar.button("Run Screener")

# get tickers list
tickers = []
if uploaded_file is not None:
    try:
        df_upload = pd.read_csv(uploaded_file)
        if 'ticker' in df_upload.columns:
            tickers = df_upload['ticker'].astype(str).str.strip().tolist()
        else:
            st.sidebar.error("CSV harus punya kolom 'ticker'")
    except Exception as e:
        st.sidebar.error("Error membaca CSV: " + str(e))

if tickers_input and not tickers:
    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

# Replace thresholds in analysis functions by reading sidebar values:
# (we'll use them when evaluating the results below)

if not tickers:
    st.warning("Masukkan minimal satu ticker di sidebar (mis. BBCA.JK).")
    st.stop()

# Run screener
if run_button:
    st.info(f"Menjalankan screener untuk {len(tickers)} ticker... (mengambil data via yfinance)")
    placeholder = st.empty()
    results = []
    for i, tk in enumerate(tickers, start=1):
        placeholder.text(f"Processing {i}/{len(tickers)}: {tk}")
        res = analyze_ticker(tk, period_days=period_days)
        # adjust adv threshold & price range per sidebar
        if 'error' not in res:
            # override adv & price conditions with sidebar values
            res['cond_adv'] = (res.get('avg_daily_value_30', 0) >= adv_threshold)
            res['cond_price_range'] = (res.get('last_price', 0) >= price_min) and (res.get('last_price', 0) <= price_max)
            # recompute passes_all
            checks = [
                res.get('cond_ma', False),
                res.get('cond_ema', False),
                res.get('cond_rsi', False),
                res.get('cond_pc5', False),
                res.get('cond_vc5', False),
                res.get('cond_adv', False),
                res.get('cond_price_range', False)
            ]
            res['passes_all'] = all(checks)
        results.append(res)

    placeholder.empty()

    df_results = pd.DataFrame(results)
    # show errors separately
    if 'error' in df_results.columns:
        errors = df_results[df_results['error'].notna()][['ticker', 'error']]
        if not errors.empty:
            st.subheader("Errors / Not enough data")
            st.dataframe(errors)

    # Clean display table
    display_cols = [
        "ticker", "last_price", "price_change_5d", "volume_change_5d",
        "ma5", "ma20", "ma50", "ema12", "ema26", "rsi14",
        "avg_daily_value_30", "passes_all"
    ]
    st.subheader("Hasil Screener (semua ticker)")
    st.dataframe(df_results[display_cols].sort_values(by=["passes_all", "price_change_5d"], ascending=[False, False]).reset_index(drop=True))

    # show passed tickers
    passed = df_results[df_results['passes_all'] == True]
    if not passed.empty:
        st.success(f"{len(passed)} ticker memenuhi semua kriteria:")
        st.table(passed[display_cols].reset_index(drop=True))
    else:
        st.info("Tidak ada ticker yang memenuhi semua kriteria. Coba longgarkan threshold atau tambahkan lebih banyak ticker.")

    st.markdown("**Catatan:** hasil bergantung pada data yfinance. Untuk universe lengkap IDX, gunakan daftar ticker IDX terbaru (mis. dari file CSV).")
else:
    st.info("Tekan **Run Screener** di sidebar untuk mulai.")
