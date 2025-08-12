import streamlit as st
import pandas as pd
import yfinance as yf
import ta

st.set_page_config(page_title="Screener Swing Trading BEI", layout="wide")

# ===== 1. Ambil daftar saham BEI =====
@st.cache_data
def get_list_saham():
    url = "https://raw.githubusercontent.com/dendi139/beidata/main/idx_tickers.csv"
    df = pd.read_csv(url)
    tickers = df['Ticker'].tolist()
    return tickers, df

# ===== 2. Ambil data harga dari yfinance =====
@st.cache_data
def get_data_yf(ticker):
    df = yf.download(ticker, period="3mo", interval="1d")
    if df.empty:
        return None
    df.dropna(inplace=True)
    return df

# ===== 3. Hitung indikator =====
def hitung_indikator(df):
    df['EMA20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
    df['EMA50'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    return df

# ===== 4. Screening Swing Trading =====
def screener_swing(df):
    latest = df.iloc[-1]
    kondisi = (
        latest['EMA20'] > latest['EMA50'] and
        latest['MACD'] > latest['MACD_signal'] and
        latest['RSI'] < 70
    )
    return kondisi

# ===== 5. UI Streamlit =====
st.title("📈 Screener Swing Trading BEI - 1 Minggu")

tickers, df_listed = get_list_saham()

hasil = []
for ticker in tickers:
    data = get_data_yf(ticker)
    if data is None:
        continue
    data = hitung_indikator(data)
    if screener_swing(data):
        hasil.append({
            "Ticker": ticker,
            "Close": data['Close'].iloc[-1],
            "EMA20": data['EMA20'].iloc[-1],
            "EMA50": data['EMA50'].iloc[-1],
            "MACD": data['MACD'].iloc[-1],
            "RSI": data['RSI'].iloc[-1]
        })

df_hasil = pd.DataFrame(hasil)
st.dataframe(df_hasil)

