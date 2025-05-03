# btc_candlestick_predictor.py

# Step 1: Install Dependencies
# Run this once in your environment
# pip install python-binance pandas numpy ta matplotlib scikit-learn streamlit tensorflow keras

from binance.client import Client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import ta
import streamlit as st
import os

# Step 2: Binance API Setup
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
client = Client(api_key, api_secret)

# Step 3: Fetch Historical Data
def fetch_btc_klines(interval='1h', lookback='7 days ago UTC'):
    klines = client.get_historical_klines("BTCUSDT", interval, lookback)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

# Step 4: Add Technical Indicators
def add_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['macd'] = ta.trend.MACD(df['close']).macd()
    df['ema'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    df['volume_ema'] = ta.trend.EMAIndicator(df['volume'], window=20).ema_indicator()
    df.dropna(inplace=True)
    return df

# Step 5: Label Data for Supervised Learning
def label_candle(df):
    df['target'] = df['close'].shift(-1) > df['close']
    df['target'] = df['target'].astype(int)
    return df

# Step 6: Train a Model
def train_model(df):
    X = df[['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'ema', 'volume_ema']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    report = classification_report(y_test, preds)
    return model, preds, y_test, report

# Step 7: Visualization

def plot_predictions(df, preds):
    df_test = df.iloc[-len(preds):].copy()
    df_test['predicted'] = preds
    plt.figure(figsize=(14, 6))
    plt.plot(df_test.index, df_test['close'], label='Close Price')
    plt.scatter(df_test.index[df_test['predicted'] == 1], df_test['close'][df_test['predicted'] == 1],
                marker='^', color='green', label='Predicted UP')
    plt.scatter(df_test.index[df_test['predicted'] == 0], df_test['close'][df_test['predicted'] == 0],
                marker='v', color='red', label='Predicted DOWN')
    plt.legend()
    plt.title("BTC/USDT Candlestick Prediction")
    plt.show()

# Step 8: Streamlit Web App

def run_streamlit_app():
    df = fetch_btc_klines()
    df = add_indicators(df)
    df = label_candle(df)
    model, preds, y_test, report = train_model(df)

    st.title("Bitcoin Candlestick Predictor")
    st.line_chart(df['close'][-100:])
    st.subheader("Prediction for Next Candle")
    st.write("📈 UP" if preds[-1] == 1 else "📉 DOWN")
    st.text(report)

if __name__ == "__main__":
    # For Streamlit, use: streamlit run btc_candlestick_predictor.py
    run_streamlit_app()
