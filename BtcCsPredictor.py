# pip install pandas numpy ta scikit-learn matplotlib streamlit requests

import requests
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import streamlit as st

# Step 1: Get OHLC from CoinGecko
def get_ohlc(days=1):
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/ohlc?vs_currency=usd&days={days}"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# Step 2: Add Indicators
def add_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['macd'] = ta.trend.MACD(df['close']).macd()
    df['ema'] = ta.trend.EMAIndicator(df['close'], window=10).ema_indicator()
    df.dropna(inplace=True)
    return df

# Step 3: Label as UP/DOWN
def label_data(df):
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)
    return df

# Step 4: Train Model
def train_model(df):
    X = df[['open', 'high', 'low', 'close', 'rsi', 'macd', 'ema']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds, y_test, classification_report(y_test, preds, output_dict=True)

# Step 5: Streamlit App
def app():
    df = get_ohlc(days=3)
    df = add_indicators(df)
    df = label_data(df)
    model, preds, y_test, report = train_model(df)

    st.title(\"🔮 Bitcoin Candlestick Predictor (Free CoinGecko API)\")
    st.line_chart(df['close'][-100:])
    st.subheader(\"📊 Next Candle Prediction:\")
    st.write(\"📈 UP\" if preds[-1] == 1 else \"📉 DOWN\")
    st.subheader(\"📋 Model Report:\")
    st.json(report)

if __name__ == \"__main__\":
    app()
