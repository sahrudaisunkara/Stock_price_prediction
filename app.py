import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Stock Market Prediction System",
    page_icon="📈",
    layout="centered"
)

st.title("📊 Stock Market Prediction System")
st.subheader("Predict Next Trading Session High & Low")
st.markdown("---")

WINDOW_SIZE = 90

# -------------------------------------------------
# Stock list (DISPLAY NAME -> TICKER)
# MUST match Jupyter training tickers
# -------------------------------------------------
stock_list = {
    "Infosys": "INFY.NS",
    "TCS": "TCS.NS",
    "Reliance": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "ITC": "ITC.NS",
    "SBI": "SBIN.NS",
    "L&T": "LT.NS",
    "Apple": "AAPL",
    "Microsoft": "MSFT"
}

selected_stock = st.selectbox(
    "Select Stock",
    options=list(stock_list.keys())
)

ticker = stock_list[selected_stock]

# -------------------------------------------------
# Load ML components per stock (cached)
# -------------------------------------------------
@st.cache_resource
def load_ml_components(ticker):
    model = load_model(f"models/{ticker}_model.h5", compile=False)
    X_scaler = joblib.load(f"scalers/{ticker}_X_scaler.pkl")
    y_scaler = joblib.load(f"scalers/{ticker}_y_scaler.pkl")
    return model, X_scaler, y_scaler


# -------------------------------------------------
# Prediction logic
# -------------------------------------------------
if st.button("Predict Next Session"):
    with st.spinner("Fetching live data & predicting..."):
        try:
            # Load model & scalers
            model, X_scaler, y_scaler = load_ml_components(ticker)

            # Fetch live data
            df = yf.download(ticker, period="1y")

            # Safety: flatten columns (yfinance MultiIndex issue)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.reset_index(inplace=True)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.dropna(inplace=True)

            if len(df) < WINDOW_SIZE:
                st.error("Not enough recent data to make prediction.")
                st.stop()

            # -------------------------------------------------
            # Moving Average & Direction (comparison signal)
            # -------------------------------------------------
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()

            latest_ma20 = df['MA20'].iloc[-1]
            latest_ma50 = df['MA50'].iloc[-1]

            if latest_ma20 > latest_ma50:
                direction = "Bullish 📈"
                direction_color = "green"
            else:
                direction = "Bearish 📉"
                direction_color = "red"

            # -------------------------------------------------
            # Prepare LSTM input
            # -------------------------------------------------
            X_data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
            X_scaled = X_scaler.transform(X_data)

            last_window = X_scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 5)

            # Predict (scaled)
            y_pred_scaled = model.predict(last_window)

            # Inverse scale (CORRECT way)
            y_pred = y_scaler.inverse_transform(y_pred_scaled)

            predicted_high = y_pred[0, 0]
            predicted_low = y_pred[0, 1]

        except FileNotFoundError:
            st.error(
                "Model or scaler file not found for this stock. "
                "Please ensure it was trained and saved correctly."
            )
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

    # -------------------------------------------------
    # Display results
    # -------------------------------------------------
    st.success(f"Prediction for **{selected_stock}**")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("📈 Predicted High", f"{predicted_high:.2f}")
    with col2:
        st.metric("📉 Predicted Low", f"{predicted_low:.2f}")

    st.markdown("---")
    st.subheader("📊 Trend Comparison (Moving Average)")

    st.markdown(
        f"<h3 style='color:{direction_color};'>Direction: {direction}</h3>",
        unsafe_allow_html=True
    )

    st.write(f"**MA20:** {latest_ma20:.2f}")
    st.write(f"**MA50:** {latest_ma50:.2f}")

    st.info(
        "Direction is derived using Moving Average crossover (MA20 vs MA50) "
        "and is shown as a comparison signal alongside AI-based price predictions."
    )
