import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
app_dir = os.path.dirname(os.path.abspath(__file__))
model         = joblib.load(os.path.join(app_dir, "model", "stock_model.pkl"))
scaler        = joblib.load(os.path.join(app_dir, "model", "scaler.pkl"))
stock_encoder = joblib.load(os.path.join(app_dir, "model", "stock_encoder.pkl"))

FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "Day", "Month", "Year", "Symbol_Encoded"]
STOCKS = ["HCLTECH", "INFOSYS", "TCS", "TECHM", "WIPRO"]

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="IT Stock Predictor", page_icon="📈")
st.title("📈 IT Stock Price Prediction")
st.caption("Predicts next trading day's close price using a Random Forest model.")

st.warning(
    "⚠️ **Disclaimer**: Predictions are based on historical patterns only and "
    "should NOT be used as financial advice."
)

with st.form("prediction_form"):
    st.subheader("Enter Stock Details")

    col1, col2 = st.columns(2)
    with col1:
        stock       = st.selectbox("Stock Symbol", STOCKS)
        open_price  = st.number_input("Open Price (₹)",  value=1000.0, step=0.5, format="%.2f")
        high_price  = st.number_input("High Price (₹)",  value=1050.0, step=0.5, format="%.2f")
        low_price   = st.number_input("Low Price (₹)",   value=950.0,  step=0.5, format="%.2f")
    with col2:
        close_price = st.number_input("Close Price (₹)", value=1020.0, step=0.5, format="%.2f")
        volume      = st.number_input("Volume Traded",   value=100000, step=1000, format="%d")
        day         = st.number_input("Day",   value=1,    min_value=1,  max_value=31)
        month       = st.number_input("Month", value=4,    min_value=1,  max_value=12)
        year        = st.number_input("Year",  value=2026, min_value=2000, max_value=2100)

    submitted = st.form_submit_button("🔮 Predict Next Close Price", use_container_width=True)

if submitted:
    try:
        symbol_encoded = int(stock_encoder.transform([stock])[0])
    except ValueError:
        st.error(f"Stock '{stock}' not recognised by the model.")
        st.stop()

    input_df = pd.DataFrame(
        [{
            "Open": open_price, "High": high_price, "Low": low_price,
            "Close": close_price, "Volume": float(volume),
            "Day": day, "Month": month, "Year": year,
            "Symbol_Encoded": symbol_encoded,
        }],
        columns=FEATURE_COLUMNS,
    )

    input_scaled = scaler.transform(input_df)
    prediction   = model.predict(input_scaled)[0]

    st.success(f"**{stock}** — Predicted Next Close Price: **₹ {prediction:,.2f}**")

    # Show delta from today's close
    delta = prediction - close_price
    direction = "▲" if delta >= 0 else "▼"
    st.metric(
        label="Change vs Today's Close",
        value=f"₹ {prediction:,.2f}",
        delta=f"{direction} ₹ {abs(delta):,.2f}  ({delta/close_price*100:.2f}%)",
    )

st.divider()
st.caption(
    "Model: Random Forest Regressor trained on NSE IT sector data "
    "(TCS, Infosys, Wipro, HCL Tech, Tech Mahindra)."
)
