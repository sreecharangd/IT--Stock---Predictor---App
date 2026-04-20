"""
train.py — Train the Random Forest model on IT sector stock CSVs.
Run this locally once before pushing to GitHub:
    python train.py
It saves model/, scaler.pkl, and stock_encoder.pkl into the model/ folder.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ── 1. Load all CSVs ──────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

STOCKS = ["TCS", "INFOSYS", "WIPRO", "HCLTECH", "TECHM"]

def clean_num(s):
    """Remove Indian-style commas and cast to float."""
    if pd.isna(s):
        return np.nan
    try:
        return float(str(s).replace(",", "").strip())
    except ValueError:
        return np.nan

dfs = []
for sym in STOCKS:
    path = os.path.join(DATA_DIR, f"{sym}.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Symbol"] = sym
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# ── 2. Clean & engineer features ─────────────────────────────────────────────
df["Open"]   = df["Open Price"].apply(clean_num)
df["High"]   = df["High Price"].apply(clean_num)
df["Low"]    = df["Low Price"].apply(clean_num)
df["Close"]  = df["Close Price"].apply(clean_num)
df["Volume"] = df["Total Traded Quantity"].apply(clean_num)

df["Date"]  = pd.to_datetime(df["Date"], format="%d-%b-%y", errors="coerce")
df["Day"]   = df["Date"].dt.day
df["Month"] = df["Date"].dt.month
df["Year"]  = df["Date"].dt.year

df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume", "Day", "Month", "Year"])
df = df.sort_values(["Symbol", "Date"])

# Target: next trading day's close price
df["Target"] = df.groupby("Symbol")["Close"].shift(-1)
df = df.dropna(subset=["Target"])

# Encode stock symbol
le = LabelEncoder()
df["Symbol_Encoded"] = le.fit_transform(df["Symbol"])

# ── 3. Train / test split ─────────────────────────────────────────────────────
FEATURES = ["Open", "High", "Low", "Close", "Volume", "Day", "Month", "Year", "Symbol_Encoded"]
X = df[FEATURES]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 4. Fit model ──────────────────────────────────────────────────────────────
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_s, y_train)

score = model.score(X_test_s, y_test)
print(f"✅  Model trained  |  R² on test set: {score:.4f}")
print(f"    Stocks encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ── 5. Save artefacts ─────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(model,  "model/stock_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(le,     "model/stock_encoder.pkl")
print("✅  Saved → model/stock_model.pkl, model/scaler.pkl, model/stock_encoder.pkl")
