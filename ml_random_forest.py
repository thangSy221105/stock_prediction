import os
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
import joblib

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands

# ==== CONFIG ====
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "stock_db"
REALTIME_COLLECTION = "stock_realtime_AAPL"
PREDICTED_COLLECTION = "predicted_output_10min_forest_AAPL"
FUTURE_SHIFT_MINUTES = 10

# ==== LOGGING ====
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("ml_random_forest.log"),
        logging.StreamHandler()
    ]
)

# ==== CONNECT DB ====
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

FEATURE_COLS = [
    'sma5', 'ema5', 'ema10', 'price_change',
    'volatility', 'momentum', 'rsi14', 'macd', 'bb_width'
]

# ==== FEATURE ENGINEERING ====
def calculate_features(df):
    df = df.copy()
    df['sma5'] = df['close'].rolling(window=5).mean()
    df['ema5'] = EMAIndicator(close=df['close'], window=5).ema_indicator()
    df['ema10'] = EMAIndicator(close=df['close'], window=10).ema_indicator()
    df['price_change'] = df['close'].pct_change()
    df['volatility'] = df['close'].rolling(window=5).std()
    df['momentum'] = df['close'] - df['close'].shift(5)
    df['rsi14'] = RSIIndicator(close=df['close'], window=14).rsi()
    df['macd'] = MACD(close=df['close']).macd_diff()
    bb = BollingerBands(close=df['close'])
    df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    return df

# ==== PREDICT FUNCTION ====
def predict_next(df_feat, model, scaler, df_raw):
    X = df_feat[FEATURE_COLS]
    X_scaled = scaler.transform(X)
    last = df_feat.iloc[-1]
    pred = model.predict([X_scaled[-1]])[0]
    prob = model.predict_proba([X_scaled[-1]])[0]
    conf = max(prob) * 100

    last_raw = df_raw.iloc[-1]
    prediction_time = last_raw['timestamp'] + timedelta(minutes=FUTURE_SHIFT_MINUTES)

    output = {
        "symbol": "AAPL",
        "timestamp": prediction_time.strftime("%Y-%m-%d %H:%M:%S"),
        "open": float(last_raw['open']),
        "high": float(last_raw['high']),
        "low": float(last_raw['low']),
        "close": float(last_raw['close']) * (1 + (0.0007 if int(pred) else -0.0007)),
        "volume": int(last_raw['volume']),
        "prediction": int(pred),
        "confidence": float(round(conf, 2)),
        "predicted_at": last_raw['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
        "model_version": "random_forest_v1"
    }

    db[PREDICTED_COLLECTION].insert_one(output)
    logging.info(f"✅ Prediction saved at {output['timestamp']} — Pred: {pred} | Conf: {conf:.2f}%")

# ==== MAIN ====
def run_prediction():
    logging.info("🚀 Starting prediction on AAPL realtime data")
    data = list(db[REALTIME_COLLECTION].find({}).sort("timestamp", 1))
    if len(data) < 50:
        logging.warning("❌ Not enough data")
        return

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df_feat = calculate_features(df)

    # === Train if needed ===
    model_path = "saved_models/rf_aapl.pkl"
    scaler_path = "saved_models/rf_scaler_aapl.pkl"

    if not os.path.exists(model_path):
        logging.info("🔧 Training model...")
        df_train = df_feat.copy()
        df_train['future_close'] = df_train['close'].shift(-FUTURE_SHIFT_MINUTES // 5)
        df_train.dropna(inplace=True)
        df_train['movement'] = (df_train['future_close'] > df_train['close']).astype(int)
        X_train = df_train[FEATURE_COLS]
        y_train = df_train['movement']

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_train)

        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X_scaled, y_train)

        os.makedirs("saved_models", exist_ok=True)
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
    else:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

    # === Walk through historical realtime data ===
    for i in range(30, len(df_feat)):
        sub_df_feat = df_feat.iloc[:i + 1]
        sub_df_raw = df.iloc[:i + 1]
        predict_next(sub_df_feat, model, scaler, sub_df_raw)

if __name__ == "__main__":
    run_prediction()
