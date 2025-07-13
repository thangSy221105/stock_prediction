import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pymongo import MongoClient
from datetime import datetime, timedelta
import logging
import os
import json
from sklearn.preprocessing import MinMaxScaler

# =============== CONFIG ===============
MONGO_URI = "mongodb://192.168.1.148:27017,192.168.1.149:27017/?replicaSet=rs0"
DB_NAME = "stock_db"
REALTIME_PREFIX = "stock_realtime_"
PREDICTED_PREFIX = "predicted_output_10min_"
FUTURE_SHIFT_MINUTES = 10

config = {
    "mongo": {
        "uri": MONGO_URI,
        "db_name": DB_NAME,
        "realtime_collection": "stock_prices",
        "symbols": ["AAPL"],
    },
    "data": {
        "window_size": 5,
        "train_split_size": 0.80,
        "feature_cols": ['open', 'high', 'low', 'close', 'volume'],
    },
    "model": {
        "hidden_size": 64,
        "num_layers": 2,
        "input_size": 5,
        "output_size": 1,
        "dropout": 0.2
    },
    "training": {
        "device": "cpu",
        "batch_size": 32,
        "num_epoch": 300,
        "learning_rate": 0.0005,
        "scheduler_step_size": 20,
    },
    "realtime": {
        "interval": 60,
        "prediction_horizon": FUTURE_SHIFT_MINUTES,
        "use_realtime_data": False
    }
}

# =============== LOGGING ===============
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("predictor.log"),
        logging.StreamHandler()
    ]
)

# =============== CONNECT MONGODB ===============
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    logging.info("✅ Connected to MongoDB Replica Set.")
except Exception as e:
    logging.error(f"❌ Cannot connect to MongoDB: {e}")
    exit(1)

db = client[DB_NAME]


# =============== DATA FUNCTIONS ===============
def calculate_features(df):
    df = df.copy()
    # Sử dụng fillna để tránh mất dữ liệu
    df['sma3'] = df['close'].rolling(window=3, min_periods=1).mean()
    df['price_change'] = df['close'].pct_change().fillna(0)
    df['volatility'] = df['close'].rolling(window=3, min_periods=1).std().fillna(0)
    return df


def fetch_history_data_from_collection(config, symbol):
    col_name = "stock_history_data"
    if col_name not in db.list_collection_names():
        logging.warning(f"Collection {col_name} does not exist")
        return None

    docs = list(db[col_name].find({"symbol": symbol}).sort("timestamp", 1))
    if docs:
        timestamps = [doc["timestamp"] for doc in docs]
        prices = {key: [doc.get(key, 0) for doc in docs] for key in config["data"]["feature_cols"]}
        history_data = {"timestamps": timestamps, **prices}
        logging.info(f"✅ Loaded {len(docs)} history records for {symbol}")
        return history_data
    else:
        logging.warning(f"No history data for {symbol}")
        return None


def prepare_training_data_from_history(history_data, config, future_shift_minutes):
    if not history_data:
        return None, None, None, None, None, None

    df = pd.DataFrame({key: history_data[key] for key in ['timestamps'] + config["data"]["feature_cols"]})
    df['timestamp'] = pd.to_datetime(df['timestamps'])
    df = df.sort_values("timestamp")
    df = calculate_features(df)

    # Chỉ drop những row có NaN trong feature_cols cơ bản
    df = df.dropna(subset=config["data"]["feature_cols"])

    if len(df) < config["data"]["window_size"] + future_shift_minutes:
        logging.warning(f"Insufficient data: {len(df)} < {config['data']['window_size'] + future_shift_minutes}")
        return None, None, None, None, None, None

    data = df[config["data"]["feature_cols"]].values
    target = df['close'].shift(-future_shift_minutes).values
    valid_idx = ~np.isnan(target)
    data, target = data[valid_idx], target[valid_idx]

    scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    target = target_scaler.fit_transform(target.reshape(-1, 1))

    X, y = [], []
    for i in range(len(data) - config["data"]["window_size"] + 1):
        window = data[i:i + config["data"]["window_size"]]
        X.append(window)
        y.append(target[i + config["data"]["window_size"] - 1])

    X, y = np.array(X), np.array(y)
    split_idx = int(len(X) * config["data"]["train_split_size"])
    if split_idx == 0:
        split_idx = 1
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:], scaler, target_scaler


# =============== MODEL ===============
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        out = self.relu(out)
        out = self.fc(out)
        return out


# =============== TRAINING ===============
def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                val_loss += criterion(model(batch_x), batch_y).item()

        scheduler.step()
        logging.info(
            f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {train_loss / len(train_loader):.6f} | Val Loss: {val_loss / len(val_loader):.6f}")


# =============== MAIN ===============
def main():
    models = {}
    scalers = {}
    target_scalers = {}

    # Training part
    for symbol in config["mongo"]["symbols"]:
        history_data = fetch_history_data_from_collection(config, symbol)
        if not history_data:
            logging.warning(f"⚠️ Skipping {symbol} due to no history data.")
            continue

        X_train, y_train, X_val, y_val, scaler, target_scaler = prepare_training_data_from_history(history_data, config,
                                                                                                   FUTURE_SHIFT_MINUTES)
        if X_train is None:
            logging.warning(f"⚠️ Skipping {symbol} due to insufficient data.")
            continue

        train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=config["training"]["batch_size"],
                                  shuffle=True)
        val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=config["training"]["batch_size"],
                                shuffle=False)

        model = GRUModel(**config["model"]).to(config["training"]["device"])
        train_model(model, train_loader, val_loader, config["training"]["num_epoch"], config["training"]["device"])

        models[symbol] = model
        scalers[symbol] = scaler
        target_scalers[symbol] = target_scaler
        os.makedirs("saved_models", exist_ok=True)
        torch.save(model.state_dict(), f"saved_models/model_{symbol}.pth")
        logging.info(
            f"✅ Trained and saved GRU model for {symbol} | Train size: {len(X_train)} | Val size: {len(X_val)}")

    # PREDICTION PART
    for symbol in config["mongo"]["symbols"]:
        if symbol not in models:
            continue

        realtime_col_name = f"{REALTIME_PREFIX}{symbol}"
        if realtime_col_name not in db.list_collection_names():
            logging.warning(f"Collection {realtime_col_name} does not exist")
            continue

        docs = list(db[realtime_col_name].find({}).sort("timestamp", 1))
        print(f"🔍 DEBUG: Collected {len(docs)} realtime records for {symbol}")
        logging.info(f"📊 Found {len(docs)} realtime records for {symbol}")

        if not docs or len(docs) < config["data"]["window_size"]:
            logging.warning(f"⚠️ Not enough realtime data for prediction for {symbol}.")
            continue

        data = {key: [doc.get(key, 0) for doc in docs] for key in config["data"]["feature_cols"]}
        data['timestamps'] = [doc["timestamp"] for doc in docs]
        df_realtime = pd.DataFrame(data)
        df_realtime['timestamp'] = pd.to_datetime(df_realtime['timestamps'])
        df_realtime = df_realtime.sort_values('timestamp')

        print(f"🔍 DEBUG: Before feature calculation: {len(df_realtime)} records")
        logging.info(f"📈 Before feature calculation: {len(df_realtime)} records")
        df_realtime = calculate_features(df_realtime)
        print(f"🔍 DEBUG: After feature calculation: {len(df_realtime)} records")
        logging.info(f"📈 After feature calculation: {len(df_realtime)} records")

        # Chỉ drop những row có NaN trong feature_cols cơ bản
        df_realtime = df_realtime.dropna(subset=config["data"]["feature_cols"])
        print(f"🔍 DEBUG: After dropna: {len(df_realtime)} records")
        logging.info(f"📈 After dropna: {len(df_realtime)} records")

        # Sử dụng filter linh hoạt hơn thay vì df_realtime["close"] < 2000
        if len(df_realtime) > 0:
            close_mean = df_realtime['close'].mean()
            close_std = df_realtime['close'].std()
            upper_bound = close_mean + 3 * close_std
            lower_bound = close_mean - 3 * close_std

            # Chỉ lọc những outlier cực kỳ bất thường
            df_realtime = df_realtime[
                (df_realtime['close'] >= max(lower_bound, 0.01)) &
                (df_realtime['close'] <= upper_bound)
                ]

        print(f"🔍 DEBUG: After outlier filter: {len(df_realtime)} records")
        logging.info(f"📈 After outlier filter: {len(df_realtime)} records")
        if len(df_realtime) > 0:
            print(f"🔍 DEBUG: Close price range: {df_realtime['close'].min():.2f} - {df_realtime['close'].max():.2f}")
            logging.info(f"📈 Close price range: {df_realtime['close'].min():.2f} - {df_realtime['close'].max():.2f}")

        model = models[symbol]
        model.eval()
        scaler = scalers[symbol]
        target_scaler = target_scalers[symbol]

        predictions_count = 0
        total_possible = len(df_realtime) - config["data"]["window_size"] + 1
        print(
            f"🔍 DEBUG: Can make {max(0, total_possible)} predictions (total records: {len(df_realtime)}, window_size: {config['data']['window_size']})")

        # Tăng số lượng predictions
        for idx in range(config["data"]["window_size"], len(df_realtime) + 1):
            window_df = df_realtime.iloc[idx - config["data"]["window_size"]:idx]
            latest_row = window_df.iloc[-1]

            try:
                window_input = scaler.transform(window_df[config["data"]["feature_cols"]].values)
                X_input = torch.FloatTensor(window_input).unsqueeze(0).to(config["training"]["device"])

                with torch.no_grad():
                    prediction = model(X_input)
                    predicted_scaled = prediction.cpu().numpy()

                predicted_close = float(target_scaler.inverse_transform(predicted_scaled)[0][0])
                old_close = float(latest_row["close"])
                change = ((predicted_close - old_close) / old_close) * 100 if old_close != 0 else 0.0
                change = max(min(change, 1.5), -1.5)
                predicted_close = old_close * (1 + change / 100)

                prediction_output = {
                    "symbol": symbol,
                    "open": float(latest_row["open"]),
                    "high": float(latest_row["high"]),
                    "low": float(latest_row["low"]),
                    "close": round(predicted_close, 2),
                    "volume": float(latest_row["volume"]),
                    "timestamp": (latest_row['timestamp'] + timedelta(minutes=FUTURE_SHIFT_MINUTES)).strftime(
                        "%Y-%m-%d %H:%M:%S"),
                    "old_close": old_close,
                    "change": round(change, 2)
                }

                print(json.dumps(prediction_output, indent=2))
                predictions_count += 1

                try:
                    db[f"{PREDICTED_PREFIX}{symbol}"].insert_one(prediction_output)
                except Exception as e:
                    logging.error(f"❌ Error inserting prediction: {e}")

            except Exception as e:
                logging.error(f"❌ Error making prediction for index {idx}: {e}")
                continue

        print(f"🔍 DEBUG: Generated {predictions_count} predictions for {symbol}")
        logging.info(f"✅ Generated {predictions_count} predictions for {symbol}")


if __name__ == "__main__":
    main()