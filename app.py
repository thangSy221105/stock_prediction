import os
import time as t
import logging
from datetime import datetime, timedelta
import pytz

import pandas as pd
import yfinance as yf
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, PyMongoError

# ==== CONFIGURATION ====
MONGO_URI = "mongodb://192.168.1.148:27017/"  # VM3
DB_NAME = "stock_db"
COLLECTION_NAME = "stock_prices"
STOCK_SYMBOLS = ["AAPL", "GOOGL", "MSFT"]
REALTIME_INTERVAL = 30  # seconds per fetch

# ==== LOGGING ====
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# ==== CONNECT TO MONGODB ====
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    logging.info("✅ MongoDB connection established.")
except ServerSelectionTimeoutError as e:
    logging.error(f"❌ MongoDB connection failed: {e}")
    exit(1)

db = client[DB_NAME]
collection = db[COLLECTION_NAME]


# ==== CHECK MARKET HOURS ====
def is_market_open():
    now = datetime.now(pytz.timezone('US/Eastern'))
    market_open = datetime.strptime("09:30", "%H:%M").time()
    market_close = datetime.strptime("16:00", "%H:%M").time()
    return now.weekday() < 5 and market_open <= now.time() <= market_close


# ==== FETCH HISTORICAL DATA (MINUTE) ====
def fetch_historical_data(symbols, total_days=30, segment_days=7):
    historical_data = []
    end_date = datetime.now()
    current_start = end_date - timedelta(days=total_days)

    while current_start < end_date:
        current_end = current_start + timedelta(days=segment_days)
        if current_end > end_date:
            current_end = end_date
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                data = stock.history(start=current_start.strftime('%Y-%m-%d'),
                                     end=current_end.strftime('%Y-%m-%d'),
                                     interval="1m")
                if data.empty:
                    logging.warning(f"No historical data for {symbol} from {current_start} to {current_end}")
                    continue
                data = data.reset_index()
                data["symbol"] = symbol
                historical_data.append(data)
            except Exception as e:
                logging.error(f"Error fetching historical for {symbol}: {e}")
        current_start = current_end

    if historical_data:
        df = pd.concat(historical_data, ignore_index=True).drop_duplicates()
        os.makedirs("data_collection", exist_ok=True)
        file_name = f"data_collection/historical_stock_data_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(file_name, index=False)
        logging.info(f"✅ Saved historical data to {file_name}")
        return df
    else:
        return pd.DataFrame()


# ==== LOAD HISTORICAL TO MONGODB ====
def load_historical_to_mongo(df):
    docs = []
    for _, row in df.iterrows():
        doc = {
            "timestamp": row["Datetime"].strftime("%Y-%m-%d %H:%M:%S") if isinstance(row["Datetime"],
                                                                                     pd.Timestamp) else str(
                row["Datetime"]),
            "symbol": row["symbol"],
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"]),
            "source": "historical"
        }
        docs.append(doc)
    if docs:
        try:
            result = collection.insert_many(docs, ordered=False)
            logging.info(f"✅ Inserted {len(result.inserted_ids)} historical documents into MongoDB.")
        except PyMongoError as e:
            logging.error(f"Error inserting historical data: {e}")


# ==== REALTIME FETCH AND INSERT ====
def fetch_and_insert_realtime(symbols):
    logging.info(f"🚀 Starting realtime collection for symbols: {symbols}")
    try:
        while True:
            if not is_market_open():
                logging.warning("Market is closed, waiting for next check.")
                t.sleep(3600)
                continue

            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    daily_data = ticker.history(period="1d", interval="1d")
                    minute_data = ticker.history(period="1d", interval="1m")

                    if daily_data.empty or minute_data.empty:
                        logging.warning(f"No data for {symbol} (daily or minute)")
                        continue

                    daily_row = daily_data.tail(1).reset_index().iloc[0]
                    minute_row = minute_data.tail(1).reset_index().iloc[0]

                    timestamp = minute_row["Datetime"]
                    open_price = float(minute_row["Open"])
                    high_price = float(daily_row["High"])
                    low_price = float(daily_row["Low"])
                    close_price = float(minute_row["Close"])
                    volume = int(minute_row["Volume"])

                    if open_price == high_price == low_price == close_price:
                        logging.warning(f"⚠️ Identical OHLC values for {symbol}: {open_price}")
                        continue

                    doc = {
                        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S") if isinstance(timestamp,
                                                                                           pd.Timestamp) else str(
                            timestamp),
                        "symbol": symbol,
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "close": close_price,
                        "volume": volume,
                        "source": "realtime"
                    }

                    collection.insert_one(doc)
                    logging.info(f"✅ Inserted realtime: {doc}")

                except Exception as e:
                    logging.error(f"⚠️ Error fetching realtime for {symbol}: {e}")

            t.sleep(REALTIME_INTERVAL)

    except KeyboardInterrupt:
        logging.info("🛑 Stopped realtime collection with Ctrl+C.")


# ==== MAIN ====
def main():
    df_hist = fetch_historical_data(STOCK_SYMBOLS, total_days=30, segment_days=7)
    if not df_hist.empty:
        load_historical_to_mongo(df_hist)
    else:
        logging.warning("⚠️ No historical data fetched. Skipping historical insertion.")

    fetch_and_insert_realtime(STOCK_SYMBOLS)


if __name__ == "__main__":
    main()