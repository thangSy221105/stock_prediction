from pymongo import MongoClient
from bson.json_util import dumps

# ==== CẤU HÌNH ====
MONGO_URI = "mongodb://192.168.1.148:27017/?replicaSet=rs0&directConnection=false&serverSelectionTimeoutMS=5000"
DB_NAME = "stock_db"
SRC_COLLECTION_NAME = "stock_prices"           # VM3 đang lưu dữ liệu tại đây
HISTORY_COLLECTION_NAME = "stock_history_data"
REALTIME_COLLECTION_BASE = "stock_realtime_"   # Dạng prefix

# ==== KẾT NỐI MONGODB ====
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
src_collection = db[SRC_COLLECTION_NAME]
history_collection = db[HISTORY_COLLECTION_NAME]

# ==== BẮT ĐẦU THEO DÕI CHANGE STREAM ====
pipeline = [{'$match': {}}]
change_stream = src_collection.watch(pipeline, full_document='updateLookup')

print("📡 Đang theo dõi Change Stream từ MongoDB VM3...")

for change in change_stream:
    full_doc = change.get("fullDocument")
    if not full_doc:
        continue

    source = full_doc.get("source", "unknown")
    symbol = full_doc.get("symbol", "").upper().strip()
    timestamp = full_doc.get("timestamp", "")

    record = {
        "symbol": symbol,
        "timestamp": timestamp,
        "open": full_doc.get("open", None),
        "high": full_doc.get("high", None),
        "low": full_doc.get("low", None),
        "close": full_doc.get("close", None),
        "volume": full_doc.get("volume", None),
    }

    # Ghi rõ thông tin
    print(f"📥 Nhận dữ liệu: {dumps(record, indent=2)}")

    if source == "historical":
        history_collection.insert_one(record)
        print("✅ Đã ghi vào collection: stock_history_data")
    elif source == "realtime":
        if symbol:
            # Tạo tên collection riêng cho mỗi symbol
            realtime_collection_name = REALTIME_COLLECTION_BASE + symbol
            realtime_collection = db[realtime_collection_name]
            realtime_collection.insert_one(record)
            print(f"✅ Đã ghi vào collection: {realtime_collection_name}")
        else:
            print("⚠️  Dữ liệu realtime không có symbol, bỏ qua.")
    else:
        print(f"⚠️  Unknown source: {source} -> Bỏ qua hoặc xử lý riêng nếu cần.")
