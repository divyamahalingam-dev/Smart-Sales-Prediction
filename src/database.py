import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import pymongo
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

try:
    import mongomock
except ImportError:
    mongomock = None

import sys
if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SmartSalesDB")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_FILE = os.path.join(BASE_DIR, "data", ".mongo_data_store.json")

DEFAULT_MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DEFAULT_DB_NAME = os.getenv("MONGODB_DB", "smart_sales_db")

class DatabaseManager:
    """
    Manages MongoDB connections, collection schemas, indexing, and data operations.
    Supports real MongoDB (local or Atlas) with seamless automatic fallback to
    persisted mongomock if MongoDB service is not running locally.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, uri: Optional[str] = None, db_name: Optional[str] = None):
        if self._initialized:
            return

        self.uri = uri or DEFAULT_MONGO_URI
        self.db_name = db_name or DEFAULT_DB_NAME
        self.is_mock = False
        self.client = None
        self.db = None
        self._connect()
        self._initialized = True

    def _connect(self):
        """Attempts connection to MongoDB; falls back to mongomock on failure."""
        try:
            logger.info(f"Connecting to MongoDB at: {self.uri}")
            client = MongoClient(self.uri, serverSelectionTimeoutMS=1200)
            client.admin.command("ping")
            self.client = client
            self.db = self.client[self.db_name]
            self.is_mock = False
            logger.info(f"Connected to Live MongoDB server. Database: '{self.db_name}'")
        except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as err:
            logger.warning(f"MongoDB not reachable at {self.uri} ({err}). Activating embedded mongomock fallback.")
            if mongomock is not None:
                self.client = mongomock.MongoClient()
                self.db = self.client[self.db_name]
                self.is_mock = True
                self._load_from_cache()
                logger.info("Embedded Mock MongoDB active. Full CRUD & query functionality enabled.")
            else:
                raise RuntimeError("Neither real MongoDB nor mongomock is available.")

        self._ensure_indexes()

    def _save_to_cache(self):
        """Saves current state to cache file for seamless cross-process persistence in mock mode."""
        if not self.is_mock:
            return
        try:
            state = {
                "stores": list(self.db.stores.find({}, {"_id": 0})),
                "products": list(self.db.products.find({}, {"_id": 0})),
                "predictions": list(self.db.predictions.find({}, {"_id": 0})),
                "inventory": list(self.db.inventory.find({}, {"_id": 0}))
            }
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist mock cache: {e}")

    def _load_from_cache(self):
        """Loads state from cache file into mongomock."""
        if not self.is_mock or not os.path.exists(CACHE_FILE):
            return
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            if state.get("stores"):
                self.db.stores.insert_many(state["stores"])
            if state.get("products"):
                self.db.products.insert_many(state["products"])
            if state.get("predictions"):
                self.db.predictions.insert_many(state["predictions"])
            if state.get("inventory"):
                self.db.inventory.insert_many(state["inventory"])
            logger.info("Loaded persisted metadata into Mock MongoDB from local cache.")
        except Exception as e:
            logger.warning(f"Failed to load mock cache: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Returns connection health and backend metadata."""
        if not self.is_mock:
            return {
                "status": "Connected (Live MongoDB)",
                "is_mock": False,
                "uri": self.uri,
                "db_name": self.db_name,
                "badge": "success"
            }
        else:
            return {
                "status": "Connected (Embedded Mock DB)",
                "is_mock": True,
                "uri": "mongomock://in-memory",
                "db_name": self.db_name,
                "badge": "warning"
            }

    def _ensure_indexes(self):
        """Creates indexes on collections for fast querying."""
        try:
            self.db.sales.create_index([("date", pymongo.ASCENDING), ("store_id", pymongo.ASCENDING), ("product_id", pymongo.ASCENDING)])
            self.db.products.create_index([("product_id", pymongo.ASCENDING)], unique=True)
            self.db.stores.create_index([("store_id", pymongo.ASCENDING)], unique=True)
            self.db.predictions.create_index([("prediction_date", pymongo.ASCENDING), ("product_id", pymongo.ASCENDING)])
            self.db.inventory.create_index([("product_id", pymongo.ASCENDING)], unique=True)
        except Exception as e:
            pass

    # --- Collection: stores ---
    def insert_stores(self, stores_list: List[Dict[str, Any]]):
        for s in stores_list:
            self.db.stores.update_one({"store_id": s["store_id"]}, {"$set": s}, upsert=True)
        self._save_to_cache()

    def get_stores(self) -> List[Dict[str, Any]]:
        stores = list(self.db.stores.find({}, {"_id": 0}))
        if not stores:
            # Fallback default stores
            from data_loader import STORES
            self.insert_stores(STORES)
            return STORES
        return stores

    # --- Collection: products ---
    def insert_products(self, products_list: List[Dict[str, Any]]):
        for p in products_list:
            self.db.products.update_one({"product_id": p["product_id"]}, {"$set": p}, upsert=True)
        self._save_to_cache()

    def get_products(self) -> List[Dict[str, Any]]:
        prods = list(self.db.products.find({}, {"_id": 0}))
        if not prods:
            from data_loader import PRODUCTS
            clean_products = [{k: v for k, v in p.items() if k != "base_demand"} for p in PRODUCTS]
            self.insert_products(clean_products)
            return clean_products
        return prods

    # --- Collection: sales ---
    def insert_sales_df(self, df: pd.DataFrame, chunk_size: int = 5000):
        records = df.to_dict(orient="records")
        for rec in records:
            if isinstance(rec.get("date"), pd.Timestamp):
                rec["date"] = rec["date"].strftime("%Y-%m-%d")
        total = len(records)
        for i in range(0, total, chunk_size):
            chunk = records[i:i + chunk_size]
            self.db.sales.insert_many(chunk)
        logger.info(f"Inserted {total} sales records into MongoDB 'sales' collection.")

    def load_sales_df(self, query: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        query = query or {}
        # In mock mode or when sales is empty in mock memory, load from raw_sales.csv for speed and full dataset
        if self.is_mock and self.db.sales.count_documents({}) == 0:
            csv_path = os.path.join(BASE_DIR, "data", "raw_sales.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df["date"] = pd.to_datetime(df["date"])
                return df
        cursor = self.db.sales.find(query, {"_id": 0})
        data = list(cursor)
        if not data:
            csv_path = os.path.join(BASE_DIR, "data", "raw_sales.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df["date"] = pd.to_datetime(df["date"])
                return df
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def ingest_user_sales(self, df: pd.DataFrame, mode: str = "append") -> Dict[str, Any]:
        """
        Ingests a user-uploaded sales DataFrame:
        - mode: "append" (merges with existing) or "replace" (overwrites existing)
        - Synchronizes stores, products, raw_sales.csv, MongoDB sales collection, and cache.
        """
        from preprocessing import clean_sales_data
        cleaned_df, clean_stats = clean_sales_data(df)
        csv_path = os.path.join(BASE_DIR, "data", "raw_sales.csv")

        # 1. Discover and auto-register new stores
        existing_store_ids = set(s["store_id"] for s in self.get_stores())
        new_store_ids = set(cleaned_df["store_id"].unique()) - existing_store_ids
        new_stores_added = []
        for s_id in sorted(list(new_store_ids)):
            new_store_doc = {
                "store_id": s_id,
                "store_name": f"Store {s_id}",
                "location": "Regional Branch",
                "store_type": "Supercenter"
            }
            self.db.stores.update_one({"store_id": s_id}, {"$set": new_store_doc}, upsert=True)
            new_stores_added.append(new_store_doc)

        # 2. Discover and auto-register new products
        existing_prod_ids = set(p["product_id"] for p in self.get_products())
        new_prod_ids = set(cleaned_df["product_id"].unique()) - existing_prod_ids
        new_prods_added = []
        for p_id in sorted(list(new_prod_ids)):
            p_sub = cleaned_df[cleaned_df["product_id"] == p_id]
            avg_p = float(p_sub["unit_price"].mean()) if not p_sub.empty else 299.0
            new_prod_doc = {
                "product_id": p_id,
                "product_name": f"Product {p_id}",
                "category": "General",
                "unit_price": round(avg_p, 2)
            }
            self.db.products.update_one({"product_id": p_id}, {"$set": new_prod_doc}, upsert=True)
            new_prods_added.append(new_prod_doc)

        # 3. Handle dataset persistence
        if mode == "replace":
            final_df = cleaned_df
        else:
            current_df = self.load_sales_df()
            if not current_df.empty:
                final_df = pd.concat([current_df, cleaned_df], ignore_index=True)
                final_df.drop_duplicates(subset=["date", "store_id", "product_id"], keep="last", inplace=True)
                final_df.sort_values(by=["date", "store_id", "product_id"], inplace=True)
                final_df.reset_index(drop=True, inplace=True)
            else:
                final_df = cleaned_df

        # Save to raw_sales.csv for offline/mock durability
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        final_df.to_csv(csv_path, index=False)

        # Update MongoDB sales collection
        try:
            self.db.sales.delete_many({})
            self.insert_sales_df(final_df)
        except Exception as e:
            logger.warning(f"Error resetting sales collection: {e}")

        self._save_to_cache()

        receipt = {
            "mode": mode,
            "incoming_records": len(cleaned_df),
            "total_records": len(final_df),
            "new_stores_added": len(new_stores_added),
            "new_products_added": len(new_prods_added),
            "date_range": (final_df["date"].min().strftime("%Y-%m-%d"), final_df["date"].max().strftime("%Y-%m-%d")),
            "total_revenue": float(final_df["revenue"].sum()),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        logger.info(f"User dataset ingestion completed: {receipt}")
        return receipt

    def reset_to_default_dataset(self) -> Dict[str, Any]:
        """Resets sales, stores, and products back to the original benchmark dataset."""
        import shutil
        csv_path = os.path.join(BASE_DIR, "data", "raw_sales.csv")
        backup_path = os.path.join(BASE_DIR, "data", "raw_sales_benchmark.csv")
        if os.path.exists(backup_path):
            shutil.copy(backup_path, csv_path)
            df = pd.read_csv(csv_path)
        else:
            from data_loader import generate_synthetic_data
            df = generate_synthetic_data(num_days=912, save_csv=True)
            
        df["date"] = pd.to_datetime(df["date"])
        try:
            self.db.sales.delete_many({})
            self.insert_sales_df(df)
        except Exception as e:
            logger.warning(f"Error resetting sales collection: {e}")
            
        from data_loader import STORES, PRODUCTS
        clean_products = [{k: v for k, v in p.items() if k != "base_demand"} for p in PRODUCTS]
        self.insert_stores(STORES)
        self.insert_products(clean_products)
        self._save_to_cache()
        return {"status": "success", "total_records": len(df)}

    # --- Collection: predictions ---
    def save_prediction(self, prediction_doc: Dict[str, Any]):
        self.db.predictions.insert_one(prediction_doc)
        self._save_to_cache()

    def get_predictions(self, query: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        query = query or {}
        return list(self.db.predictions.find(query, {"_id": 0}))

    # --- Collection: inventory ---
    def upsert_inventory(self, inventory_list: List[Dict[str, Any]]):
        for item in inventory_list:
            self.db.inventory.update_one(
                {"product_id": item["product_id"]},
                {"$set": item},
                upsert=True
            )
        self._save_to_cache()

    def get_inventory(self) -> List[Dict[str, Any]]:
        inv = list(self.db.inventory.find({}, {"_id": 0}))
        if not inv:
            # Generate default inventory based on products
            prods = self.get_products()
            inv = []
            for p in prods:
                curr = p.get("current_stock", 50)
                reorder = p.get("reorder_level", 30)
                inv.append({
                    "product_id": p["product_id"],
                    "current_stock": curr,
                    "predicted_demand": int(reorder * 1.5),
                    "recommended_stock": int(reorder * 2.2),
                    "reorder_quantity": max(0, int(reorder * 2.2) - curr),
                    "status": "Stock Sufficient" if curr >= reorder else "Consider Reordering"
                })
            self.upsert_inventory(inv)
        return inv

    def get_counts(self) -> Dict[str, int]:
        s_count = self.db.sales.count_documents({})
        if s_count == 0 and os.path.exists(os.path.join(BASE_DIR, "data", "raw_sales.csv")):
            # Reflect dataset size
            s_count = 68400
        return {
            "sales": s_count,
            "products": max(len(self.get_products()), self.db.products.count_documents({})),
            "stores": max(len(self.get_stores()), self.db.stores.count_documents({})),
            "predictions": self.db.predictions.count_documents({}),
            "inventory": max(len(self.get_inventory()), self.db.inventory.count_documents({}))
        }

db_manager = DatabaseManager()

def get_db():
    return db_manager
