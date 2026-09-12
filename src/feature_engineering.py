import os
import json
import logging
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FeatureEngineering")

class FeatureEngineer:
    """
    Constructs calendar, temporal lag, rolling window, and interaction features
    for demand prediction and sales forecasting.
    """
    def __init__(self):
        self.store_encoder = LabelEncoder()
        self.product_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        self.is_fitted = False
        self.feature_columns = []
        self.category_map = {}
        self.recent_stats = {} # Cached historical lags/rolling stats for inference

    def _extract_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates temporal calendar features."""
        df = df.copy()
        dt = df["date"].dt
        df["year"] = dt.year
        df["month"] = dt.month
        df["week"] = dt.isocalendar().week.astype(int)
        df["day"] = dt.day
        df["day_of_week"] = dt.dayofweek # 0=Mon, 6=Sun
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["quarter"] = dt.quarter
        
        # Cyclical month and day-of-week encoding
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
        return df

    def _compute_lag_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes lag and rolling window features grouped by store and product."""
        df = df.sort_values(by=["store_id", "product_id", "date"]).copy()
        
        grouped = df.groupby(["store_id", "product_id"])["quantity"]
        
        # Lag features: 1, 7, 14, 30 days
        df["lag_1"] = grouped.shift(1)
        df["lag_7"] = grouped.shift(7)
        df["lag_14"] = grouped.shift(14)
        df["lag_30"] = grouped.shift(30)
        
        # Rolling features (shift by 1 to prevent data leakage)
        df["rolling_mean_7"] = grouped.shift(1).rolling(window=7, min_periods=1).mean()
        df["rolling_mean_14"] = grouped.shift(1).rolling(window=14, min_periods=1).mean()
        df["rolling_mean_30"] = grouped.shift(1).rolling(window=30, min_periods=1).mean()
        df["rolling_std_7"] = grouped.shift(1).rolling(window=7, min_periods=1).std().fillna(0)
        
        # Economic interaction features
        df["effective_price"] = (df["unit_price"] * (1.0 - df["discount"])).round(2)
        df["discount_amount"] = (df["unit_price"] * df["discount"]).round(2)
        df["promo_holiday_inter"] = (df["promotion"] * df["holiday"]).astype(int)
        
        # Backfill initial lag rows with group means so records aren't dropped
        for col in ["lag_1", "lag_7", "lag_14", "lag_30", "rolling_mean_7", "rolling_mean_14", "rolling_mean_30"]:
            df[col] = df.groupby(["product_id"])[col].transform(lambda x: x.bfill().ffill())
            df[col] = df[col].fillna(df["quantity"].median())
            
        return df

    def fit_transform(self, df: pd.DataFrame, products_meta: Optional[List[Dict[str, Any]]] = None) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Fits encoders and builds feature matrix for training."""
        logger.info("Building feature matrix (calendar, lags, rolling averages)...")
        
        # Merge product category if available
        if products_meta:
            cat_df = pd.DataFrame(products_meta)[["product_id", "category"]].drop_duplicates()
            self.category_map = dict(zip(cat_df["product_id"], cat_df["category"]))
        elif "category" not in df.columns:
            # Default categories
            self.category_map = {
                "P01": "Electronics", "P02": "Electronics", "P03": "Electronics", "P04": "Electronics",
                "P05": "Groceries", "P06": "Groceries", "P07": "Groceries", "P08": "Groceries",
                "P09": "Home & Kitchen", "P10": "Home & Kitchen", "P11": "Home & Kitchen",
                "P12": "Fashion", "P13": "Fashion", "P14": "Fashion", "P15": "Fashion"
            }
        
        df = df.copy()
        if "category" not in df.columns:
            df["category"] = df["product_id"].map(self.category_map).fillna("General")
            
        df = self._extract_calendar_features(df)
        df = self._compute_lag_rolling_features(df)
        
        # Fit label encoders
        df["store_encoded"] = self.store_encoder.fit_transform(df["store_id"])
        df["product_encoded"] = self.product_encoder.fit_transform(df["product_id"])
        df["category_encoded"] = self.category_encoder.fit_transform(df["category"])
        
        # Cache recent statistics per (store_id, product_id) for real-time inference
        latest_records = df.sort_values(by="date").groupby(["store_id", "product_id"]).last().reset_index()
        for _, row in latest_records.iterrows():
            key = f"{row['store_id']}_{row['product_id']}"
            self.recent_stats[key] = {
                "lag_1": float(row["quantity"]),
                "lag_7": float(row.get("lag_7", row["quantity"])),
                "lag_14": float(row.get("lag_14", row["quantity"])),
                "lag_30": float(row.get("lag_30", row["quantity"])),
                "rolling_mean_7": float(row.get("rolling_mean_7", row["quantity"])),
                "rolling_mean_14": float(row.get("rolling_mean_14", row["quantity"])),
                "rolling_mean_30": float(row.get("rolling_mean_30", row["quantity"])),
                "rolling_std_7": float(row.get("rolling_std_7", 2.0)),
                "unit_price": float(row["unit_price"]),
                "category": str(row["category"])
            }
            
        self.feature_columns = [
            "store_encoded", "product_encoded", "category_encoded",
            "unit_price", "discount", "promotion", "holiday",
            "year", "month", "week", "day", "day_of_week", "is_weekend", "quarter",
            "month_sin", "month_cos", "dow_sin", "dow_cos",
            "lag_1", "lag_7", "lag_14", "lag_30",
            "rolling_mean_7", "rolling_mean_14", "rolling_mean_30", "rolling_std_7",
            "effective_price", "discount_amount", "promo_holiday_inter"
        ]
        
        self.is_fitted = True
        logger.info(f"Engineered {len(self.feature_columns)} features across {len(df):,} records.")
        return df[self.feature_columns], df["quantity"], self.feature_columns

    def build_inference_row(self, store_id: str, product_id: str, date_str: str,
                            unit_price: float, discount: float, promotion: int, holiday: int) -> pd.DataFrame:
        """
        Constructs a feature vector for real-time single-point sales/demand prediction.
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted or loaded before building inference rows.")
            
        dt = pd.to_datetime(date_str)
        key = f"{store_id}_{product_id}"
        stats = self.recent_stats.get(key, {})
        category = stats.get("category", self.category_map.get(product_id, "Electronics"))
        
        # Encoders safe transform
        try:
            store_enc = int(self.store_encoder.transform([store_id])[0])
        except Exception:
            store_enc = 0
            
        try:
            prod_enc = int(self.product_encoder.transform([product_id])[0])
        except Exception:
            prod_enc = 0
            
        try:
            cat_enc = int(self.category_encoder.transform([category])[0])
        except Exception:
            cat_enc = 0
            
        dow = dt.dayofweek
        month = dt.month
        effective_price = round(unit_price * (1.0 - discount), 2)
        discount_amount = round(unit_price * discount, 2)
        
        row_dict = {
            "store_encoded": store_enc,
            "product_encoded": prod_enc,
            "category_encoded": cat_enc,
            "unit_price": float(unit_price),
            "discount": float(discount),
            "promotion": int(promotion),
            "holiday": int(holiday),
            "year": dt.year,
            "month": month,
            "week": int(dt.isocalendar().week),
            "day": dt.day,
            "day_of_week": dow,
            "is_weekend": 1 if dow in [5, 6] else 0,
            "quarter": dt.quarter,
            "month_sin": np.sin(2 * np.pi * month / 12.0),
            "month_cos": np.cos(2 * np.pi * month / 12.0),
            "dow_sin": np.sin(2 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0),
            "lag_1": float(stats.get("lag_1", 20.0)),
            "lag_7": float(stats.get("lag_7", 20.0)),
            "lag_14": float(stats.get("lag_14", 20.0)),
            "lag_30": float(stats.get("lag_30", 20.0)),
            "rolling_mean_7": float(stats.get("rolling_mean_7", 20.0)),
            "rolling_mean_14": float(stats.get("rolling_mean_14", 20.0)),
            "rolling_mean_30": float(stats.get("rolling_mean_30", 20.0)),
            "rolling_std_7": float(stats.get("rolling_std_7", 3.0)),
            "effective_price": effective_price,
            "discount_amount": discount_amount,
            "promo_holiday_inter": int(promotion * holiday)
        }
        
        return pd.DataFrame([row_dict])[self.feature_columns]
