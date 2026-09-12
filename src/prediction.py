import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))
from database import get_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PredictionEngine")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

class PredictionEngine:
    """
    Inference engine for sales quantity prediction and expected revenue forecasting.
    Logs predictions to MongoDB 'predictions' collection.
    """
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.bundle = None
        self.model = None
        self.fe = None
        self.model_name = "XGBoost Regressor"
        self._load_bundle()

    def _load_bundle(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Trained model bundle not found at {self.model_path}. Please run train_model.py first.")
        self.bundle = joblib.load(self.model_path)
        self.model = self.bundle["model"]
        self.fe = self.bundle["feature_engineer"]
        self.model_name = self.bundle.get("best_model_name", "XGBoost Regressor")
        logger.info(f"Loaded prediction model: {self.model_name}")

    def predict_sales(self, store_id: str, product_id: str, date_str: str,
                      unit_price: float, discount: float = 0.0, promotion: int = 0,
                      holiday: int = 0, log_to_mongo: bool = True) -> Dict[str, Any]:
        """
        Predicts sales quantity and estimated revenue for a single transaction scenario.
        Returns:
            dict containing:
              - predicted_quantity (int)
              - predicted_revenue (float)
              - unit_price (float)
              - effective_price (float)
              - discount_pct (float)
              - confidence_interval (tuple of lower, upper)
        """
        # Construct feature vector
        X_infer = self.fe.build_inference_row(
            store_id=store_id,
            product_id=product_id,
            date_str=date_str,
            unit_price=unit_price,
            discount=discount,
            promotion=promotion,
            holiday=holiday
        )
        
        raw_pred = float(self.model.predict(X_infer)[0])
        pred_qty = max(0, int(round(raw_pred)))
        
        # Calculate revenue: Quantity * (Unit Price * (1 - Discount))
        effective_price = round(unit_price * (1.0 - discount), 2)
        pred_revenue = round(pred_qty * effective_price, 2)
        
        # Estimated 90% confidence interval based on test RMSE (~7.5 units)
        rmse = 7.5
        ci_lower = max(0, int(round(raw_pred - 1.645 * rmse)))
        ci_upper = max(pred_qty, int(round(raw_pred + 1.645 * rmse)))
        
        result = {
            "prediction_date": date_str,
            "product_id": product_id,
            "store_id": store_id,
            "unit_price": unit_price,
            "discount": discount,
            "promotion": promotion,
            "holiday": holiday,
            "predicted_quantity": pred_qty,
            "predicted_revenue": pred_revenue,
            "effective_price": effective_price,
            "confidence_interval": (ci_lower, ci_upper),
            "model_name": self.model_name,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if log_to_mongo:
            try:
                db = get_db()
                db.save_prediction({
                    "prediction_date": date_str,
                    "product_id": product_id,
                    "store_id": store_id,
                    "predicted_quantity": pred_qty,
                    "predicted_revenue": pred_revenue,
                    "model_name": self.model_name,
                    "created_at": result["created_at"]
                })
            except Exception as e:
                logger.warning(f"Failed to log prediction to MongoDB: {e}")
                
        return result

    def forecast_horizon(self, store_id: str, product_id: str, start_date: str,
                         days: int = 30, unit_price: float = 1000.0,
                         default_discount: float = 0.0) -> pd.DataFrame:
        """
        Generates forward multi-day demand and revenue forecasts.
        """
        start = pd.to_datetime(start_date)
        records = []
        
        from data_loader import HOLIDAYS_MD
        
        for i in range(days):
            curr = start + timedelta(days=i)
            d_str = curr.strftime("%Y-%m-%d")
            md = curr.strftime("%m-%d")
            is_hol = 1 if md in HOLIDAYS_MD else 0
            is_promo = 1 if (curr.isocalendar()[1] % 4 == 0 or is_hol == 1) else 0
            disc = 0.15 if is_promo else default_discount
            
            p = self.predict_sales(
                store_id=store_id,
                product_id=product_id,
                date_str=d_str,
                unit_price=unit_price,
                discount=disc,
                promotion=is_promo,
                holiday=is_hol,
                log_to_mongo=False
            )
            records.append(p)
            
        return pd.DataFrame(records)

_engine = None

def get_prediction_engine():
    global _engine
    if _engine is None:
        _engine = PredictionEngine()
    return _engine

if __name__ == "__main__":
    engine = get_prediction_engine()
    sample = engine.predict_sales(
        store_id="ST01",
        product_id="P01",
        date_str="2026-09-15",
        unit_price=45000,
        discount=0.15,
        promotion=1,
        holiday=0
    )
    print("Sample Sales Prediction Result:")
    print(sample)
