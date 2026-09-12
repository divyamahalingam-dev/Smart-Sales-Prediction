import os
import sys
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))
from database import get_db
from prediction import get_prediction_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SmartInventory")

class InventoryOptimizer:
    """
    Computes dynamic safety stock, recommended inventory levels, reorder quantities,
    and stockout risk classifications based on ML demand forecasting.
    """
    def __init__(self, service_level_z: float = 1.65, lead_time_days: int = 7):
        # 1.65 corresponds to 95% cycle service level
        self.z = service_level_z
        self.lead_time = lead_time_days

    def calculate_recommendation(self, product_id: str, current_stock: int,
                                 predicted_demand: int, demand_std: float = 10.0,
                                 reorder_level: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculates recommended inventory and reorder quantity for a single product.
        Formula:
            Safety Stock = Z * std_dev * sqrt(lead_time)
            Recommended Stock = Predicted Demand + Safety Stock
            Reorder Quantity = max(0, Recommended Stock - Current Stock)
        """
        safety_stock = int(np.ceil(self.z * demand_std * np.sqrt(self.lead_time / 7.0)))
        recommended_stock = int(predicted_demand + safety_stock)
        reorder_qty = max(0, recommended_stock - current_stock)
        
        # Determine Status
        if current_stock <= safety_stock or (current_stock < 0.20 * predicted_demand and predicted_demand > 0):
            status = "Stockout Risk"
            risk_level = "High"
            badge_color = "red"
        elif current_stock < recommended_stock * 0.70 or (reorder_level and current_stock <= reorder_level):
            status = "Consider Reordering"
            risk_level = "Medium"
            badge_color = "orange"
        else:
            status = "Stock Sufficient"
            risk_level = "Low"
            badge_color = "green"
            
        return {
            "product_id": product_id,
            "current_stock": int(current_stock),
            "predicted_demand": int(predicted_demand),
            "safety_stock": int(safety_stock),
            "recommended_stock": int(recommended_stock),
            "reorder_quantity": int(reorder_qty),
            "status": status,
            "risk_level": risk_level,
            "badge_color": badge_color
        }

    def generate_system_inventory_report(self, horizon_days: int = 14, default_store: str = "ST01") -> pd.DataFrame:
        """
        Forecasts demand across all products in catalog and updates MongoDB 'inventory'.
        """
        db = get_db()
        products = db.get_products()
        engine = get_prediction_engine()
        
        inventory_items = []
        today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
        
        for p in products:
            p_id = p["product_id"]
            curr_stock = p.get("current_stock", 50)
            reorder_lvl = p.get("reorder_level", 30)
            unit_price = p.get("unit_price", 1000)
            
            # Forecast demand over the horizon
            f_df = engine.forecast_horizon(
                store_id=default_store,
                product_id=p_id,
                start_date=today_str,
                days=horizon_days,
                unit_price=unit_price
            )
            total_predicted_demand = int(f_df["predicted_quantity"].sum())
            demand_std = float(f_df["predicted_quantity"].std()) if len(f_df) > 1 else 5.0
            if np.isnan(demand_std) or demand_std < 1.0:
                demand_std = 3.0
                
            rec = self.calculate_recommendation(
                product_id=p_id,
                current_stock=curr_stock,
                predicted_demand=total_predicted_demand,
                demand_std=demand_std,
                reorder_level=reorder_lvl
            )
            
            rec["product_name"] = p.get("product_name", p_id)
            rec["category"] = p.get("category", "General")
            rec["brand"] = p.get("brand", "")
            rec["unit_price"] = unit_price
            rec["estimated_reorder_cost"] = round(rec["reorder_quantity"] * unit_price, 2)
            
            inventory_items.append(rec)
            
        # Update MongoDB
        db.upsert_inventory(inventory_items)
        logger.info(f"Updated {len(inventory_items)} inventory records in MongoDB.")
        
        df_inv = pd.DataFrame(inventory_items)
        return df_inv

_optimizer = None

def get_inventory_optimizer(service_level_z: float = 1.65, lead_time_days: int = 7):
    global _optimizer
    if _optimizer is None:
        _optimizer = InventoryOptimizer(service_level_z=service_level_z, lead_time_days=lead_time_days)
    return _optimizer

if __name__ == "__main__":
    opt = get_inventory_optimizer()
    report = opt.generate_system_inventory_report()
    print("Generated Inventory Optimization Report:")
    print(report[["product_id", "product_name", "current_stock", "predicted_demand", "recommended_stock", "reorder_quantity", "status"]].head(10))
