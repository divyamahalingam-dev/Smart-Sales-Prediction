import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))
from database import get_db

STORES = [
    {"store_id": "ST01", "store_name": "Metro MegaStore", "location": "Mumbai", "store_type": "Supercenter"},
    {"store_id": "ST02", "store_name": "Capital Retail Hub", "location": "Delhi", "store_type": "Hypermarket"},
    {"store_id": "ST03", "store_name": "Silicon Express", "location": "Bangalore", "store_type": "City Express"},
    {"store_id": "ST04", "store_name": "Coastal Mart", "location": "Chennai", "store_type": "Supercenter"},
    {"store_id": "ST05", "store_name": "Heritage Mall Outlet", "location": "Kolkata", "store_type": "Mall Outlet"}
]

PRODUCTS = [
    # Electronics
    {"product_id": "P01", "product_name": 'UltraHD 4K Smart TV 55"', "category": "Electronics", "brand": "Samsung", "unit_price": 45000, "current_stock": 45, "reorder_level": 30, "base_demand": 5},
    {"product_id": "P02", "product_name": "Noise-Cancelling Headphones", "category": "Electronics", "brand": "Sony", "unit_price": 14999, "current_stock": 70, "reorder_level": 50, "base_demand": 12},
    {"product_id": "P03", "product_name": "Bluetooth Smart Speaker", "category": "Electronics", "brand": "JBL", "unit_price": 4499, "current_stock": 110, "reorder_level": 60, "base_demand": 25},
    {"product_id": "P04", "product_name": "Ergonomic Wireless Mouse", "category": "Electronics", "brand": "Logitech", "unit_price": 1999, "current_stock": 140, "reorder_level": 80, "base_demand": 35},
    
    # Groceries
    {"product_id": "P05", "product_name": "Organic Basmati Rice 5kg", "category": "Groceries", "brand": "India Gate", "unit_price": 650, "current_stock": 280, "reorder_level": 200, "base_demand": 60},
    {"product_id": "P06", "product_name": "Cold-Pressed Olive Oil 1L", "category": "Groceries", "brand": "Borges", "unit_price": 850, "current_stock": 160, "reorder_level": 100, "base_demand": 30},
    {"product_id": "P07", "product_name": "Whole Wheat Atta 10kg", "category": "Groceries", "brand": "Aashirvaad", "unit_price": 420, "current_stock": 350, "reorder_level": 250, "base_demand": 85},
    {"product_id": "P08", "product_name": "Premium California Almonds 500g", "category": "Groceries", "brand": "Nutraj", "unit_price": 499, "current_stock": 90, "reorder_level": 80, "base_demand": 40},
    
    # Home & Kitchen
    {"product_id": "P09", "product_name": "Stainless Steel Cookware 3Pc", "category": "Home & Kitchen", "brand": "Prestige", "unit_price": 3299, "current_stock": 50, "reorder_level": 40, "base_demand": 10},
    {"product_id": "P10", "product_name": "Digital Air Fryer 4.5L", "category": "Home & Kitchen", "brand": "Philips", "unit_price": 6999, "current_stock": 35, "reorder_level": 25, "base_demand": 8},
    {"product_id": "P11", "product_name": "Rapid Electric Kettle 1.8L", "category": "Home & Kitchen", "brand": "Pigeon", "unit_price": 999, "current_stock": 120, "reorder_level": 70, "base_demand": 22},
    
    # Fashion & Apparel
    {"product_id": "P12", "product_name": "Men Casual Slim Cotton Shirt", "category": "Fashion", "brand": "Allen Solly", "unit_price": 1599, "current_stock": 95, "reorder_level": 60, "base_demand": 20},
    {"product_id": "P13", "product_name": "Women Embroidered Kurti", "category": "Fashion", "brand": "Biba", "unit_price": 1899, "current_stock": 80, "reorder_level": 55, "base_demand": 18},
    {"product_id": "P14", "product_name": "Unisex Running Road Shoes", "category": "Fashion", "brand": "Puma", "unit_price": 2799, "current_stock": 65, "reorder_level": 45, "base_demand": 15},
    {"product_id": "P15", "product_name": "Classic Aviator Sunglasses", "category": "Fashion", "brand": "Ray-Ban", "unit_price": 4999, "current_stock": 40, "reorder_level": 30, "base_demand": 7}
]

# Major festive/holiday calendar (dates in format MM-DD)
HOLIDAYS_MD = {
    "01-01": "New Year",
    "01-26": "Republic Day",
    "03-25": "Holi Festival",
    "08-15": "Independence Day",
    "10-02": "Gandhi Jayanti",
    "10-12": "Dussehra",
    "11-01": "Diwali Festival",
    "11-02": "Govardhan Puja",
    "12-25": "Christmas Celebration",
    "12-31": "New Year Eve"
}

def generate_sales_data(start_date="2024-01-01", end_date="2026-06-30", random_seed=42) -> pd.DataFrame:
    """Generates realistic daily multi-store, multi-product retail sales records."""
    np.random.seed(random_seed)
    
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    records = []
    
    store_factors = {
        "ST01": 1.35, # Metro Supercenter (high volume)
        "ST02": 1.20, # Capital Hypermarket
        "ST03": 0.85, # Silicon City Express (smaller footprint)
        "ST04": 1.05, # Coastal Mart
        "ST05": 0.95  # Heritage Mall
    }
    
    for current_date in date_range:
        d_str = current_date.strftime("%Y-%m-%d")
        md = current_date.strftime("%m-%d")
        is_holiday = 1 if md in HOLIDAYS_MD else 0
        dow = current_date.dayofweek # 0=Mon, 6=Sun
        
        # Weekend multiplier (Fri=1.15, Sat=1.45, Sun=1.55)
        dow_mult = 1.0
        if dow == 4:
            dow_mult = 1.18
        elif dow == 5:
            dow_mult = 1.48
        elif dow == 6:
            dow_mult = 1.55
        
        # Month seasonality (e.g., Oct-Nov Diwali & festival season, Dec holiday shopping)
        month = current_date.month
        month_mult = 1.0
        if month in [10, 11]:
            month_mult = 1.40
        elif month == 12:
            month_mult = 1.30
        elif month in [1, 7]: # Sale clearance months
            month_mult = 1.15
        
        # Yearly growth trend (e.g. 8% year-over-year)
        year_trend = 1.0 + 0.08 * (current_date.year - 2024)
        
        # Periodic promotions (e.g., promotional campaigns on certain weeks)
        is_promo_week = (current_date.isocalendar()[1] % 4 == 0) or (is_holiday == 1)
        
        for store in STORES:
            s_id = store["store_id"]
            s_mult = store_factors[s_id]
            
            for prod in PRODUCTS:
                p_id = prod["product_id"]
                base = prod["base_demand"]
                unit_price = prod["unit_price"]
                
                # Promotion probability
                promo = 1 if (is_promo_week and np.random.rand() > 0.35) else (1 if np.random.rand() < 0.08 else 0)
                
                # Discount rate
                if promo == 1:
                    discount = float(np.random.choice([0.10, 0.15, 0.20, 0.25, 0.30], p=[0.3, 0.3, 0.2, 0.15, 0.05]))
                else:
                    discount = float(np.random.choice([0.0, 0.05], p=[0.85, 0.15]))
                
                # Demand factors
                discount_lift = 1.0 + (discount * 1.8) # Price elasticity
                promo_lift = 1.35 if promo == 1 else 1.0
                holiday_lift = 1.65 if is_holiday == 1 else 1.0
                
                expected_demand = base * s_mult * dow_mult * month_mult * year_trend * promo_lift * holiday_lift * discount_lift
                
                # Poisson/Negative Binomial noise
                quantity = int(np.random.poisson(expected_demand))
                if quantity < 1:
                    quantity = int(np.random.choice([0, 1], p=[0.4, 0.6]))
                
                # Calculate revenue
                actual_price = unit_price * (1.0 - discount)
                revenue = round(quantity * actual_price, 2)
                
                records.append({
                    "date": d_str,
                    "store_id": s_id,
                    "product_id": p_id,
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "discount": discount,
                    "promotion": promo,
                    "holiday": is_holiday,
                    "revenue": revenue
                })
                
    df = pd.DataFrame(records)
    return df

def run_data_pipeline(seed_mongo: bool = True):
    """Executes data generation, CSV export, and MongoDB seeding."""
    print("=" * 60)
    print("Step 1: Generating Retail Sales Dataset (2024 to 2026)...")
    print("=" * 60)
    
    df = generate_sales_data()
    print(f"Generated {len(df):,} transaction records across {len(STORES)} stores and {len(PRODUCTS)} products.")
    
    # Save to data/raw_sales.csv
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "raw_sales.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved raw dataset to: {csv_path}")
    
    if seed_mongo:
        print("\n" + "=" * 60)
        print("Step 2: Ingesting Data into MongoDB...")
        print("=" * 60)
        db = get_db()
        print(f"Database status: {db.get_status()}")
        
        # Ingest Stores
        print(f"Inserting {len(STORES)} stores into 'stores' collection...")
        db.insert_stores(STORES)
        
        # Ingest Products
        clean_products = [{k: v for k, v in p.items() if k != "base_demand"} for p in PRODUCTS]
        print(f"Inserting {len(clean_products)} products into 'products' collection...")
        db.insert_products(clean_products)
        
        # Ingest Sales (take latest 10,000 for quick MongoDB initialization or full)
        print("Inserting sales records into 'sales' collection...")
        db.insert_sales_df(df)
        
        # Initialize Inventory Collection
        print("Initializing initial 'inventory' records...")
        inventory_records = []
        for p in clean_products:
            # Current stock, default placeholder recommended stock
            curr = p["current_stock"]
            reorder = p["reorder_level"]
            inventory_records.append({
                "product_id": p["product_id"],
                "current_stock": curr,
                "predicted_demand": 0,
                "recommended_stock": reorder * 2,
                "reorder_quantity": max(0, (reorder * 2) - curr),
                "status": "Stock Sufficient" if curr >= reorder else "Consider Reordering"
            })
        db.upsert_inventory(inventory_records)
        
        counts = db.get_counts()
        print(f"MongoDB Collection Records: {counts}")
        print("\nData collection & MongoDB ingestion completed successfully!")

if __name__ == "__main__":
    run_data_pipeline()
