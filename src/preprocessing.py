import logging
from typing import Dict, Tuple, Any, List, Optional
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DataPreprocessing")

def clean_sales_data(df: pd.DataFrame, cap_outliers: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw sales transaction data:
    1. Removes duplicate records.
    2. Enforces data type conversions.
    3. Handles missing values (forward fill / mean imputation).
    4. Validates domain constraints (quantity >= 0, discount in [0, 1]).
    5. Optionally caps extreme outliers using 99.5th percentile threshold.
    6. Ensures revenue mathematical consistency.
    """
    initial_count = len(df)
    logger.info(f"Starting data preprocessing on {initial_count:,} records.")
    
    clean_df = df.copy()
    
    # 1. Deduplication
    clean_df.drop_duplicates(subset=["date", "store_id", "product_id"], keep="first", inplace=True)
    duplicates_removed = initial_count - len(clean_df)
    
    # 2. Date conversion
    if not pd.api.types.is_datetime64_any_dtype(clean_df["date"]):
        clean_df["date"] = pd.to_datetime(clean_df["date"])
        
    # 3. Handle missing values
    missing_before = clean_df.isnull().sum().to_dict()
    clean_df["quantity"] = clean_df["quantity"].fillna(0)
    clean_df["unit_price"] = clean_df["unit_price"].fillna(clean_df["unit_price"].median())
    clean_df["discount"] = clean_df["discount"].fillna(0.0)
    clean_df["promotion"] = clean_df["promotion"].fillna(0).astype(int)
    clean_df["holiday"] = clean_df["holiday"].fillna(0).astype(int)
    
    # 4. Domain constraints
    clean_df["quantity"] = clean_df["quantity"].clip(lower=0).round().astype(int)
    clean_df["discount"] = clean_df["discount"].clip(lower=0.0, upper=0.90)
    clean_df["unit_price"] = clean_df["unit_price"].clip(lower=1.0)
    
    # 5. Outlier handling via 99.5th percentile cap per product
    outliers_capped = 0
    if cap_outliers:
        p995 = clean_df.groupby("product_id")["quantity"].transform(lambda x: x.quantile(0.995))
        is_outlier = clean_df["quantity"] > p995
        outliers_capped = int(is_outlier.sum())
        clean_df["quantity"] = np.where(is_outlier, np.ceil(p995).astype(int), clean_df["quantity"])
    
    # 6. Recalculate revenue consistency
    clean_df["revenue"] = (clean_df["quantity"] * clean_df["unit_price"] * (1.0 - clean_df["discount"])).round(2)
    
    # Sort chronologically
    clean_df.sort_values(by=["date", "store_id", "product_id"], inplace=True)
    clean_df.reset_index(drop=True, inplace=True)
    
    stats = {
        "initial_rows": initial_count,
        "final_rows": len(clean_df),
        "duplicates_removed": duplicates_removed,
        "outliers_capped": outliers_capped,
        "date_range": (clean_df["date"].min().strftime("%Y-%m-%d"), clean_df["date"].max().strftime("%Y-%m-%d")),
        "missing_values_handled": sum(missing_before.values())
    }
    
    logger.info(f"Preprocessing completed. Cleaned dataset has {len(clean_df):,} rows.")
    return clean_df, stats

# Column alias dictionary for flexible user uploads (supports UCI Online Retail, Kaggle Superstore, etc.)
COLUMN_ALIASES = {
    "date": [
        "date", "Date", "DATE", "order_date", "OrderDate", "Order Date", "order date",
        "InvoiceDate", "Invoice Date", "invoicedate", "invoice_date", "timestamp", "Timestamp",
        "Day", "day", "Transaction_Date", "Transaction Date", "transaction_date", "Sale_Date",
        "Sale Date", "Period", "Time", "Order_Date"
    ],
    "store_id": [
        "store_id", "Store_Id", "Store_ID", "store", "Store", "STORE", "Store ID", "store_code",
        "location_id", "Country", "country", "Region", "region", "Branch", "branch", "City",
        "city", "State", "state", "Market", "market", "Warehouse", "Outlet", "Store_Name", "Store Name"
    ],
    "product_id": [
        "product_id", "Product_Id", "Product_ID", "product", "Product", "PRODUCT", "Product ID",
        "sku", "SKU", "item_id", "StockCode", "Stock_Code", "stock_code", "Description", "description",
        "Item", "item", "Item_Name", "Item Name", "Product_Name", "Product Name", "Product_Title",
        "Title", "Model", "Item_Code", "Sub-Category"
    ],
    "quantity": [
        "quantity", "Quantity", "QUANTITY", "units", "Units", "volume", "units_sold", "Qty",
        "qty", "Units Sold", "Quantity Ordered", "Sales Count", "Count", "Num_Items"
    ],
    "unit_price": [
        "unit_price", "Unit_Price", "price", "Price", "PRICE", "unit_cost", "Price Per Unit",
        "selling_price", "Selling Price", "UnitPrice", "Unit Price", "Rate", "rate"
    ],
    "revenue": [
        "revenue", "Revenue", "REVENUE", "sales", "Sales", "SALES", "total", "Total",
        "Amount", "amount", "Line_Total", "Order_Total", "Total_Amount"
    ],
    "discount": [
        "discount", "Discount", "DISCOUNT", "discount_rate", "Discount Rate", "Discount %", "Disc"
    ],
    "promotion": [
        "promotion", "Promotion", "PROMOTION", "promo", "is_promotion", "promo_active", "Campaign", "Deal"
    ],
    "holiday": [
        "holiday", "Holiday", "HOLIDAY", "is_holiday", "holiday_flag", "Weekend"
    ]
}

def validate_sales_schema(df: pd.DataFrame) -> Tuple[bool, pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Intelligently validates, auto-detects, maps column aliases, and normalizes
    user-uploaded sales/order datasets (CSV, Excel, etc.).
    Returns:
        (is_valid, normalized_df, error_messages, summary_stats)
    """
    errors: List[str] = []
    normalized_df = df.copy()
    
    # 1. Column normalization via explicit aliases
    col_mapping = {}
    lower_cols = {str(c).strip().lower(): c for c in normalized_df.columns}
    
    for target_col, aliases in COLUMN_ALIASES.items():
        found = False
        for alias in aliases:
            if alias in normalized_df.columns:
                col_mapping[alias] = target_col
                found = True
                break
        if not found:
            for alias in aliases:
                if alias.lower() in lower_cols:
                    col_mapping[lower_cols[alias.lower()]] = target_col
                    found = True
                    break
                    
    normalized_df.rename(columns=col_mapping, inplace=True)

    # 2. Smart Fallbacks for unmapped columns
    # Fallback for Date
    if "date" not in normalized_df.columns:
        date_candidates = [c for c in normalized_df.columns if "date" in str(c).lower() or "time" in str(c).lower()]
        if date_candidates:
            normalized_df.rename(columns={date_candidates[0]: "date"}, inplace=True)
        else:
            errors.append("Could not locate a Date column (e.g. 'date', 'order_date', 'InvoiceDate').")

    # Fallback for Store
    if "store_id" not in normalized_df.columns:
        store_candidates = [c for c in normalized_df.columns if any(k in str(c).lower() for k in ["store", "country", "region", "branch", "city", "location"])]
        if store_candidates:
            normalized_df["store_id"] = normalized_df[store_candidates[0]].astype(str)
        else:
            normalized_df["store_id"] = "Online_Store"

    # Fallback for Product
    if "product_id" not in normalized_df.columns:
        prod_candidates = [c for c in normalized_df.columns if any(k in str(c).lower() for k in ["product", "sku", "item", "stock", "desc", "title"])]
        if prod_candidates:
            normalized_df["product_id"] = normalized_df[prod_candidates[0]].astype(str)
        else:
            normalized_df["product_id"] = "Product_" + (normalized_df.index % 15 + 1).astype(str)

    # Fallback for Quantity
    if "quantity" not in normalized_df.columns:
        qty_candidates = [c for c in normalized_df.columns if any(k in str(c).lower() for k in ["qty", "quant", "unit", "vol", "count"])]
        if qty_candidates:
            normalized_df["quantity"] = pd.to_numeric(normalized_df[qty_candidates[0]], errors="coerce").fillna(1)
        else:
            normalized_df["quantity"] = 1

    # Fallback for Unit Price / Revenue
    if "unit_price" not in normalized_df.columns:
        if "revenue" in normalized_df.columns:
            rev_numeric = pd.to_numeric(normalized_df["revenue"], errors="coerce").fillna(100.0)
            qty_numeric = pd.to_numeric(normalized_df["quantity"], errors="coerce").fillna(1).replace(0, 1)
            normalized_df["unit_price"] = (rev_numeric / qty_numeric).clip(lower=1.0).round(2)
        else:
            price_candidates = [c for c in normalized_df.columns if any(k in str(c).lower() for k in ["price", "cost", "rate", "amount", "sale"])]
            if price_candidates:
                normalized_df["unit_price"] = pd.to_numeric(normalized_df[price_candidates[0]], errors="coerce").fillna(100.0)
            else:
                normalized_df["unit_price"] = 199.0

    if errors:
        return False, normalized_df, errors, {}

    # 3. Validate and parse dates
    try:
        normalized_df["date"] = pd.to_datetime(normalized_df["date"], errors="coerce")
        normalized_df.dropna(subset=["date"], inplace=True)
        normalized_df["date"] = normalized_df["date"].dt.normalize()
        if len(normalized_df) == 0:
            errors.append("Date column exists but could not parse any valid date timestamps.")
            return False, normalized_df, errors, {}
    except Exception as e:
        errors.append(f"Failed to parse 'date' column: {e}")
        return False, normalized_df, errors, {}

    # 4. Clean and enforce numeric types
    normalized_df["quantity"] = pd.to_numeric(normalized_df["quantity"], errors="coerce").fillna(1)
    # Filter out returns or cancellations (negative or zero quantities)
    normalized_df = normalized_df[normalized_df["quantity"] > 0]
    normalized_df["quantity"] = normalized_df["quantity"].round().astype(int)

    normalized_df["unit_price"] = pd.to_numeric(normalized_df["unit_price"], errors="coerce").fillna(100.0).clip(lower=0.5).round(2)

    # 5. Populate optional columns
    if "discount" not in normalized_df.columns:
        normalized_df["discount"] = 0.0
    else:
        normalized_df["discount"] = pd.to_numeric(normalized_df["discount"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=0.9)

    if "promotion" not in normalized_df.columns:
        normalized_df["promotion"] = 0
    else:
        normalized_df["promotion"] = pd.to_numeric(normalized_df["promotion"], errors="coerce").fillna(0).astype(int)

    if "holiday" not in normalized_df.columns:
        normalized_df["holiday"] = (normalized_df["date"].dt.weekday >= 5).astype(int)
    else:
        normalized_df["holiday"] = pd.to_numeric(normalized_df["holiday"], errors="coerce").fillna(0).astype(int)

    # Recalculate revenue consistency
    normalized_df["revenue"] = (normalized_df["quantity"] * normalized_df["unit_price"] * (1.0 - normalized_df["discount"])).round(2)

    # 6. Clean string IDs (trim, limit description length to 32 chars for clean graphs)
    normalized_df["store_id"] = normalized_df["store_id"].astype(str).str.strip().str[:24]
    normalized_df["product_id"] = normalized_df["product_id"].astype(str).str.strip().str[:32]

    # Deduplicate & Sort chronologically
    normalized_df.drop_duplicates(subset=["date", "store_id", "product_id"], keep="first", inplace=True)
    normalized_df.sort_values(by=["date", "store_id", "product_id"], inplace=True)
    normalized_df.reset_index(drop=True, inplace=True)

    if len(normalized_df) == 0:
        errors.append("Dataset is empty after filtering non-positive records.")
        return False, normalized_df, errors, {}

    # 7. Summary statistics
    summary = {
        "row_count": len(normalized_df),
        "unique_stores": int(normalized_df["store_id"].nunique()),
        "store_list": sorted(list(normalized_df["store_id"].unique().tolist())),
        "unique_products": int(normalized_df["product_id"].nunique()),
        "product_list": sorted(list(normalized_df["product_id"].unique().tolist())),
        "total_units": int(normalized_df["quantity"].sum()),
        "total_revenue": float(normalized_df["revenue"].sum()),
        "min_date": normalized_df["date"].min().strftime("%Y-%m-%d"),
        "max_date": normalized_df["date"].max().strftime("%Y-%m-%d"),
    }
    
    return True, normalized_df, [], summary

def generate_sample_sales_csv(num_rows: int = 50) -> str:
    """Generates a sample sales dataset CSV string conforming to the system schema."""
    import random
    from datetime import datetime, timedelta
    
    sample_stores = ["S001", "S002", "S003", "S004", "S005"]
    sample_products = [
        ("P001", 1299.0), ("P002", 799.0), ("P003", 249.0),
        ("P004", 499.0), ("P005", 149.0), ("P006", 89.0),
        ("P007", 349.0), ("P008", 599.0), ("P009", 99.0),
        ("P010", 129.0), ("P011", 179.0), ("P012", 219.0),
        ("P013", 899.0), ("P014", 649.0), ("P015", 429.0)
    ]
    
    rows = ["date,store_id,product_id,quantity,unit_price,discount,promotion,holiday,revenue"]
    start_date = datetime(2024, 1, 1)
    
    for i in range(num_rows):
        cur_date = start_date + timedelta(days=i % 30)
        s_id = random.choice(sample_stores)
        p_id, price = random.choice(sample_products)
        qty = random.randint(5, 75)
        disc = random.choice([0.0, 0.05, 0.10, 0.15, 0.20])
        promo = 1 if disc > 0.10 else 0
        is_hol = 1 if cur_date.weekday() >= 5 else 0
        rev = round(qty * price * (1.0 - disc), 2)
        rows.append(f"{cur_date.strftime('%Y-%m-%d')},{s_id},{p_id},{qty},{price},{disc},{promo},{is_hol},{rev}")
        
    return "\n".join(rows)

if __name__ == "__main__":
    import os
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(base_dir, "data", "raw_sales.csv")
    if os.path.exists(csv_path):
        raw = pd.read_csv(csv_path)
        cleaned, s = clean_sales_data(raw)
        print("Summary stats:", s)
