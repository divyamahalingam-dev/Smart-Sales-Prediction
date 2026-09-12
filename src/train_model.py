import os
import sys
import json
import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))
from database import get_db
from preprocessing import clean_sales_data
from feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ModelTraining")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_evaluate_models(df_input: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    End-to-end model training, evaluation, and serialization:
    1. Loads dataset from df_input, MongoDB, or raw_sales.csv.
    2. Cleans data via preprocessing pipeline.
    3. Engineers features (calendar, lags, rolling stats).
    4. Chronological 80/20 train-test split.
    5. Trains:
       - Baseline: Linear Regression
       - Ridge Regression (regularized linear)
       - Random Forest Regressor
       - XGBoost Regressor (primary candidate)
    6. Computes MAE, RMSE, R² on unseen test set.
    7. Identifies best model, exports weights & metadata.
    """
    print("=" * 65)
    print("Step 1: Loading & Cleaning Sales Dataset...")
    print("=" * 65)
    
    if df_input is not None:
        df_raw = df_input
    else:
        csv_path = os.path.join(DATA_DIR, "raw_sales.csv")
        if os.path.exists(csv_path):
            df_raw = pd.read_csv(csv_path)
        else:
            db = get_db()
            df_raw = db.load_sales_df()
        
    cleaned_df, clean_stats = clean_sales_data(df_raw)
    print(f"Loaded {len(cleaned_df):,} records spanning {clean_stats['date_range'][0]} to {clean_stats['date_range'][1]}.")
    
    print("\n" + "=" * 65)
    print("Step 2: Feature Engineering & Preprocessing...")
    print("=" * 65)
    
    db = get_db()
    products_meta = db.get_products()
    
    fe = FeatureEngineer()
    X, y, feature_cols = fe.fit_transform(cleaned_df, products_meta)
    
    # Chronological time-series split (80% train, 20% test)
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    test_dates = cleaned_df["date"].iloc[split_idx:]
    print(f"Training set: {len(X_train):,} rows | Test set: {len(X_test):,} rows")
    print(f"Evaluation period: {test_dates.min().strftime('%Y-%m-%d')} to {test_dates.max().strftime('%Y-%m-%d')}")
    
    print("\n" + "=" * 65)
    print("Step 3: Training Multiple Machine Learning Models...")
    print("=" * 65)
    
    models = {
        "Linear Regression (Baseline)": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1),
        "XGBoost Regressor": XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1)
    }
    
    results = {}
    fitted_models = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Demand cannot be negative
        y_pred = np.clip(y_pred, 0, None)
        
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        
        results[name] = {
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "R2": round(r2, 4)
        }
        fitted_models[name] = model
        print(f"  --> {name:<30} | MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.4f}")
        
    # Select best model based on highest R² and lowest RMSE
    best_name = max(results, key=lambda k: results[k]["R2"])
    best_model = fitted_models[best_name]
    print("\n" + "=" * 65)
    print(f"Best Performing Model: {best_name}")
    print(f"Metrics -> MAE: {results[best_name]['MAE']}, RMSE: {results[best_name]['RMSE']}, R²: {results[best_name]['R2']}")
    print("=" * 65)
    
    # Extract Feature Importance (from XGBoost or Random Forest)
    feature_importance = {}
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feature_importance = dict(sorted(zip(feature_cols, [round(float(v), 4) for v in importances]), key=lambda x: x[1], reverse=True))
    elif hasattr(fitted_models.get("XGBoost Regressor"), "feature_importances_"):
        importances = fitted_models["XGBoost Regressor"].feature_importances_
        feature_importance = dict(sorted(zip(feature_cols, [round(float(v), 4) for v in importances]), key=lambda x: x[1], reverse=True))
        
    # Save artifacts
    model_bundle = {
        "best_model_name": best_name,
        "model": best_model,
        "feature_engineer": fe,
        "feature_columns": feature_cols,
        "all_models": fitted_models,
        "metrics": results,
        "feature_importance": feature_importance
    }
    
    pkl_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(model_bundle, pkl_path)
    print(f"Saved complete model bundle to: {pkl_path}")
    
    # Save JSON metadata for lightweight reporting/inspection
    metadata = {
        "best_model": best_name,
        "metrics": results,
        "feature_columns": feature_cols,
        "top_features": list(feature_importance.keys())[:10],
        "training_records": len(X_train),
        "test_records": len(X_test),
        "evaluation_period": {
            "start": test_dates.min().strftime("%Y-%m-%d"),
            "end": test_dates.max().strftime("%Y-%m-%d")
        }
    }
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to: {meta_path}")
    return metadata

if __name__ == "__main__":
    train_and_evaluate_models()
