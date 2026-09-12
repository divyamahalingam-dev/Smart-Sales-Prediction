import json
import os

NOTEBOOKS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "notebooks"))
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.12.14"
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [s + "\n" for s in source.strip().split("\n")]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [s + "\n" for s in source.strip().split("\n")]
    }

# -------------------------------------------------------------
# 1. 01_data_collection.ipynb
# -------------------------------------------------------------
nb1_cells = [
    md_cell("# 01. Data Collection & MongoDB Database Ingestion\n"
            "**Smart Sales Prediction and Demand Forecasting System**\n\n"
            "This notebook demonstrates:\n"
            "1. Connecting to MongoDB (or local mock engine if MongoDB is not running).\n"
            "2. Defining collections: `sales`, `products`, `stores`, `predictions`, `inventory`.\n"
            "3. Ingesting historical multi-store retail sales data."),
    code_cell("import sys, os\nsys.path.append('../src')\nfrom database import get_db\nimport pandas as pd\n\ndb = get_db()\nprint('MongoDB Status:', db.get_status())"),
    code_cell("# Inspect collections\ncounts = db.get_counts()\nprint('Collection Record Counts:')\nfor coll, cnt in counts.items():\n    print(f'  - {coll}: {cnt:,}')"),
    code_cell("# Sample Products\nproducts = db.get_products()\npd.DataFrame(products).head(10)"),
    code_cell("# Sample Stores\nstores = db.get_stores()\npd.DataFrame(stores)")
]

with open(os.path.join(NOTEBOOKS_DIR, "01_data_collection.ipynb"), "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb1_cells), f, indent=2)

# -------------------------------------------------------------
# 2. 02_data_cleaning.ipynb
# -------------------------------------------------------------
nb2_cells = [
    md_cell("# 02. Data Preprocessing & Cleaning Pipeline\n"
            "**Smart Sales Prediction and Demand Forecasting System**\n\n"
            "This notebook handles:\n"
            "1. Deduplication and data integrity verification.\n"
            "2. Missing value handling and domain constraint validations.\n"
            "3. Outlier detection and capping.\n"
            "4. Mathematical consistency of price, discount, quantity, and revenue."),
    code_cell("import sys, os\nsys.path.append('../src')\nimport pandas as pd\nfrom preprocessing import clean_sales_data\n\nraw_path = '../data/raw_sales.csv'\nraw_df = pd.read_csv(raw_path)\nprint(f'Raw dataset shape: {raw_df.shape}')\nraw_df.head()"),
    code_cell("# Run preprocessing\ncleaned_df, stats = clean_sales_data(raw_df, cap_outliers=True)\nprint('Cleaning Summary:')\nfor k, v in stats.items():\n    print(f'  {k}: {v}')"),
    code_cell("# Verify numerical constraints and revenue formula\nrevenue_check = (cleaned_df['quantity'] * cleaned_df['unit_price'] * (1.0 - cleaned_df['discount'])).round(2)\ndiff = (cleaned_df['revenue'] - revenue_check).abs().max()\nprint(f'Maximum discrepancy in revenue calculation: {diff:.4f}')")
]

with open(os.path.join(NOTEBOOKS_DIR, "02_data_cleaning.ipynb"), "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb2_cells), f, indent=2)

# -------------------------------------------------------------
# 3. 03_eda.ipynb
# -------------------------------------------------------------
nb3_cells = [
    md_cell("# 03. Exploratory Data Analysis (EDA)\n"
            "**Smart Sales Prediction and Demand Forecasting System**\n\n"
            "Visualizing trends, seasonality, store performance, holiday surges, and promotional lifts."),
    code_cell("import sys, os\nsys.path.append('../src')\nimport pandas as pd\nimport plotly.express as px\n\nclean_df = pd.read_csv('../data/raw_sales.csv')\nclean_df['date'] = pd.to_datetime(clean_df['date'])\nclean_df.info()"),
    code_cell("# Monthly Revenue Trend\nclean_df['year_month'] = clean_df['date'].dt.to_period('M').dt.to_timestamp()\nmonthly = clean_df.groupby('year_month')['revenue'].sum().reset_index()\nfig = px.line(monthly, x='year_month', y='revenue', title='Monthly Revenue Trend')\nfig.show()"),
    code_cell("# Day of Week Demand Distribution\ndow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']\nclean_df['dow'] = clean_df['date'].dt.dayofweek.map(lambda x: dow_names[x])\ndow_summary = clean_df.groupby('dow')['quantity'].mean().reindex(dow_names).reset_index()\nfig_dow = px.bar(dow_summary, x='dow', y='quantity', title='Average Sales Quantity by Day of Week')\nfig_dow.show()"),
    code_cell("# Promotional Lift\npromo_comp = clean_df.groupby('promotion')['quantity'].agg(['mean', 'median', 'std'])\nprint('Promotional Uplift Table:')\nprint(promo_comp)")
]

with open(os.path.join(NOTEBOOKS_DIR, "03_eda.ipynb"), "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb3_cells), f, indent=2)

# -------------------------------------------------------------
# 4. 04_feature_engineering.ipynb
# -------------------------------------------------------------
nb4_cells = [
    md_cell("# 04. Feature Engineering for Demand Forecasting\n"
            "**Smart Sales Prediction and Demand Forecasting System**\n\n"
            "Constructing temporal calendar features, lag features (1, 7, 14, 30 days), rolling statistics, and interaction terms."),
    code_cell("import sys, os\nsys.path.append('../src')\nimport pandas as pd\nfrom feature_engineering import FeatureEngineer\n\nraw_df = pd.read_csv('../data/raw_sales.csv')\nraw_df['date'] = pd.to_datetime(raw_df['date'])\n\nfe = FeatureEngineer()\nX, y, feature_cols = fe.fit_transform(raw_df)\nprint(f'Constructed {len(feature_cols)} features across {len(X):,} records.')\nprint('Feature List:', feature_cols)"),
    code_cell("# Inspect correlation of top features with sales quantity\ncorr_df = X.copy()\ncorr_df['target_quantity'] = y\ncorrs = corr_df.corr()['target_quantity'].sort_values(ascending=False)\nprint('Top Positive & Negative Correlations with Demand:')\nprint(corrs.head(10))\nprint(corrs.tail(5))")
]

with open(os.path.join(NOTEBOOKS_DIR, "04_feature_engineering.ipynb"), "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb4_cells), f, indent=2)

# -------------------------------------------------------------
# 5. 05_model_training.ipynb
# -------------------------------------------------------------
nb5_cells = [
    md_cell("# 05. Machine Learning Model Training & Evaluation\n"
            "**Smart Sales Prediction and Demand Forecasting System**\n\n"
            "Benchmarking:\n"
            "1. Linear Regression (Baseline)\n"
            "2. Ridge Regression\n"
            "3. Random Forest Regressor\n"
            "4. XGBoost Regressor\n"
            "Using chronological 80/20 train/test evaluation metrics: MAE, RMSE, and R²."),
    code_cell("import sys, os\nsys.path.append('../src')\nimport json, joblib\nimport pandas as pd\n\n# Inspect model metadata\nwith open('../models/model_metadata.json', 'r') as f:\n    meta = json.load(f)\n\nprint('Best Selected Model:', meta['best_model'])\nprint('Evaluation Benchmark Metrics:')\npd.DataFrame.from_dict(meta['metrics'], orient='index')"),
    code_cell("# Load serialized bundle\nbundle = joblib.load('../models/best_model.pkl')\nprint('Top 10 Feature Importances:')\nfor f, imp in list(bundle['feature_importance'].items())[:10]:\n    print(f'  {f:<22}: {imp:.4f}')")
]

with open(os.path.join(NOTEBOOKS_DIR, "05_model_training.ipynb"), "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb5_cells), f, indent=2)

print("Generated all 5 Jupyter Notebooks successfully in:", NOTEBOOKS_DIR)
