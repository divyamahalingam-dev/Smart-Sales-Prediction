# Smart Sales Prediction and Demand Forecasting System Using Machine Learning and MongoDB

An enterprise-grade, end-to-end Data Science and Machine Learning platform for retail sales forecasting, revenue estimation, and automated inventory replenishment optimization.

---

## 🚀 Key Features

- **NoSQL MongoDB Data Layer:** Structured document store with 5 indexed collections: `sales`, `products`, `stores`, `predictions`, and `inventory` (with automatic embedded fallback engine).
- **Advanced Feature Engineering:** 29 engineered features including calendar dynamics, cyclical sine/cosine transformations, lag features (1, 7, 14, 30 days), and rolling window metrics (7, 14, 30-day means and standard deviations).
- **Rigorous ML Benchmarking:** Evaluates Linear Regression, Ridge Regression, Random Forest, and XGBoost Regressor using strict chronological out-of-time test sets.
- **Top Model Performance:** XGBoost achieves **$R^2 = 0.9653$** and **$\text{RMSE} = 7.519$ units** on unseen test data.
- **Smart Inventory Recommendation:** Automated safety stock calculation ($Z=1.65$ for 95% service level), recommended inventory targets, reorder quantities, and stockout risk classifications.
- **Interactive Streamlit Dashboard:** Multi-page analytics app with executive KPIs, dynamic EDA filters, scenario forecasting simulator, restock tables, and live MongoDB explorer.
- **Academic Project Deliverables:** 5 step-by-step Jupyter Notebooks, complete academic project report, and a 20-step final-year presentation guide.

---

## 📂 Project Structure

```
Smart-Sales-Prediction/
├── data/
│   └── raw_sales.csv             # 68,400+ transaction records (2.5 years)
├── notebooks/
│   ├── 01_data_collection.ipynb   # MongoDB setup & data ingestion
│   ├── 02_data_cleaning.ipynb     # Deduplication, outliers, integrity checks
│   ├── 03_eda.ipynb               # Exploratory data analysis & trends
│   ├── 04_feature_engineering.ipynb# Lag, rolling, & calendar features
│   └── 05_model_training.ipynb    # ML benchmarking & serialization
├── src/
│   ├── __init__.py
│   ├── database.py               # MongoDB connection manager & fallback
│   ├── data_loader.py            # Retail data generator & DB seeder
│   ├── preprocessing.py          # Data cleaning & validation
│   ├── feature_engineering.py    # Lag, rolling, & calendar feature pipeline
│   ├── train_model.py            # Model training & benchmark
│   ├── prediction.py             # Inference engine & revenue forecasting
│   └── inventory.py              # Safety stock & reorder optimization
├── models/
│   ├── best_model.pkl            # Serialized XGBoost bundle & transformers
│   └── model_metadata.json       # Benchmark metrics & feature importance
├── dashboard/
│   └── app.py                    # Streamlit interactive web dashboard
├── report/
│   ├── project_report.md         # Comprehensive academic report
│   └── presentation_guide.md     # 20-step viva defense & slide guide
├── requirements.txt              # Package dependencies
├── run_app.bat                   # 1-click dashboard launcher
└── README.md
```

---

## 🛠️ Quickstart Installation & Setup

### 1. Environment Activation
Activate the project's dedicated virtual environment:
```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Or Command Prompt
.\.venv\Scripts\activate.bat
```

*(If you wish to reinstall dependencies in another environment, run `uv pip install -r requirements.txt` or `pip install -r requirements.txt`)*

### 2. Generate Dataset & Seed MongoDB (Optional - already initialized)
```bash
python src/data_loader.py
```

### 3. Train & Benchmark Machine Learning Models
```bash
python src/train_model.py
```

### 4. Launch the Interactive Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
Or simply double-click **`run_app.bat`**!

The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 📊 Model Evaluation Summary

Benchmarked across 68,400 records with a strict chronological 80/20 train/test split:

| Model | MAE (Units) | RMSE (Units) | $R^2$ Score | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | 9.422 | 14.469 | 0.8714 | Baseline |
| **Ridge Regression** | 9.420 | 14.469 | 0.8714 | Regularized |
| **Random Forest** | 5.625 | 8.544 | 0.9552 | Nonlinear Ensemble |
| **XGBoost Regressor** | **5.235** | **7.519** | **0.9653** | **Selected Best** |

---

## 📦 Inventory Optimization Formula

$$\text{Safety Stock} = Z \times \sigma_d \times \sqrt{\frac{L}{7}}$$
$$\text{Recommended Stock} = \text{Predicted Demand} + \text{Safety Stock}$$
$$\text{Reorder Quantity} = \max(0, \text{Recommended Stock} - \text{Current Stock})$$

- **$Z = 1.65$** (95% cycle service level)
- **Status Alerts:**
  - 🚨 `Stockout Risk` (Red)
  - ⚠️ `Consider Reordering` (Amber)
  - ✅ `Stock Sufficient` (Green)

---

## 🎓 Academic Documentation & Viva Defense
- Detailed Project Report: [report/project_report.md](report/project_report.md)
- Presentation & Viva Guide: [report/presentation_guide.md](report/presentation_guide.md)
- Jupyter Notebooks: [notebooks/](notebooks/)
