# 📊 Smart Sales Prediction & Demand Forecasting System
### *An Enterprise End-to-End Machine Learning Platform with NoSQL MongoDB Architecture & Interactive Business Intelligence Dashboard*

[![Live Streamlit App](https://img.shields.io/badge/Live_App-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://smart-sales-predictiongit.streamlit.app)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/divyamahalingam-dev/Smart-Sales-Prediction)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Database](https://img.shields.io/badge/Database-MongoDB%20NoSQL-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![ML Model](https://img.shields.io/badge/Model-XGBoost%20Regressor%20(R%C2%B2%3D0.965)-orange?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)

---

## 🌐 Live Web Deployment & Demo Access

The system is deployed and accessible worldwide:
👉 **[Launch Live Dashboard on Streamlit Cloud](https://smart-sales-predictiongit.streamlit.app)**

### ⚡ Instant 1-Click Demo Credentials:
The application includes an **Enterprise Authentication Gatekeeper** with 1-click demo login buttons (no typing required for rapid evaluation):

| Role | Username | Password | Access Privileges |
| :--- | :--- | :--- | :--- |
| 👑 **Administrator** | `admin` | `admin123` | Full enterprise control, dataset upload, AI model retraining & DB hub |
| 💼 **Operations Manager** | `manager` | `sales123` | Store analytics leaderboard, inventory radar & safety stock recommendations |
| 📈 **Demand Analyst** | `analyst` | `analyst123` | ML benchmark arena, split-screen scenario simulator & revenue forecasting |
| 👩‍💻 **Project Owner** | `divya` | `divya123` | Lead Data Scientist / Project Owner |
| 👨‍💻 **ML Architect** | `santhosh` | `santhosh123` | Lead Machine Learning Systems Architect |

---

## 🎯 Executive Overview & Abstract

In modern omnichannel retail and supply chains, inaccurate demand forecasting causes severe operational inefficiencies: **stockouts** lead to immediate revenue loss and customer dissatisfaction, while **overstocking** ties up working capital and inflates carrying costs.

This project delivers an enterprise-grade, data science and machine learning platform that unifies:
1. **Multi-Store NoSQL Ingestion:** Structured MongoDB storage across 6 collections with zero-downtime embedded fallback.
2. **Dynamic 29-Dimensional Feature Pipeline:** Automated calendar cycles, multi-horizon lags, rolling volatility, and promotional multipliers.
3. **Chronological ML Benchmarking:** Strict out-of-time evaluation across Linear Regression, Ridge, Random Forest, and **XGBoost ($R^2 = 0.9653$)**.
4. **Operations Research Inventory Radar:** Safety stock computation ($Z = 1.65$ for 95% cycle service level), dynamic reorder points, and automated stockout risk classification.
5. **Corporate Business Intelligence Interface:** PowerBI/Tableau-inspired corporate light theme with top horizontal tab navigation and real-time scenario simulation.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RETAIL ENTERPRISE DATA SOURCES                           │
│     • Benchmark Dataset (68,400 Records)  • User Uploads (.csv, .xlsx, .xls) │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  ZERO-CLICK INGESTION & NORMALIZATION                       │
│     • Header alias detection (InvoiceDate, Quantity, UnitPrice, Store, SKU) │
│     • Deduplication, outlier clipping, and automated missing value handling  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   NOSQL MONGODB DOCUMENT STORE (6 Collections)              │
│   [ sales ]   [ products ]   [ stores ]   [ predictions ]   [ inventory ]   [ users ]
│     • Automatic indexing on (date, store_id, product_id, username)          │
│     • Embedded Mock MongoDB engine for zero-configuration fallback           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   29-DIMENSIONAL FEATURE ENGINEERING                        │
│   • Cyclical sine/cosine of month, day-of-year, and day-of-week             │
│   • Lags: 1, 7, 14, 30 days | Rolling Windows: 7, 14, 30-day mean & std     │
│   • Price elasticity ratios, promotion uplifts, holiday indicators           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 MACHINE LEARNING BENCHMARKING ENGINE                        │
│   • Linear Regression ($R^2=0.8714$)     • Ridge Regularization ($R^2=0.8714$)│
│   • Random Forest ($R^2=0.9552$)         • XGBoost Regressor ($R^2=0.9653$) ★│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  OPERATIONS INVENTORY OPTIMIZATION                          │
│   • Safety Stock = Z * sigma_d * sqrt(LeadTime / 7)                         │
│   • Recommended Stock = Predicted Demand + Safety Stock                     │
│   • Reorder Quantity = max(0, Recommended Stock - Current Stock)            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│            CORPORATE BUSINESS INTELLIGENCE DASHBOARD (Streamlit)            │
│  [1] Executive Overview       [2] Sales Analytics     [3] Demand Predictor  │
│  [4] Inventory Radar          [5] ML Benchmark Arena  [6] Upload Dataset    │
│  [7] MongoDB Hub Explorer     [8] Enterprise Auth & Session Profile         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Modules & Capabilities

### 1. 🔐 Enterprise Authentication & Role-Based Access Control
- Salted **SHA-256 password hashing** with session state preservation.
- Pre-configured demo buttons for instant evaluation.
- Live **User Registration** tab that persists new accounts directly into the MongoDB `users` collection.
- Top navigation bar profile badge displaying active user name, role, and a 1-click **Logout** button.

### 2. 📥 Zero-Click Automated Dataset Ingestion (Excel & CSV)
- Full support for `.xlsx`, `.xls`, and `.csv` files powered by `openpyxl` and `xlrd`.
- **Zero-Click Ingestion:** Uploading a file immediately normalizes schemas, registers newly discovered products and stores, updates MongoDB, and synchronizes all 7 dashboard tabs without requiring manual button clicks.
- **Intelligent Alias Mapper:** Auto-detects standard formats (e.g., `InvoiceDate`, `Country`, `StockCode`, `UnitPrice`, `Quantity`).
- **1-Click AI Retraining:** Train the XGBoost forecasting model on the uploaded dataset with a single click.
- **1-Click Benchmark Restore:** Instantly roll back to the default 68,400 benchmark transactions anytime.

### 3. 📈 Advanced Feature Engineering (29 Features)
- **Calendar & Cyclical:** Day-of-week, month, quarter, day-of-year, weekend flag, holiday flag, cyclical sine/cosine transforms.
- **Lag Features:** 1-day, 7-day, 14-day, and 30-day historical demand lags.
- **Rolling Aggregations:** 7-day, 14-day, and 30-day rolling moving averages and volatility (standard deviations).
- **Pricing & Interaction:** Price elasticity ratios, promotional flags (+35% uplift), and store velocity interactions.

### 4. 🔬 Machine Learning Benchmark Arena
Models evaluated under strict chronological 80/20 train/test split:

| Machine Learning Model | MAE (Units) | RMSE (Units) | $R^2$ Score | Operational Decision |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression** | 9.422 | 14.469 | 0.8714 | Baseline linear reference |
| **Ridge Regression** | 9.420 | 14.469 | 0.8714 | $L_2$ regularized benchmark |
| **Random Forest Regressor** | 5.625 | 8.544 | 0.9552 | Nonlinear ensemble |
| **XGBoost Regressor** | **5.235** | **7.519** | **0.9653** | **Selected Champion Model** |

### 5. 📦 Dynamic Inventory Optimization Radar
- **Safety Stock Formula:**
  $$\text{Safety Stock} = Z \times \sigma_d \times \sqrt{\frac{L}{7}}$$
  *(Where $Z = 1.65$ corresponds to a 95% cycle service level, $\sigma_d$ is demand standard deviation, and $L$ is vendor lead time in days).*
- **Automated Alerts:**
  - 🚨 `Stockout Risk` (Critical replenishment required immediately)
  - ⚠️ `Consider Reordering` (Approaching safety margin)
  - ✅ `Stock Sufficient` (Healthy inventory status)

### 6. 🗄️ MongoDB NoSQL Data Store Architecture
Includes 6 indexed collections:
- `sales`: Historical retail transaction records.
- `products`: Product catalog, categories, baseline demand, and unit prices.
- `stores`: Multi-store branches, geographical regions, and store formats.
- `predictions`: Out-of-time inference logs and forecasted revenues.
- `inventory`: Safety stock thresholds, lead times, and restock targets.
- `users`: Authenticated user accounts with salted password hashes and roles.
- **Zero-Config Fallback:** Automatically switches to an embedded `mongomock` engine with persistent cache if local MongoDB is offline.

---

## 📂 Project Structure

```
Smart-Sales-Prediction/
├── .streamlit/
│   └── config.toml                  # Streamlit cloud theme, minimal toolbar & server config
├── data/
│   ├── .mongo_data_store.json       # Persisted NoSQL cache for embedded mock engine
│   ├── raw_sales.csv                # Active working transaction dataset
│   └── raw_sales_benchmark.csv      # Permanent 68,400 benchmark backup dataset
├── dashboard/
│   ├── app.py                       # Main multi-page Streamlit BI dashboard
│   ├── login.py                     # Authentication portal, demo logins & user registration
│   └── styles.py                    # Corporate BI light theme CSS & Plotly palettes
├── models/
│   ├── best_model.pkl               # Serialized champion XGBoost model & feature bundle
│   └── model_metadata.json          # Model benchmark metrics & feature importances
├── notebooks/
│   ├── 01_data_collection.ipynb     # MongoDB schema initialization & data seeding
│   ├── 02_data_cleaning.ipynb       # Outlier treatment, deduplication, integrity checks
│   ├── 03_eda.ipynb                 # Demand velocity, seasonality & promotional heatmaps
│   ├── 04_feature_engineering.ipynb # 29-feature transformation pipeline
│   └── 05_model_training.ipynb      # Comparative ML model training & serialization
├── report/
│   ├── project_report.md            # Comprehensive academic final-year project report
│   └── presentation_guide.md        # 20-step viva defense & slide deck master guide
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Retail transaction generator & catalog seeder
│   ├── database.py                  # MongoDB manager, user auth, and persistent mock fallback
│   ├── feature_engineering.py       # Cyclical, lag, and rolling window feature pipeline
│   ├── inventory.py                 # Safety stock, reorder point & replenishment optimizer
│   ├── prediction.py                # Inference pipeline & revenue projection engine
│   ├── preprocessing.py             # Schema alias detection & data cleaning engine
│   └── train_model.py               # ML training, cross-validation & benchmark serialization
├── .gitignore                       # Excludes .venv, cache files, and bytecode
├── requirements.txt                 # Complete dependency specification
├── run_app.bat                      # 1-click Windows desktop launcher
└── README.md                        # Master documentation
```

---

## 🛠️ Local Installation & Setup

### Option 1: Quick Launch (Windows 1-Click)
Simply double-click the **`run_app.bat`** file in the project folder. It activates the environment and opens the dashboard in your default browser.

---

### Option 2: Step-by-Step Manual Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/divyamahalingam-dev/Smart-Sales-Prediction.git
cd Smart-Sales-Prediction
```

#### 2. Create and Activate Virtual Environment
```bash
# Using Python venv (Windows)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Using Python venv (macOS/Linux)
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. (Optional) Run Initial Data Ingestion & Model Training
The repository already includes pre-generated data and serialized models. If you wish to retrain from scratch:
```bash
python src/data_loader.py
python src/train_model.py
```

#### 5. Launch the Streamlit Web Application
```bash
streamlit run dashboard/app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🎓 Academic Documentation & Viva Defense

This repository is formatted for final-year engineering and master's degree capstone submissions:
- **Full Project Report:** [`report/project_report.md`](report/project_report.md) — Complete 9-chapter thesis document covering Literature Review, System Architecture, Mathematical Formulations, Implementation Details, Results, and Future Enhancements.
- **Viva Presentation Guide:** [`report/presentation_guide.md`](report/presentation_guide.md) — 20-step viva defense roadmap, slide deck layout, and anticipated examiner questions with model answers.
- **Interactive Jupyter Notebooks:** [`notebooks/`](notebooks/) — Five standalone notebooks demonstrating the complete data science lifecycle from raw ingestion to model benchmarking.

---

## 👥 Contributors & Authors

- **Divya Mahalingam** ([@divyamahalingam-dev](https://github.com/divyamahalingam-dev)) — *Project Owner & Lead Data Scientist*
- **Santhosh Kumar** ([@santhoshkumar-01-IT](https://github.com/santhoshkumar-01-IT)) — *Lead Machine Learning Systems Architect*

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
