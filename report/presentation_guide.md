# Final-Year Project Presentation & Viva Defense Guide

**Project Title:** Smart Sales Prediction and Demand Forecasting System Using Machine Learning and MongoDB  
**Target Duration:** 15 – 20 Minutes  

---

## 20-Step Presentation Slide Structure & Talking Points

### Slide 1: Title & Introduction
- **Slide Content:** Project Title, Student Details, Guide/Department, Tech Stack Logos (Python, MongoDB, XGBoost, Streamlit, Scikit-learn).
- **Speaker Script:** "Good morning/afternoon respected panel members. Today I will present our project: *Smart Sales Prediction and Demand Forecasting System Using Machine Learning and MongoDB*. Our objective is to replace traditional static inventory heuristics with an AI-driven predictive supply chain system."

### Slide 2: Problem Statement
- **Slide Content:** Retail dilemmas: Stockouts vs. Overstocking, Capital wastage, unpredictable holiday surges.
- **Key Metric:** Stockouts cause an estimated 4-8% revenue loss in retail, while overstocking inflates carrying costs by 20-30% annually.

### Slide 3: Existing Systems & Their Limitations
- **Slide Content:** Comparison table: Moving averages, manual spreadsheets, static reorder levels.
- **Speaker Point:** "Existing systems treat demand as linear and static. They fail to capture complex non-linear factors like promotional discounts, day-of-week patterns, and calendar holidays."

### Slide 4: Proposed System Overview
- **Slide Content:** End-to-end pipeline combining NoSQL document storage (MongoDB), advanced feature engineering, gradient-boosted ML models, and automated inventory optimization.

### Slide 5: System Architecture Diagram
- **Slide Content:** Architecture flowchart showing Data Layer -> Preprocessing -> Feature Engineering -> ML Benchmark -> Prediction Engine -> Smart Inventory -> Streamlit Dashboard.

### Slide 6: MongoDB Database Design & Collections
- **Slide Content:** Schemas for the 5 collections: `sales`, `products`, `stores`, `predictions`, and `inventory`.
- **Panel Talking Point:** "Why MongoDB? Retail sales events are heterogeneous. MongoDB allows high write-throughput, flexible indexing on `(date, store_id, product_id)`, and clean document-oriented models for inventory status."

### Slide 7: Data Collection & Dataset Characteristics
- **Slide Content:** Multi-store (5 stores), multi-product (15 SKUs across 4 categories: Electronics, Groceries, Home & Kitchen, Fashion), spanning 2.5 years (68,400 transaction records).

### Slide 8: Data Preprocessing & Integrity
- **Slide Content:** Deduplication, missing value imputation, domain range verification, 99.5th percentile outlier handling, and mathematical revenue synchronization.

### Slide 9: Exploratory Data Analysis (EDA)
- **Slide Content:** Visual charts: Weekend volume uplift (+50%), Holiday surges (+65%), and promotional demand shifts.

### Slide 10: Feature Engineering Strategy
- **Slide Content:** 29 predictive features:
  - Temporal: Day of week, month, quarter, cyclical sine/cosine encodings.
  - Historical Lags: Lag 1, Lag 7, Lag 14, Lag 30 days.
  - Rolling Windows: 7-day, 14-day, 30-day moving averages and 7-day standard deviation.
  - Interaction Terms: Effective price and promotional holiday intersections.

### Slide 11: Machine Learning Algorithms Benchmarked
- **Slide Content:** Explanation of the 4 evaluated models:
  1. Linear Regression (Baseline)
  2. Ridge Regression ($L_2$ Regularization)
  3. Random Forest (Bagging ensemble of decision trees)
  4. XGBoost Regressor (Gradient-boosted decision trees with shrinkage and column subsampling)

### Slide 12: Model Evaluation & Benchmark Results
- **Slide Content:** Actual benchmark table:
  - Linear Regression: MAE = 9.422, RMSE = 14.469, $R^2$ = 0.8714
  - Ridge Regression: MAE = 9.420, RMSE = 14.469, $R^2$ = 0.8714
  - Random Forest: MAE = 5.625, RMSE = 8.544, $R^2$ = 0.9552
  - **XGBoost Regressor: MAE = 5.235, RMSE = 7.519, $R^2$ = 0.9653**

### Slide 13: Feature Importance Analysis
- **Slide Content:** Top features driving predictions: `rolling_mean_7`, `lag_1`, `effective_price`, `promotion`, `day_of_week`.

### Slide 14: Sales & Demand Prediction Engine (Live Demo)
- **Demonstration:** Show user input form in Streamlit: Select Store ST01, Product P01, set 15% discount and promo flag -> View instant predicted quantity and expected revenue with 90% confidence interval.

### Slide 15: Smart Inventory Recommendation Logic
- **Slide Content:** Supply chain mathematical formulation:
  $$\text{Safety Stock} = Z \cdot \sigma_d \cdot \sqrt{L/7}$$
  $$\text{Recommended Stock} = \text{Predicted Demand} + \text{Safety Stock}$$
  $$\text{Reorder Quantity} = \max(0, \text{Recommended Stock} - \text{Current Stock})$$

### Slide 16: Risk Classification & Stockout Prevention
- **Slide Content:** Color-coded status thresholds:
  - `Stockout Risk` (Red - Stock < Safety Buffer)
  - `Consider Reordering` (Amber - Stock < Reorder Point)
  - `Stock Sufficient` (Green - Adequate stock)

### Slide 17: Interactive Streamlit Dashboard Walkthrough
- **Demonstration:** Walk the panel through the 6 modules:
  1. Executive Dashboard (KPIs, revenue area chart, category donut)
  2. Sales Analytics (Filters, heatmaps, elasticity scatter)
  3. Sales Predictor (Simulation form)
  4. Smart Inventory (Restock matrix, total reorder capital)
  5. Model Performance (Metric benchmark & importance)
  6. MongoDB Data Hub (Document inspector)

### Slide 18: Operational Results & Business Impact
- **Slide Content:** 48% reduction in forecasting error vs baseline, elimination of guesswork in safety stock calculations, clear visibility of restock capital requirements.

### Slide 19: Limitations & Future Enhancements
- **Slide Content:** Current limitations (requires historical lag warm-up, closed catalog). Future scope: Deep learning (TFT/LSTM), supplier automated purchase orders, external weather APIs.

### Slide 20: Conclusion & References
- **Slide Content:** Concluding summary, key technologies used, Q&A invitation.

---

## Likely Viva / Defense Questions & Answers

**Q1: Why did you choose XGBoost over a Deep Learning model like LSTM or Transformer?**  
*Answer:* For tabular retail sales with engineered temporal lags and rolling statistics, tree-based gradient boosting algorithms like XGBoost consistently achieve state-of-the-art accuracy with faster training times, lower compute overhead, and direct feature importance interpretability.

**Q2: How do you prevent data leakage during time-series model evaluation?**  
*Answer:* We performed a strict chronological train/test split (the first 80% of dates for training, the subsequent 20% strictly for testing). Furthermore, all rolling and lag features are calculated with a 1-step backward shift (`shift(1)`), ensuring no future target values are accessible during training.

**Q3: How does your system handle scenarios where MongoDB is down?**  
*Answer:* We engineered a fault-tolerant database manager in `src/database.py`. If a live MongoDB server is not detected, it automatically activates an embedded mock engine that preserves identical collection schemas, CRUD operations, and persistent caching, guaranteeing zero downtime.
