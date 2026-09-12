# Smart Sales Prediction and Demand Forecasting System Using Machine Learning and MongoDB

**Academic & Technical Project Report**  
**Domain:** Data Science, Machine Learning, Business Analytics, NoSQL Databases  

---

## Executive Summary
In contemporary retail and supply chain operations, accurate sales forecasting and proactive inventory management directly dictate profitability, customer satisfaction, and capital efficiency. Traditional inventory systems rely on static threshold formulas or subjective rule sets that fail to accommodate multi-variate interactions such as calendar seasonality, promotional uplifts, holiday surges, and price elasticity. 

This project implements an end-to-end data science and machine learning solution: **Smart Sales Prediction & Demand Forecasting System**. The system leverages **MongoDB** as a flexible document data layer, an advanced Python feature engineering pipeline (temporal lags, rolling averages, calendar cyclical encodings), and rigorous benchmarking of multiple machine learning models (**Linear Regression, Ridge Regression, Random Forest, and XGBoost Regressor**). The best-performing model achieves an **$R^2$ of 0.9653** and a **Root Mean Squared Error (RMSE) of 7.519 units** on unseen test data.

The project integrates an automated **Smart Inventory Optimization Engine** calculating dynamic safety stocks ($Z=1.65$ for 95% cycle service level), recommended stock targets, reorder quantities, and automated risk classification (`Stockout Risk`, `Consider Reordering`, `Stock Sufficient`). A multi-page interactive **Streamlit Dashboard** provides enterprise-grade visual analytics, real-time prediction forms, automated inventory reordering reports, and live MongoDB introspection.

---

## 1. Problem Statement
Retail organizations across physical and omnichannel channels struggle with:
1. **Stockouts:** Inability to fulfill customer demand during holiday surges or promotional campaigns, resulting in direct revenue loss and customer dissatisfaction.
2. **Overstocking & Capital Lock-In:** Accumulation of slow-moving inventory, leading to elevated warehousing carrying costs, depreciation, and obsolescence.
3. **Decoupled Pricing & Demand:** Lack of dynamic visibility into price elasticity and discount effectiveness.
4. **Disparate Silos:** Unstructured sales logs separated from predictive inventory and decision-making systems.

---

## 2. Existing Limitations vs Proposed System

| Dimension | Traditional Retail Systems | Proposed Smart System |
| :--- | :--- | :--- |
| **Forecasting Method** | Static Moving Averages / Intuition | Machine Learning (XGBoost, Random Forest) |
| **Feature Dynamics** | Single-variable historical sales | 29 engineered temporal, lag, rolling & holiday features |
| **Inventory Logic** | Fixed min-max reorder points | Dynamic Safety Stock based on demand variance ($Z \cdot \sigma_d \sqrt{L}$) |
| **Database Architecture** | Rigid relational schemas | Scalable, document-oriented NoSQL (MongoDB) |
| **Decision Support** | Static spreadsheet reports | Interactive real-time Streamlit BI Dashboard |

---

## 3. System Architecture & Workflow

```
+-----------------------------------------------------------------------------+
|                                DATA LAYER                                   |
|  Retail Transactions -> MongoDB (sales, products, stores, predictions, inv) |
+-----------------------------------------------------------------------------+
                                     |
                                     v
+-----------------------------------------------------------------------------+
|                           PREPROCESSING & EDA                               |
|  Deduplication -> Outlier Capping -> Missing Imputation -> Visual Heatmaps  |
+-----------------------------------------------------------------------------+
                                     |
                                     v
+-----------------------------------------------------------------------------+
|                           FEATURE ENGINEERING                               |
|  Calendar (DOW, Month, Holiday) -> Lags (1,7,14,30d) -> Rolling Means & STD |
+-----------------------------------------------------------------------------+
                                     |
                                     v
+-----------------------------------------------------------------------------+
|                           MODEL BENCHMARKING                                |
|  Linear Baseline vs Ridge vs Random Forest vs XGBoost -> Metric Evaluation  |
+-----------------------------------------------------------------------------+
                                     |
                                     v
+-----------------------------------------------------------------------------+
|                         BUSINESS INFERENCE ENGINE                           |
|  Point Demand Prediction -> Revenue Estimation -> Safety Stock & Reorder Qty|
+-----------------------------------------------------------------------------+
                                     |
                                     v
+-----------------------------------------------------------------------------+
|                           STREAMLIT DASHBOARD                               |
|  Executive KPIs -> EDA Filters -> Inference Form -> Inventory Restock Table |
+-----------------------------------------------------------------------------+
```

---

## 4. MongoDB Database Schema Design

The system implements 5 core collections in the `smart_sales_db` database:

1. **`sales` Collection**:
   - `date`: Transaction date (ISO format `YYYY-MM-DD`).
   - `store_id`: Store identifier (e.g., `ST01`, `ST02`).
   - `product_id`: Product SKU identifier (e.g., `P01`, `P02`).
   - `quantity`: Number of units sold.
   - `unit_price`: Base selling price.
   - `discount`: Discount percentage applied ($0.00 - 0.90$).
   - `promotion`: Binary flag ($1$ = promotional campaign, $0$ = regular).
   - `holiday`: Binary flag ($1$ = public holiday/festival, $0$ = normal day).
   - `revenue`: Realized revenue $= \text{quantity} \times \text{unit\_price} \times (1 - \text{discount})$.
   - *Index*: `(date ASC, store_id ASC, product_id ASC)`

2. **`products` Collection**:
   - `product_id`, `product_name`, `category`, `brand`, `current_stock`, `reorder_level`.
   - *Index*: `product_id` (Unique).

3. **`stores` Collection**:
   - `store_id`, `store_name`, `location`, `store_type`.
   - *Index*: `store_id` (Unique).

4. **`predictions` Collection**:
   - `prediction_date`, `product_id`, `store_id`, `predicted_quantity`, `predicted_revenue`, `model_name`, `created_at`.
   - *Index*: `(prediction_date ASC, product_id ASC)`.

5. **`inventory` Collection**:
   - `product_id`, `current_stock`, `predicted_demand`, `safety_stock`, `recommended_stock`, `reorder_quantity`, `status`.
   - *Index*: `product_id` (Unique).

*Resilience Architecture:* The database layer (`src/database.py`) features a transparent fallback engine. If a live MongoDB server is unavailable, the application operates seamlessly using an embedded mock storage engine while preserving full MongoDB query semantics and persistence.

---

## 5. Exploratory Data Analysis & Key Findings

Analysis of 68,400 transaction records across 5 stores and 15 products revealed:
- **Day of Week Effect:** Saturday and Sunday experience an average sales volume **48% to 55% higher** than mid-week days (Tuesday-Wednesday).
- **Festival & Holiday Lift:** Public holidays (Diwali, New Year, Republic Day, Independence Day) exhibit an average **65% increase** in demand volume.
- **Promotional Sensitivity:** Promotional campaigns combined with a 15-20% discount yield an average **35% demand uplift** across fast-moving categories.
- **Category Variations:** Groceries exhibit high turnover volume with lower ticket size; Electronics display lower unit velocity with higher ticket margins.

---

## 6. Feature Engineering Matrix

The feature pipeline transforms raw transactional timestamps into 29 predictive features:
- **Temporal Calendar:** `year`, `month`, `week`, `day`, `day_of_week`, `is_weekend`, `quarter`.
- **Cyclical Features:** `month_sin`, `month_cos`, `dow_sin`, `dow_cos` capturing continuous cyclical periodicity.
- **Lagged Demand:** `lag_1`, `lag_7`, `lag_14`, `lag_30` representing recent sales history.
- **Rolling Windows:** `rolling_mean_7`, `rolling_mean_14`, `rolling_mean_30`, `rolling_std_7` capturing localized moving momentum and demand volatility.
- **Economic Interactions:** `effective_price`, `discount_amount`, `promo_holiday_inter`.
- **Categorical Encodings:** Label-encoded `store_encoded`, `product_encoded`, `category_encoded`.

---

## 7. Machine Learning Model Benchmark & Evaluation

Models were evaluated on a chronological 80/20 train/test split (54,720 training rows, 13,680 unseen evaluation rows):

| Model Algorithm | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression (Baseline)** | 9.422 | 14.469 | 0.8714 |
| **Ridge Regression** | 9.420 | 14.469 | 0.8714 |
| **Random Forest Regressor** | 5.625 | 8.544 | 0.9552 |
| **XGBoost Regressor (Best)** | **5.235** | **7.519** | **0.9653** |

### Key Takeaways:
- **XGBoost** outperformed all algorithms, reducing RMSE by **48.0%** relative to the linear baseline.
- Gradient boosted tree ensembles effectively model nonlinear interaction terms between promotions, holidays, day-of-week, and rolling lag windows.

---

## 8. Smart Inventory & Safety Stock Optimization

The inventory optimization engine applies industrial supply chain formulas:

$$\text{Safety Stock } (SS) = Z \times \sigma_d \times \sqrt{\frac{L}{7}}$$
$$\text{Recommended Stock } = \text{Predicted Demand } (D_{\text{pred}}) + SS$$
$$\text{Reorder Quantity} = \max(0, \text{Recommended Stock} - \text{Current Stock})$$

Where:
- $Z = 1.65$ (standard normal z-score corresponding to a 95% cycle service level).
- $\sigma_d$: Standard deviation of daily forecast demand.
- $L = 7$ days (supplier lead time).

### Classification Thresholds:
- **Stockout Risk (Red):** $\text{Current Stock} \le SS$ or $\text{Current Stock} < 0.20 \times D_{\text{pred}}$.
- **Consider Reordering (Amber):** $\text{Current Stock} \le \text{reorder\_level}$ or $\text{Current Stock} < 0.70 \times \text{Recommended Stock}$.
- **Stock Sufficient (Green):** $\text{Current Stock} \ge \text{Recommended Stock}$.

---

## 9. Streamlit Dashboard Features
1. **Executive Dashboard:** Top-level revenue KPIs, monthly trend line charts, category contribution donut chart, store revenue horizontal bar chart, top volume products.
2. **Sales Analytics (EDA):** Dynamic multiselect filters by store, category, and date range; Day-of-week bar charts; Month $\times$ Day heatmap; promotional lift comparisons; price elasticity scatter plot.
3. **Sales & Demand Predictor:** Scenario simulator allowing interactive selection of store, SKU, date, unit price, discount slider, promo checkbox, holiday toggle; generates predicted units, expected revenue, 90% confidence intervals, and a 30-day forward demand trajectory chart.
4. **Smart Inventory & Reorder System:** Real-time stockout risk alerts, restock volume recommendations, estimated reorder capital required, safety stock service level selector (90%, 95%, 99%).
5. **Model Evaluation & Benchmark:** Side-by-side metric tables (MAE, RMSE, $R^2$), error comparison charts, and top 12 XGBoost feature importance rankings.
6. **MongoDB Data Hub:** Real-time collection monitor, document counts, connection mode indicator, and live document dataframes.

---

## 10. Conclusion & Future Roadmap
The system successfully bridges machine learning research and practical enterprise operations. By anchoring demand forecasting into an automated inventory decision engine backed by MongoDB, businesses can prevent stockout revenue loss while minimizing working capital tied up in excess inventory.

### Future Enhancements:
- Automated supplier purchase order generation via email/webhook.
- Deep learning time-series models (Temporal Fusion Transformers / LSTM).
- Geographic weather and macro-economic inflation indicators.
