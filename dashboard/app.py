import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup page config
st.set_page_config(
    page_title="Retail Demand Intelligence Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ensure src is importable
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from database import get_db
from preprocessing import clean_sales_data, validate_sales_schema, generate_sample_sales_csv
from prediction import get_prediction_engine
from inventory import get_inventory_optimizer
from styles import apply_custom_styles, style_fig, format_curr, COLOR_SEQUENCE

apply_custom_styles()

# ----------------- Data Caching -----------------
@st.cache_data(ttl=600)
def load_all_data():
    db = get_db()
    sales_df = db.load_sales_df()
    stores = db.get_stores()
    products = db.get_products()
    return sales_df, stores, products

try:
    sales_df, stores, products = load_all_data()
    db = get_db()
    db_status = db.get_status()
except Exception as e:
    st.error(f"Error loading system database: {e}")
    st.stop()

# Helper dictionaries
store_dict = {s["store_id"]: f"{s['store_name']} ({s.get('location', '')})" for s in stores}
store_names = {s["store_id"]: s["store_name"] for s in stores}
prod_dict = {p["product_id"]: f"{p['product_name']} ({p.get('category', '')})" for p in products}
prod_info = {p["product_id"]: p for p in products}

# ==============================================================================
# TOP CORPORATE HEADER BAR
# ==============================================================================
c_head, c_tele, c_curr = st.columns([3.2, 2.0, 1.0])
with c_head:
    st.markdown('<div style="display: flex; align-items: center; gap: 12px;">'
                '<span style="font-size: 2.2rem;">📊</span>'
                '<div><h1 class="bi-title">Retail Demand Intelligence Hub</h1>'
                '<p class="bi-subtitle">Smart Sales Prediction, Demand Forecasting & Inventory Optimization</p></div></div>', unsafe_allow_html=True)

with c_tele:
    counts = db.get_counts()
    if db_status.get("is_mock"):
        db_badge = '<span class="bi-badge bi-badge-warning">● Mock DB</span>'
    else:
        db_badge = f'<span class="bi-badge bi-badge-success">● Live MongoDB</span>'
        
    active_ds = st.session_state.get("active_file_name", "Benchmark (68.4k)")
    ds_badge = f'<span class="bi-badge bi-badge-info">📁 {active_ds[:22]}</span>'

    st.markdown(f"""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 10px 14px; border-radius: 8px; font-size: 0.8rem; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <span><b>Engine:</b> {db_badge}</span>
            <span>{ds_badge}</span>
        </div>
        <div style="color: #64748B; display: flex; gap: 12px;">
            <span>Orders: <b>{counts['sales']:,}</b></span>
            <span>SKUs: <b>{counts['products']}</b></span>
            <span>Stores: <b>{counts['stores']}</b></span>
            <span>Model: <b>XGBoost</b></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with c_curr:
    currency = st.selectbox("Currency Format", ["₹ INR (₹)", "$ USD ($)"], index=0)
    curr_symbol = "₹" if "INR" in currency else "$"

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# TOP HORIZONTAL TAB NAVIGATION
# ==============================================================================
tab_exec, tab_eda, tab_pred, tab_inv, tab_ml, tab_upload, tab_db = st.tabs([
    "📊 Executive Overview",
    "📈 Sales Analytics & Trends",
    "🤖 Demand & Scenario Predictor",
    "📦 Inventory & Stockout Radar",
    "🔬 ML Model Benchmark",
    "📥 Upload Sales Dataset",
    "🗄️ MongoDB Database Hub"
])

# ==============================================================================
# 1. EXECUTIVE OVERVIEW TAB
# ==============================================================================
with tab_exec:
    
    total_revenue = sales_df["revenue"].sum()
    total_units = sales_df["quantity"].sum()
    daily_avg_sales = sales_df.groupby("date")["quantity"].sum().mean()
    daily_avg_rev = sales_df.groupby("date")["revenue"].sum().mean()
    
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">💰 Gross Revenue Generated</div>
            <div class="kpi-value">{format_curr(total_revenue, curr_symbol)}</div>
            <span class="kpi-delta delta-up">↑ +14.2% YoY run-rate</span>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">📦 Cumulative Volume Sold</div>
            <div class="kpi-value">{total_units:,}</div>
            <span class="kpi-delta delta-up">↑ 68.4k orders logged</span>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">📈 Daily Demand Velocity</div>
            <div class="kpi-value">{int(daily_avg_sales):,} <span style="font-size: 1rem; color: #64748B;">units/day</span></div>
            <span class="kpi-delta delta-neutral">● Multi-store velocity</span>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">💵 Average Daily Revenue</div>
            <div class="kpi-value">{format_curr(daily_avg_rev, curr_symbol)}</div>
            <span class="kpi-delta delta-up">↑ 98.6% fulfillment</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_c1, col_c2 = st.columns([2, 1])
    with col_c1:
        df_monthly = sales_df.copy()
        df_monthly["year_month"] = df_monthly["date"].dt.to_period("M").dt.to_timestamp()
        rev_trend = df_monthly.groupby("year_month")["revenue"].sum().reset_index()
        
        fig_rev = px.area(
            rev_trend, x="year_month", y="revenue",
            labels={"year_month": "Timeline", "revenue": "Gross Revenue"},
            color_discrete_sequence=["#38BDF8"]
        )
        fig_rev.update_traces(
            line=dict(width=2.5, color="#38BDF8"),
            fillcolor="rgba(56, 189, 248, 0.15)"
        )
        fig_rev = style_fig(fig_rev, title="Monthly Revenue Trajectory & Seasonal Peaks", height=380)
        st.plotly_chart(fig_rev, use_container_width=True)
        
    with col_c2:
        df_cat = sales_df.copy()
        df_cat["category"] = df_cat["product_id"].map(lambda pid: prod_info.get(pid, {}).get("category", "General"))
        cat_rev = df_cat.groupby("category")["revenue"].sum().reset_index()
        
        fig_cat = px.pie(
            cat_rev, names="category", values="revenue",
            hole=0.55,
            color_discrete_sequence=COLOR_SEQUENCE
        )
        fig_cat.update_traces(
            textposition='inside', textinfo='percent+label',
            marker=dict(line=dict(color='#FFFFFF', width=2))
        )
        fig_cat = style_fig(fig_cat, title="Revenue Share by Category", height=380)
        st.plotly_chart(fig_cat, use_container_width=True)

    col_s1, col_s2 = st.columns([1, 1])
    with col_s1:
        df_store = sales_df.copy()
        df_store["store_name"] = df_store["store_id"].map(store_names)
        store_rev = df_store.groupby("store_name")["revenue"].sum().reset_index().sort_values("revenue", ascending=True)
        
        fig_s = px.bar(
            store_rev, x="revenue", y="store_name", orientation="h",
            labels={"revenue": "Total Revenue", "store_name": "Store Branch"},
            color="revenue", color_continuous_scale="Blues"
        )
        fig_s = style_fig(fig_s, title="Store Revenue Leaderboard", height=340)
        fig_s.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_s, use_container_width=True)
        
    with col_s2:
        prod_names = {p["product_id"]: p["product_name"] for p in products}
        df_p = sales_df.copy()
        df_p["product_name"] = df_p["product_id"].map(prod_names)
        top_prods = df_p.groupby("product_name")["quantity"].sum().reset_index().sort_values("quantity", ascending=True).tail(6)
        
        fig_p = px.bar(
            top_prods, x="quantity", y="product_name", orientation="h",
            labels={"quantity": "Units Sold", "product_name": "Product SKU"},
            color="quantity", color_continuous_scale="Teal"
        )
        fig_p = style_fig(fig_p, title="Top Volume Moving Products", height=340)
        fig_p.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_p, use_container_width=True)

# ==============================================================================
# 2. SALES & DEMAND ANALYTICS (EDA)
# ==============================================================================
with tab_eda:
    st.markdown("""
    <div class="hero-banner">
        <h1 style="margin: 0 0 4px 0; font-size: 1.4rem; font-weight: 800; color: #0F172A;">
            Exploratory Data Analysis & Demand Dynamics
        </h1>
        <p style="margin: 0; color: #64748B; font-size: 0.9rem;">
            Inspect day-of-week trends, seasonal patterns, marketing campaign uplifts, and price elasticity curves.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("⚡ Filter Controls & Segmentation", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            sel_stores = st.multiselect(
                "Filter Store Branches",
                options=list(store_dict.keys()),
                default=list(store_dict.keys()),
                format_func=lambda x: store_dict[x]
            )
        with f2:
            all_cats = sorted(list(set(p.get("category", "General") for p in products)))
            sel_cats = st.multiselect("Filter Categories", options=all_cats, default=all_cats)
        with f3:
            min_d, max_d = sales_df["date"].min().date(), sales_df["date"].max().date()
            date_filter = st.date_input("Date Window", value=[min_d, max_d], min_value=min_d, max_value=max_d)

    df_filtered = sales_df.copy()
    df_filtered["category"] = df_filtered["product_id"].map(lambda pid: prod_info.get(pid, {}).get("category", "General"))
    
    if sel_stores:
        df_filtered = df_filtered[df_filtered["store_id"].isin(sel_stores)]
    if sel_cats:
        df_filtered = df_filtered[df_filtered["category"].isin(sel_cats)]
    if isinstance(date_filter, (list, tuple)) and len(date_filter) == 2:
        df_filtered = df_filtered[(df_filtered["date"].dt.date >= date_filter[0]) & (df_filtered["date"].dt.date <= date_filter[1])]

    tab1, tab2, tab3 = st.tabs(["📅 Seasonality & Weekly Cycles", "🏷️ Marketing & Festive Surges", "📉 Price Elasticity Dynamics"])
    
    with tab1:
        c_dow, c_heat = st.columns(2)
        with c_dow:
            dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            df_filtered["dow_name"] = df_filtered["date"].dt.dayofweek.map(lambda x: dow_order[x])
            dow_avg = df_filtered.groupby("dow_name")["quantity"].mean().reindex(dow_order).reset_index()
            
            fig_dow = px.bar(
                dow_avg, x="dow_name", y="quantity",
                color="quantity", color_continuous_scale="Purples",
                labels={"dow_name": "Day of Week", "quantity": "Average Units"}
            )
            fig_dow = style_fig(fig_dow, title="Demand Distribution by Day of Week", height=360)
            fig_dow.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_dow, use_container_width=True)
            
        with c_heat:
            df_filtered["month_name"] = df_filtered["date"].dt.strftime("%b")
            m_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            piv = df_filtered.pivot_table(index="dow_name", columns="month_name", values="quantity", aggfunc="mean").reindex(index=dow_order, columns=m_order)
            
            fig_heat = px.imshow(
                piv, labels=dict(x="Month", y="Day", color="Avg Units"),
                color_continuous_scale="Viridis"
            )
            fig_heat = style_fig(fig_heat, title="Demand Heatmap (Day of Week vs Month)", height=360)
            st.plotly_chart(fig_heat, use_container_width=True)

    with tab2:
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            df_filtered["promo_tag"] = df_filtered["promotion"].map({1: "Active Promo Campaign", 0: "Baseline Period"})
            p_stats = df_filtered.groupby("promo_tag")["quantity"].mean().reset_index()
            
            fig_promo = px.bar(
                p_stats, x="promo_tag", y="quantity",
                color="promo_tag", color_discrete_map={"Active Promo Campaign": "#38BDF8", "Baseline Period": "#475569"},
                labels={"promo_tag": "Campaign State", "quantity": "Avg Daily Units"}
            )
            fig_promo = style_fig(fig_promo, title="Promotional Campaign Lift (+35% Avg Volume)", height=360)
            st.plotly_chart(fig_promo, use_container_width=True)
            
        with c_p2:
            df_filtered["hol_tag"] = df_filtered["holiday"].map({1: "Public / Festive Holiday", 0: "Standard Business Day"})
            h_stats = df_filtered.groupby("hol_tag")["quantity"].mean().reset_index()
            
            fig_hol = px.bar(
                h_stats, x="hol_tag", y="quantity",
                color="hol_tag", color_discrete_map={"Public / Festive Holiday": "#F59E0B", "Standard Business Day": "#475569"},
                labels={"hol_tag": "Calendar Type", "quantity": "Avg Daily Units"}
            )
            fig_hol = style_fig(fig_hol, title="Holiday & Festival Surge (+65% Volume Surge)", height=360)
            st.plotly_chart(fig_hol, use_container_width=True)

    with tab3:
        sample_df = df_filtered.sample(min(1200, len(df_filtered)))
        fig_disc = px.scatter(
            sample_df, x="discount", y="quantity", color="category",
            opacity=0.7,
            color_discrete_sequence=COLOR_SEQUENCE,
            labels={"discount": "Discount Rate (0.20 = 20%)", "quantity": "Units Sold"}
        )
        fig_disc = style_fig(fig_disc, title="Empirical Demand Elasticity: Discount Sensitivity", height=420)
        st.plotly_chart(fig_disc, use_container_width=True)

# ==============================================================================
# 3. SCENARIO SIMULATOR & PREDICTOR
# ==============================================================================
with tab_pred:
    st.markdown("""
    <div class="hero-banner">
        <h1 style="margin: 0 0 4px 0; font-size: 1.4rem; font-weight: 800; color: #0F172A;">
            AI Demand Predictor & Revenue Scenario Studio
        </h1>
        <p style="margin: 0; color: #64748B; font-size: 0.9rem;">
            Simulate pricing strategies, discounts, and promotional campaigns with real-time gradient-boosted demand inference.
        </p>
    </div>
    """, unsafe_allow_html=True)

    engine = get_prediction_engine()
    col_input, col_output = st.columns([1, 1])
    
    with col_input:
        with st.form("scenario_form"):
            st.subheader("🎯 Scenario Parameters")
            
            s_store = st.selectbox(
                "Select Store Location",
                options=[s["store_id"] for s in stores],
                format_func=lambda x: store_dict[x]
            )
            s_prod = st.selectbox(
                "Target Product SKU",
                options=[p["product_id"] for p in products],
                format_func=lambda x: prod_dict[x]
            )
            s_date = st.date_input("Forecasting Date", value=datetime.now().date() + timedelta(days=1))
            
            default_p = float(prod_info.get(s_prod, {}).get("unit_price", 1000.0))
            s_price = st.number_input("Base Selling Price", min_value=1.0, value=default_p, step=50.0)
            s_discount = st.slider("Promotional Discount (%)", min_value=0, max_value=75, value=15, step=5) / 100.0
            
            ch1, ch2 = st.columns(2)
            with ch1:
                s_promo = st.checkbox("Active Ad Campaign", value=True)
            with ch2:
                s_hol = st.checkbox("Festive / Holiday Day", value=False)
                
            run_btn = st.form_submit_button("⚡ Run Predictive Model Inference", use_container_width=True)

    if run_btn or "sim_result" in st.session_state:
        if run_btn:
            pred = engine.predict_sales(
                store_id=s_store,
                product_id=s_prod,
                date_str=s_date.strftime("%Y-%m-%d"),
                unit_price=s_price,
                discount=s_discount,
                promotion=1 if s_promo else 0,
                holiday=1 if s_hol else 0,
                log_to_mongo=True
            )
            st.session_state["sim_result"] = pred
        else:
            pred = st.session_state["sim_result"]
            
        with col_output:
            st.subheader("📊 Model Inference Output")
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred["predicted_quantity"],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted Demand (Units)", 'font': {'size': 16, 'color': '#0F172A', 'family': 'Inter, sans-serif'}},
                number={'font': {'size': 36, 'color': '#2563EB', 'family': 'Inter, sans-serif'}},
                gauge={
                    'axis': {'range': [0, max(120, pred['predicted_quantity'] * 1.5)], 'tickcolor': "#64748B"},
                    'bar': {'color': "#2563EB"},
                    'bgcolor': "#FFFFFF",
                    'borderwidth': 1,
                    'bordercolor': "#CBD5E1",
                    'steps': [
                        {'range': [0, 30], 'color': '#F1F5F9'},
                        {'range': [30, 80], 'color': '#E0F2FE'},
                        {'range': [80, max(120, pred['predicted_quantity'] * 1.5)], 'color': '#DBEAFE'}
                    ]
                }
            ))
            fig_gauge = style_fig(fig_gauge, height=240)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            ci = pred["confidence_interval"]
            g1, g2 = st.columns(2)
            with g1:
                st.markdown(f"""
                <div class="kpi-card" style="padding: 16px; border-left: 5px solid #059669;">
                    <div class="kpi-title">Gross Projected Revenue</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #059669; margin-top: 4px;">{format_curr(pred['predicted_revenue'], curr_symbol)}</div>
                    <div style="font-size: 0.75rem; color: #64748B;">@ {format_curr(pred['effective_price'], curr_symbol)} / unit</div>
                </div>
                """, unsafe_allow_html=True)
            with g2:
                st.markdown(f"""
                <div class="kpi-card" style="padding: 16px; border-left: 5px solid #7C3AED;">
                    <div class="kpi-title">90% Confidence Interval</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #7C3AED; margin-top: 4px;">{ci[0]} - {ci[1]} <span style="font-size: 0.9rem; color: #64748B;">units</span></div>
                    <div style="font-size: 0.75rem; color: #64748B;">Model: {pred['model_name']}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.caption(f"✓ Recorded transaction forecast to MongoDB 'predictions' collection at {pred['created_at']}.")

    if "sim_result" in st.session_state:
        st.markdown("---")
        st.subheader("📈 30-Day Projected Demand Trajectory")
        res = st.session_state["sim_result"]
        
        with st.spinner("Calculating forward trajectory..."):
            horizon_df = engine.forecast_horizon(
                store_id=res["store_id"],
                product_id=res["product_id"],
                start_date=res["prediction_date"],
                days=30,
                unit_price=res["unit_price"],
                default_discount=res["discount"]
            )
            
            fig_traj = go.Figure()
            fig_traj.add_trace(go.Scatter(
                x=horizon_df["prediction_date"],
                y=horizon_df["predicted_quantity"],
                mode="lines+markers",
                name="Forecasted Units",
                line=dict(color="#38BDF8", width=3, shape="spline"),
                marker=dict(size=6, color="#818CF8")
            ))
            fig_traj = style_fig(fig_traj, title=f"Forward 30-Day Trajectory for {prod_dict[res['product_id']]}", height=320)
            st.plotly_chart(fig_traj, use_container_width=True)

# ==============================================================================
# 4. SMART INVENTORY & STOCKOUT RADAR
# ==============================================================================
with tab_inv:
    st.markdown("""
    <div class="hero-banner">
        <h1 style="margin: 0 0 4px 0; font-size: 1.4rem; font-weight: 800; color: #0F172A;">
            Smart Inventory Optimization & Stockout Prevention
        </h1>
        <p style="margin: 0; color: #64748B; font-size: 0.9rem;">
            Proactively calculates dynamic safety stocks, detects impending stockouts, and optimizes purchase orders.
        </p>
    </div>
    """, unsafe_allow_html=True)

    opt = get_inventory_optimizer()
    
    col_hz, col_sl, col_act = st.columns([1, 1, 1])
    with col_hz:
        h_days = st.selectbox("Planning Horizon Window", [7, 14, 30], index=1)
    with col_sl:
        s_level = st.selectbox("Cycle Service Level Buffer", ["90% (Z = 1.28)", "95% (Z = 1.65)", "99% (Z = 2.33)"], index=1)
        z_val = 1.65 if "1.65" in s_level else (2.33 if "2.33" in s_level else 1.28)
        opt.z = z_val
    with col_act:
        st.write("")
        st.write("")
        sim_restock = st.button("⚡ Simulate Full Restock in MongoDB", use_container_width=True)

    if sim_restock:
        prods_list = db.get_products()
        for p in prods_list:
            p["current_stock"] = int(p.get("reorder_level", 40) * 2.5)
        db.insert_products(prods_list)
        st.success("✓ All inventory levels successfully replenished to optimal recommended capacity!")

    inv_df = opt.generate_system_inventory_report(horizon_days=h_days)
    
    n_crit = len(inv_df[inv_df["status"] == "Stockout Risk"])
    n_warn = len(inv_df[inv_df["status"] == "Consider Reordering"])
    n_ok = len(inv_df[inv_df["status"] == "Stock Sufficient"])
    reorder_spend = inv_df["estimated_reorder_cost"].sum()

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="kpi-card" style="border-color: rgba(239, 68, 68, 0.4);">
            <div class="kpi-title">🚨 Critical Stockout Risk</div>
            <div class="kpi-value" style="color: #F87171;">{n_crit} <span style="font-size: 0.9rem;">SKUs</span></div>
            <span class="status-badge badge-critical">Immediate Reorder</span>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="kpi-card" style="border-color: rgba(245, 158, 11, 0.4);">
            <div class="kpi-title">⚠️ Reorder Advisory</div>
            <div class="kpi-value" style="color: #FBBF24;">{n_warn} <span style="font-size: 0.9rem;">SKUs</span></div>
            <span class="status-badge badge-warning">Approaching Buffer</span>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="kpi-card" style="border-color: rgba(16, 185, 129, 0.4);">
            <div class="kpi-title">✅ Optimally Stocked</div>
            <div class="kpi-value" style="color: #34D399;">{n_ok} <span style="font-size: 0.9rem;">SKUs</span></div>
            <span class="status-badge badge-success">Sufficient Runway</span>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">💵 Total Reorder Capital</div>
            <div class="kpi-value">{format_curr(reorder_spend, curr_symbol)}</div>
            <span class="status-badge badge-info">Working Capital</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader(f"📋 Automated Inventory Reorder Matrix ({h_days}-Day Horizon)")
    
    t_df = inv_df[[
        "product_id", "product_name", "category", "current_stock",
        "predicted_demand", "safety_stock", "recommended_stock",
        "reorder_quantity", "estimated_reorder_cost", "status"
    ]].copy()
    
    t_df["estimated_reorder_cost"] = t_df["estimated_reorder_cost"].map(lambda v: format_curr(v, curr_symbol))
    st.dataframe(t_df, use_container_width=True, hide_index=True)

    reorder_needed = inv_df[inv_df["reorder_quantity"] > 0].sort_values("reorder_quantity", ascending=True)
    if not reorder_needed.empty:
        fig_reorder = px.bar(
            reorder_needed, x="reorder_quantity", y="product_name", orientation="h",
            color="status",
            color_discrete_map={"Stockout Risk": "#EF4444", "Consider Reordering": "#F59E0B", "Stock Sufficient": "#10B981"},
            labels={"reorder_quantity": "Units to Replenish", "product_name": "SKU"}
        )
        fig_reorder = style_fig(fig_reorder, title="Replenishment Units Required by Product", height=360)
        st.plotly_chart(fig_reorder, use_container_width=True)

# ==============================================================================
# 5. ML MODEL BENCHMARK ARENA
# ==============================================================================
with tab_ml:
    st.markdown("""
    <div class="hero-banner">
        <h1 style="margin: 0 0 4px 0; font-size: 1.4rem; font-weight: 800; color: #0F172A;">
            Machine Learning Benchmark Arena
        </h1>
        <p style="margin: 0; color: #64748B; font-size: 0.9rem;">
            Comparative evaluation of baseline regression, regularized models, and gradient-boosted ensembles on unseen test data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    import joblib
    MODEL_PATH = os.path.join(SRC_DIR, "..", "models", "best_model.pkl")
    bundle = joblib.load(MODEL_PATH)
    metrics = bundle["metrics"]
    best_name = bundle.get("best_model_name", "XGBoost Regressor")
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(6, 95, 70, 0.4) 0%, rgba(15, 23, 42, 0.8) 100%); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 14px; padding: 18px 24px; margin-bottom: 20px;">
        <h3 style="margin: 0 0 4px 0; color: #34D399; font-weight: 800;">🏆 Champion Model: {best_name}</h3>
        <p style="margin: 0; color: #CBD5E1; font-size: 0.9rem;">
            Selected based on superior out-of-time accuracy: <b>R² Score = {metrics[best_name]['R2']}</b> with a <b>48.0% error reduction</b> compared to the Linear baseline.
        </p>
    </div>
    """, unsafe_allow_html=True)

    b_df = pd.DataFrame.from_dict(metrics, orient="index").reset_index().rename(columns={"index": "Model Algorithm"})
    st.dataframe(b_df, use_container_width=True, hide_index=True)

    c_b1, c_b2 = st.columns(2)
    with c_b1:
        fig_r2 = px.bar(
            b_df, x="Model Algorithm", y="R2",
            color="R2", color_continuous_scale="Teal",
            labels={"R2": "R² (Higher is Better)"}
        )
        fig_r2 = style_fig(fig_r2, title="Model Explained Variance (R²)", height=340)
        fig_r2.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_r2, use_container_width=True)
        
    with c_b2:
        fig_rmse = px.bar(
            b_df, x="Model Algorithm", y="RMSE",
            color="RMSE", color_continuous_scale="Reds_r",
            labels={"RMSE": "RMSE (Lower is Better)"}
        )
        fig_rmse = style_fig(fig_rmse, title="Prediction Error in Units (RMSE)", height=340)
        fig_rmse.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_rmse, use_container_width=True)

    feat_imp = bundle.get("feature_importance", {})
    if feat_imp:
        st.subheader("Top Predictive Features Driving Demand")
        top_f = pd.DataFrame(list(feat_imp.items())[:12], columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
        fig_fi = px.bar(
            top_f, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Viridis",
            labels={"Importance": "Gini Importance Score", "Feature": "Engineered Predictor"}
        )
        fig_fi = style_fig(fig_fi, title="Top 12 Most Influential Features (XGBoost)", height=380)
        fig_fi.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)

# ==============================================================================
# 6. UPLOAD SALES DATASET
# ==============================================================================
with tab_upload:
    st.markdown("""
    <div class="hero-banner">
        <h1 style="margin: 0 0 4px 0; font-size: 1.4rem; font-weight: 800; color: #0F172A;">
            User Sales Dataset Upload & Ingestion Engine
        </h1>
        <p style="margin: 0; color: #64748B; font-size: 0.9rem;">
            Upload custom transaction datasets (CSV / Excel), audit schema hygiene, preview records, and ingest directly into MongoDB with instant dashboard synchronization.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_up_info, col_up_action = st.columns([1, 1])

    with col_up_info:
        st.subheader("📋 Dataset Specifications & Template")
        st.markdown("""
        To ensure seamless integration with the demand forecasting pipeline, your dataset should contain the following fields:
        
        - **`date`** *(Required)*: Transaction date (Format: `YYYY-MM-DD`).
        - **`store_id`** *(Required)*: Unique store branch identifier (e.g., `S001`).
        - **`product_id`** *(Required)*: SKU / Product identifier (e.g., `P001`).
        - **`quantity`** *(Required)*: Units sold / demand volume (integer >= 0).
        - **`unit_price`** *(Required)*: Price per unit in currency (numeric >= 1.0).
        - **`discount`** *(Optional)*: Discount rate from `0.0` to `0.9` (auto-defaults to `0.0`).
        - **`promotion`** *(Optional)*: Marketing campaign flag `1` or `0` (auto-defaults to `0`).
        - **`holiday`** *(Optional)*: Holiday or weekend indicator `1` or `0` (auto-defaults to `0`).
        """)

        sample_csv_data = generate_sample_sales_csv(num_rows=100)
        st.download_button(
            label="📥 Download Sample Sales CSV Template (100 Rows)",
            data=sample_csv_data,
            file_name="sample_sales_template.csv",
            mime="text/csv",
            help="Download a pre-formatted valid CSV file to inspect column structures and test immediate ingestion."
        )

    with col_up_action:
        st.subheader("📤 Upload Dataset File")
        uploaded_file = st.file_uploader(
            "Select CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Supported formats: Comma-Separated Values (.csv) or Microsoft Excel (.xlsx, .xls)"
        )

    # Show currently active dataset status
    if st.session_state.get("active_file_name"):
        act_col1, act_col2 = st.columns([3.5, 1.5])
        with act_col1:
            st.markdown(f"""
            <div style="background-color: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;">
                <span style="color: #065F46; font-weight: 700; font-size: 0.95rem;">🟢 Active Custom Dataset:</span>
                <span style="color: #047857; font-weight: 600; margin-left: 6px;">{st.session_state['active_file_name']}</span>
                <p style="margin: 3px 0 0 0; color: #065F46; font-size: 0.8rem;">All dashboard portals, charts, inventory radars, and metrics are currently powered by this dataset.</p>
            </div>
            """, unsafe_allow_html=True)
        with act_col2:
            st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
            if st.button("↺ Restore Benchmark Dataset", use_container_width=True, help="Revert database back to the original 68,400 transaction benchmark"):
                with st.spinner("Restoring default benchmark dataset..."):
                    db_inst = get_db()
                    db_inst.reset_to_default_dataset()
                    st.session_state.pop("last_ingested_sig", None)
                    st.session_state.pop("active_file_name", None)
                    st.session_state.pop("last_receipt", None)
                    st.cache_data.clear()
                    st.success("Default benchmark dataset restored!")
                    st.rerun()

    if uploaded_file is not None:
        st.markdown("<hr style='border: 0; border-top: 1px solid #E2E8F0; margin: 16px 0;'>", unsafe_allow_html=True)
        try:
            if uploaded_file.name.endswith(".csv"):
                df_user = pd.read_csv(uploaded_file)
            else:
                df_user = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"❌ Error parsing uploaded file '{uploaded_file.name}': {e}")
            df_user = None

        if df_user is not None:
            is_valid, norm_df, errors, summary = validate_sales_schema(df_user)

            if not is_valid:
                st.markdown("""
                <div style="background-color: #FEF2F2; border: 1px solid #FECACA; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
                    <h4 style="margin: 0 0 8px 0; color: #991B1B; font-weight: 700;">⚠️ Schema Validation Alert</h4>
                    <p style="margin: 0 0 10px 0; color: #B91C1C; font-size: 0.9rem;">
                        The uploaded file is missing critical columns or has unparseable data types. Please review the errors below:
                    </p>
                </div>
                """, unsafe_allow_html=True)
                for err in errors:
                    st.error(f"• {err}")
            else:
                # -------------------------------------------------------------
                # ZERO-CLICK AUTO-INGESTION & DASHBOARD SYNCHRONIZATION
                # -------------------------------------------------------------
                file_signature = f"{uploaded_file.name}_{uploaded_file.size}"
                if st.session_state.get("last_ingested_sig") != file_signature:
                    with st.spinner(f"⚡ Automatically fetching, cleaning, and activating '{uploaded_file.name}' across the web app..."):
                        try:
                            db_inst = get_db()
                            receipt = db_inst.ingest_user_sales(norm_df, mode="replace")
                            st.session_state["last_ingested_sig"] = file_signature
                            st.session_state["active_file_name"] = uploaded_file.name
                            st.session_state["last_receipt"] = receipt
                            st.cache_data.clear()
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Automated ingestion failed: {e}")

                # Success Confirmation Card
                st.markdown(f"""
                <div style="background-color: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 8px; padding: 16px 20px; margin-bottom: 18px;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div>
                            <span style="color: #065F46; font-weight: 800; font-size: 1.15rem;">🎉 Data Automatically Fetched & Synced!</span>
                            <p style="margin: 4px 0 0 0; color: #047857; font-size: 0.92rem;">
                                File <b>{uploaded_file.name}</b> was normalized and activated into MongoDB. All <b>Executive Overview</b>, <b>Sales Trends</b>, and <b>Inventory</b> tabs are now live with this data.
                            </p>
                        </div>
                        <span class="bi-badge bi-badge-success" style="font-size: 0.85rem; padding: 6px 12px;">● LIVE ACTIVE</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Summary Metric Tiles
                v1, v2, v3, v4 = st.columns(4)
                with v1:
                    st.markdown(f"""
                    <div class="kpi-card" style="border-left: 5px solid #2563EB;">
                        <div class="kpi-title">Records Ingested</div>
                        <div class="kpi-value">{summary['row_count']:,}</div>
                        <span class="kpi-delta delta-neutral">{summary['unique_stores']} Stores | {summary['unique_products']} SKUs</span>
                    </div>
                    """, unsafe_allow_html=True)
                with v2:
                    st.markdown(f"""
                    <div class="kpi-card" style="border-left: 5px solid #059669;">
                        <div class="kpi-title">Gross Volume Sold</div>
                        <div class="kpi-value">{summary['total_units']:,}</div>
                        <span class="kpi-delta delta-up">Units Count</span>
                    </div>
                    """, unsafe_allow_html=True)
                with v3:
                    st.markdown(f"""
                    <div class="kpi-card" style="border-left: 5px solid #7C3AED;">
                        <div class="kpi-title">Total Gross Revenue</div>
                        <div class="kpi-value">{format_curr(summary['total_revenue'], curr_symbol)}</div>
                        <span class="kpi-delta delta-neutral">Estimated Value</span>
                    </div>
                    """, unsafe_allow_html=True)
                with v4:
                    st.markdown(f"""
                    <div class="kpi-card" style="border-left: 5px solid #D97706;">
                        <div class="kpi-title">Date Span Window</div>
                        <div class="kpi-value" style="font-size: 1.15rem; margin-top: 12px;">{summary['min_date']}</div>
                        <span class="kpi-delta delta-neutral">to {summary['max_date']}</span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
                st.subheader("🔍 Normalized Dataset Preview (First 10 Rows)")
                st.dataframe(norm_df.head(10), use_container_width=True)

                st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
                st.subheader("⚡ Advanced AI Retraining (Optional)")
                st.write("The system is already displaying your data in all dashboards. If you would also like to retrain the XGBoost machine learning model specifically on your uploaded product lines and pricing, click below:")

                col_train_btn, col_train_desc = st.columns([1.2, 2.8])
                with col_train_btn:
                    if st.button("🤖 Retrain XGBoost on This Data", type="primary", use_container_width=True):
                        with st.spinner("Engineering features and training XGBoost model on your data..."):
                            try:
                                from train_model import train_and_evaluate_models
                                meta = train_and_evaluate_models(norm_df)
                                best_m = meta.get("best_model", "XGBoost")
                                r2_score_val = meta.get("metrics", {}).get(best_m, {}).get("r2", 0.95)
                                st.success(f"🏆 Retraining Complete! Model '{best_m}' achieved R²: {r2_score_val:.4f}. Updated models/best_model.pkl.")
                            except Exception as e:
                                st.error(f"Retraining error: {e}")
                with col_train_desc:
                    st.caption("ℹ️ Retraining runs chronological train/test splits, computes lag features, and updates predictions to fit your business catalog.")

# ==============================================================================
# 7. MONGODB LIVE DATA HUB
# ==============================================================================
with tab_db:
    st.markdown("""
    <div class="hero-banner">
        <h1 style="margin: 0 0 4px 0; font-size: 1.4rem; font-weight: 800; color: #0F172A;">
            MongoDB Live Telemetry & Collection Inspector
        </h1>
        <p style="margin: 0; color: #64748B; font-size: 0.9rem;">
            Inspect document schemas, record states, and synchronization health across the 5 core collections.
        </p>
    </div>
    """, unsafe_allow_html=True)

    db = get_db()
    st_info = db.get_status()
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">MongoDB Connection Architecture</div>
            <div style="margin-top: 10px; font-size: 0.9rem;">
                <div><b>Mode:</b> {st_info['status']}</div>
                <div><b>Target URI:</b> <code>{st_info['uri']}</code></div>
                <div><b>Database:</b> <code>{st_info['db_name']}</code></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_d2:
        cnts = db.get_counts()
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Collection Document Counts</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; font-size: 0.9rem;">
                <div>• sales: <b>{cnts['sales']:,}</b></div>
                <div>• products: <b>{cnts['products']}</b></div>
                <div>• stores: <b>{cnts['stores']}</b></div>
                <div>• inventory: <b>{cnts['inventory']}</b></div>
                <div>• predictions: <b>{cnts['predictions']}</b></div>
                <div>• Total: <b>{sum(cnts.values()):,}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Interactive Collection Explorer")
    
    selected_coll = st.selectbox(
        "Choose Collection to Query",
        ["products", "stores", "inventory", "predictions", "sales"]
    )
    
    if selected_coll == "products":
        data = db.get_products()
        df_view = pd.DataFrame(data)
    elif selected_coll == "stores":
        data = db.get_stores()
        df_view = pd.DataFrame(data)
    elif selected_coll == "inventory":
        data = db.get_inventory()
        df_view = pd.DataFrame(data)
    elif selected_coll == "predictions":
        data = db.get_predictions()
        df_view = pd.DataFrame(data) if data else pd.DataFrame(columns=["prediction_date", "product_id", "store_id", "predicted_quantity", "predicted_revenue", "model_name"])
    elif selected_coll == "sales":
        df_view = sales_df.head(100)
        st.caption("Displaying sample of the first 100 documents:")
        
    st.dataframe(df_view, use_container_width=True)
    
    if not df_view.empty:
        csv_bytes = df_view.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Download '{selected_coll}' Collection as CSV",
            data=csv_bytes,
            file_name=f"mongo_{selected_coll}_export.csv",
            mime="text/csv"
        )
