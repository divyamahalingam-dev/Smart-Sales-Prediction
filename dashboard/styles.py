import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

CUSTOM_LIGHT_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Clean Corporate Light Background */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }

    /* Top Corporate Navigation Header */
    .bi-header-bar {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 22px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .bi-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: #0F172A;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .bi-subtitle {
        font-size: 0.85rem;
        color: #64748B;
        margin: 2px 0 0 0;
    }

    /* Structured PowerBI/Tableau KPI Tiles */
    .bi-kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        position: relative;
        transition: box-shadow 0.2s ease, transform 0.2s ease;
    }
    .bi-kpi-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.07);
        transform: translateY(-2px);
    }
    .bi-kpi-blue { border-left: 5px solid #2563EB; }
    .bi-kpi-emerald { border-left: 5px solid #059669; }
    .bi-kpi-violet { border-left: 5px solid #7C3AED; }
    .bi-kpi-amber { border-left: 5px solid #D97706; }

    .bi-kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
    }
    .bi-kpi-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        margin: 6px 0 2px 0;
        letter-spacing: -0.02em;
    }
    .bi-kpi-sub {
        font-size: 0.78rem;
        font-weight: 600;
        color: #059669;
    }

    /* Status Badges */
    .bi-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        text-align: center;
    }
    .bi-badge-critical {
        background-color: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FCA5A5;
    }
    .bi-badge-warning {
        background-color: #FEF3C7;
        color: #B45309;
        border: 1px solid #FCD34D;
    }
    .bi-badge-success {
        background-color: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
    }
    .bi-badge-info {
        background-color: #E0F2FE;
        color: #0369A1;
        border: 1px solid #7DD3FC;
    }

    /* Clean White Containers */
    div[data-testid="stForm"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
        padding: 22px !important;
    }
    div[data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
    }

    /* Clean Tab Bar Styling */
    div[data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #FFFFFF;
        padding: 6px 8px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        margin-bottom: 18px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    }
    div[data-baseweb="tab"] {
        height: 42px;
        border-radius: 6px;
        font-weight: 600;
        color: #475569;
        padding: 0 16px;
        background-color: transparent;
    }
    div[aria-selected="true"] {
        background-color: #EFF6FF !important;
        color: #1D4ED8 !important;
        border-bottom: 2px solid #2563EB !important;
    }

    /* Clean White DataFrames */
    div[data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        overflow: hidden;
    }

    /* Corporate Section Header / Hero Banner */
    .hero-banner {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .hero-banner h1 {
        color: #0F172A !important;
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        margin: 0 0 6px 0 !important;
    }
    .hero-banner p {
        color: #64748B !important;
        font-size: 0.9rem !important;
        margin: 0 !important;
    }

    /* Unified Corporate KPI Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        border-left: 5px solid #2563EB;
        transition: box-shadow 0.2s ease, transform 0.2s ease;
    }
    .kpi-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.07);
        transform: translateY(-2px);
    }
    .kpi-title {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        margin: 6px 0 2px 0;
        letter-spacing: -0.02em;
    }
    .kpi-delta {
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
    }
    .delta-up {
        color: #059669;
        background-color: #ECFDF5;
        border: 1px solid #A7F3D0;
    }
    .delta-down {
        color: #DC2626;
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
    }
    .delta-neutral {
        color: #475569;
        background-color: #F1F5F9;
        border: 1px solid #E2E8F0;
    }
</style>
"""

PLOTLY_TEMPLATE = "plotly_white"
PLOTLY_BG = "#FFFFFF"
PLOTLY_PLOT_BG = "#FAFAFA"
COLOR_SEQUENCE = ["#2563EB", "#0284C7", "#059669", "#D97706", "#7C3AED", "#DC2626"]

def apply_custom_styles():
    st.markdown(CUSTOM_LIGHT_CSS, unsafe_allow_html=True)

def style_fig(fig, title=None, height=360):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=PLOTLY_BG,
        plot_bgcolor=PLOTLY_PLOT_BG,
        font=dict(family="Inter, Segoe UI, sans-serif", color="#334155"),
        title=dict(text=title, font=dict(color="#0F172A", size=15, weight=700)) if title else None,
        margin=dict(l=20, r=20, t=45 if title else 20, b=20),
        height=height,
        xaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0"),
        yaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0"),
        hoverlabel=dict(bgcolor="#FFFFFF", font_size=12, font_family="Inter, sans-serif", bordercolor="#CBD5E1")
    )
    return fig

def format_curr(val, curr_symbol="₹"):
    if curr_symbol == "₹":
        if val >= 10_000_000:
            return f"₹{val/10_000_000:.2f} Cr"
        elif val >= 100_000:
            return f"₹{val/100_000:.2f} Lakh"
        else:
            return f"₹{val:,.2f}"
    else:
        if val >= 1_000_000:
            return f"${val/1_000_000:.2f}M"
        elif val >= 1_000:
            return f"${val/1_000:.2f}K"
        else:
            return f"${val:,.2f}"
