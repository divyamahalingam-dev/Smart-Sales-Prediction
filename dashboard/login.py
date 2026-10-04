"""
Authentication and Login Portal for Retail Demand Intelligence Hub.
Provides enterprise authentication, password verification, 1-click demo access,
account registration, and authenticated session management.
"""

from datetime import datetime
from typing import Dict, Any, Optional
import streamlit as st
from database import get_db

DEMO_ACCOUNTS = [
    {
        "role_badge": "👑 Administrator",
        "username": "admin",
        "password": "admin123",
        "full_name": "System Administrator",
        "role": "Administrator",
        "desc": "Full enterprise privileges & database hub"
    },
    {
        "role_badge": "💼 Operations Manager",
        "username": "manager",
        "password": "sales123",
        "full_name": "Operations Lead",
        "role": "Store Operations Manager",
        "desc": "Inventory replenishment & store analytics"
    },
    {
        "role_badge": "📈 Demand Analyst",
        "username": "analyst",
        "password": "analyst123",
        "full_name": "Demand Forecaster",
        "role": "Demand Analyst",
        "desc": "ML benchmarking & scenario simulation"
    }
]


def login_user(user: Dict[str, Any]):
    """Sets session state for authenticated user."""
    st.session_state["authenticated"] = True
    st.session_state["user"] = user
    st.session_state["username"] = user.get("username", "user")
    st.session_state["full_name"] = user.get("full_name", user.get("username", "User"))
    st.session_state["user_role"] = user.get("role", "Analyst")
    st.session_state["login_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def logout_user():
    """Clears authentication session state."""
    st.session_state["authenticated"] = False
    st.session_state.pop("user", None)
    st.session_state.pop("username", None)
    st.session_state.pop("full_name", None)
    st.session_state.pop("user_role", None)
    st.session_state.pop("login_time", None)
    st.rerun()


def render_login_page() -> bool:
    """
    Renders the modern corporate login portal.
    Returns True if user is authenticated, False otherwise.
    """
    if st.session_state.get("authenticated", False):
        return True

    db = get_db()

    # Outer container for centering
    _, col_center, _ = st.columns([1, 1.8, 1])

    with col_center:
        # Branding Header
        st.markdown("""
        <div style="text-align: center; padding: 24px 0 16px 0;">
            <div style="font-size: 3.2rem; line-height: 1; margin-bottom: 10px;">📊</div>
            <h1 style="font-size: 1.85rem; font-weight: 800; color: #0F172A; margin: 0 0 6px 0; letter-spacing: -0.02em;">
                Retail Demand Intelligence Hub
            </h1>
            <p style="font-size: 0.92rem; color: #64748B; margin: 0 0 14px 0;">
                Smart Sales Prediction, Demand Forecasting & Inventory Optimization
            </p>
            <div style="display: inline-block; background-color: #EFF6FF; border: 1px solid #BFDBFE; color: #1D4ED8; font-weight: 700; font-size: 0.8rem; padding: 4px 14px; border-radius: 9999px;">
                🔒 Secure Enterprise Authentication
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_register, tab_credentials = st.tabs(["🔑 Sign In", "📝 Create Account", "ℹ️ Demo Credentials"])

        # ----------------- TAB 1: SIGN IN -----------------
        with tab_login:
            st.markdown("<p style='font-size: 0.85rem; color: #64748B; margin: 8px 0 14px 0;'>Sign in with your enterprise credentials to access analytics and forecasting models.</p>", unsafe_allow_html=True)

            with st.form("login_form", clear_on_submit=False):
                username_input = st.text_input("Username", placeholder="e.g. admin, manager, or analyst")
                password_input = st.text_input("Password", type="password", placeholder="Enter your password")
                submitted = st.form_submit_button("🚀 Sign In to Intelligence Hub", use_container_width=True)

                if submitted:
                    if not username_input or not password_input:
                        st.error("Please enter both username and password.")
                    else:
                        user = db.authenticate_user(username_input, password_input)
                        if user:
                            login_user(user)
                            st.success(f"Welcome back, {user.get('full_name')}!")
                            st.rerun()
                        else:
                            st.error("Invalid credentials. Please verify username and password or use the 1-Click Demo buttons below.")

            # Quick 1-Click Demo Login Section
            st.markdown("""
            <div style="margin-top: 24px; padding-top: 18px; border-top: 1px solid #E2E8F0;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #334155; margin-bottom: 10px; display: flex; align-items: center; gap: 6px;">
                    <span>⚡</span> <span>1-Click Instant Demo Access (No typing needed)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                if st.button("👑 Admin\n(Executive)", use_container_width=True, key="demo_admin"):
                    user = db.authenticate_user("admin", "admin123")
                    if user:
                        login_user(user)
                        st.rerun()
            with col_d2:
                if st.button("💼 Manager\n(Operations)", use_container_width=True, key="demo_manager"):
                    user = db.authenticate_user("manager", "sales123")
                    if user:
                        login_user(user)
                        st.rerun()
            with col_d3:
                if st.button("📈 Analyst\n(Forecasting)", use_container_width=True, key="demo_analyst"):
                    user = db.authenticate_user("analyst", "analyst123")
                    if user:
                        login_user(user)
                        st.rerun()

        # ----------------- TAB 2: REGISTER -----------------
        with tab_register:
            st.markdown("<p style='font-size: 0.85rem; color: #64748B; margin: 8px 0 14px 0;'>Register a new enterprise user. Credentials will be persisted in MongoDB.</p>", unsafe_allow_html=True)

            with st.form("register_form", clear_on_submit=True):
                reg_name = st.text_input("Full Name", placeholder="e.g. Divya Mahalingam")
                reg_username = st.text_input("Desired Username", placeholder="e.g. divya_m")
                reg_email = st.text_input("Email Address (Optional)", placeholder="divya@company.com")
                reg_password = st.text_input("Password", type="password", placeholder="Minimum 4 characters")
                reg_role = st.selectbox(
                    "Organizational Role",
                    ["Demand Analyst", "Store Operations Manager", "Administrator", "Lead Data Scientist", "Business Executive"]
                )
                reg_submit = st.form_submit_button("✨ Register & Launch Session", use_container_width=True)

                if reg_submit:
                    success, msg = db.register_user(
                        username=reg_username,
                        password=reg_password,
                        full_name=reg_name,
                        role=reg_role,
                        email=reg_email
                    )
                    if success:
                        new_user = db.authenticate_user(reg_username, reg_password)
                        if new_user:
                            login_user(new_user)
                            st.success(f"Account registered! Welcome, {new_user.get('full_name')}!")
                            st.rerun()
                        else:
                            st.success(msg)
                    else:
                        st.error(msg)

        # ----------------- TAB 3: CREDENTIALS REFERENCE -----------------
        with tab_credentials:
            st.markdown("<p style='font-size: 0.85rem; color: #64748B; margin: 8px 0 12px 0;'>Pre-configured accounts available for demonstration and evaluation:</p>", unsafe_allow_html=True)
            for acc in DEMO_ACCOUNTS:
                st.markdown(f"""
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px 14px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                        <span style="font-weight: 700; color: #0F172A; font-size: 0.88rem;">{acc['role_badge']}</span>
                        <span style="font-size: 0.75rem; color: #64748B;">{acc['full_name']}</span>
                    </div>
                    <div style="font-size: 0.82rem; color: #334155; font-family: monospace;">
                        Username: <strong>{acc['username']}</strong> &nbsp;|&nbsp; Password: <strong>{acc['password']}</strong>
                    </div>
                    <div style="font-size: 0.76rem; color: #64748B; margin-top: 4px;">{acc['desc']}</div>
                </div>
                """, unsafe_allow_html=True)

        # Security footer
        st.markdown("""
        <div style="text-align: center; margin-top: 26px; padding-top: 16px; border-top: 1px solid #E2E8F0; font-size: 0.76rem; color: #94A3B8;">
            🛡️ Salted SHA-256 Authentication &nbsp;•&nbsp; NoSQL MongoDB Persistence &nbsp;•&nbsp; Role-Based Authorization
        </div>
        """, unsafe_allow_html=True)

    return False


def render_user_profile_header():
    """Renders user profile badge and logout button in top navigation bar."""
    full_name = st.session_state.get("full_name", "User")
    role = st.session_state.get("user_role", "Analyst")

    c_prof, c_logout = st.columns([2.5, 1.2])
    with c_prof:
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 6px 12px; border-radius: 8px; display: flex; align-items: center; gap: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
            <div style="background: #EFF6FF; border: 1px solid #DBEAFE; color: #2563EB; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.85rem;">
                {full_name[:1].upper()}
            </div>
            <div style="line-height: 1.2;">
                <div style="font-size: 0.82rem; font-weight: 700; color: #0F172A; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 140px;">
                    {full_name}
                </div>
                <div style="font-size: 0.72rem; color: #2563EB; font-weight: 600;">
                    {role}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_logout:
        if st.button("🚪 Logout", key="logout_btn_header", help="End session and return to login page", use_container_width=True):
            logout_user()
