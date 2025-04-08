import streamlit as st
import os
import pandas as pd
from datetime import datetime
from auth import create_users_table, add_user, login_user
import db
import ui
import config
import detection as detection
import analytics
import tabs as tabs

# Initialize database
db.init_db()

# --- App Configuration ---
ui.setup_ui()

# --- Database Setup ---
create_users_table()

# --- Session State ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

if "analytics" not in st.session_state:
    st.session_state.analytics = config.default_analytics()

# Initialize collision area
if "collision_area" not in st.session_state:
    st.session_state.collision_area = config.default_collision_area()

# --- LOGIN / SIGNUP SCREEN ---
if not st.session_state.logged_in:
    login_button, user, passwd, signup_button, new_user, new_pass = ui.render_login_screen()

    if login_button:
        if login_user(user, passwd):
            st.success("✅ Logged in successfully!")
            st.session_state.logged_in = True
            st.session_state.username = user
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

    if signup_button:
        if add_user(new_user, new_pass):
            st.success("🎉 Account created! You can now log in.")
        else:
            st.warning("⚠️ Username already exists.")

    st.stop()

# --- MAIN APP ---
# --- Sidebar ---
logout_button, max_people = ui.render_sidebar(st.session_state.username)

if logout_button:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.success("Logged out successfully.")
    st.rerun()

# --- Main Content ---
st.title("🚶 Pedestrian Detection Dashboard")
st.markdown(
    f"<p style='color: #6b7280; margin-bottom: 2rem;'>Welcome back, <strong>{st.session_state.username}</strong>! Monitor pedestrian activity in real-time.</p>",
    unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🟢 Live Detection",
    "📸 Past Detections",
    "🔐 Admin Panel",
    "📊 Analytics",
    "🗂️ Past Sessions",
    "ℹ️ About",
    "❓ FAQ"
])

# --- TAB 1: Live Detection ---
with tab1:
    tabs.render_live_detection_tab(tab1, max_people)

# --- TAB 2: Past Detections ---
with tab2:
    tabs.render_past_detections_tab(tab2)

# --- TAB 3: Admin Panel ---
with tab3:
    tabs.render_admin_panel_tab(tab3)

# --- TAB 4: Analytics ---
with tab4:
    tabs.render_analytics_tab(tab4)

# --- TAB 5: Past Sessions ---
with tab5:
    tabs.render_past_sessions_tab(tab5)

# --- TAB 6: About ---
with tab6:
    ui.render_about_tab()

# --- TAB 7: FAQ ---
with tab7:
    ui.render_faq_tab()