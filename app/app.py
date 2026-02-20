import streamlit as st
from datetime import datetime

# ---------------- MODULE IMPORTS ----------------
# Make sure all these modules exist inside app/modules/ and each has a run() function
try:
    from modules import (
        dashboard,
        analytics_dashboard,
        traffic_prediction,
        air_quality,
        accident_detection,
        pothole_detection,
        crowd_monitoring,
        citizen_complaints,
        llm_chatbot  # ✅ this must exist as modules/llm_chatbot.py
    )
except ImportError as e:
    st.error(f"Module import error: {e}")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="UrbanBot Intelligence - Smart City AI Platform",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- HEADER ----------------
st.markdown(
    "<h1 style='text-align:center; color:#4B9CD3;'>🏙️ UrbanBot Intelligence</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align:center;'>Smart City Analytics Platform for Traffic, Infrastructure, Crowd & Air Quality Monitoring</h4>",
    unsafe_allow_html=True
)
st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🔍 Smart City Modules")

menu = st.sidebar.radio(
    "Select Module",
    [
        "🏠 Home Dashboard",
        "📊 City Analytics",
        "🚦 Traffic Prediction",
        "🌫 AQI Forecasting",
        "🚑 Accident Detection",
        "🕳 Pothole Detection",
        "👥 Crowd Monitoring",
        "🗣 Citizen Complaints",
        "🤖 AI Chatbot"
    ]
)

st.sidebar.divider()
st.sidebar.info("UrbanBot AI System\nYOLO • LSTM • NLP • LLM")

# ---------------- ROUTING ----------------
if menu == "🏠 Home Dashboard":
    dashboard.run()

elif menu == "📊 City Analytics":
    analytics_dashboard.run()

elif menu == "🚦 Traffic Prediction":
    traffic_prediction.run()

elif menu == "🌫 AQI Forecasting":
    air_quality.run()

elif menu == "🚑 Accident Detection":
    accident_detection.run()

elif menu == "🕳 Pothole Detection":
    pothole_detection.run()

elif menu == "👥 Crowd Monitoring":
    crowd_monitoring.run()

elif menu == "🗣 Citizen Complaints":
    citizen_complaints.run()

elif menu == "🤖 AI Chatbot":
    llm_chatbot.run()

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    f"© {datetime.now().year} UrbanBot Intelligence | Smart City AI Platform"
)
