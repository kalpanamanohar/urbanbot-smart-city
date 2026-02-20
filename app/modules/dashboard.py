import streamlit as st
import pandas as pd
from sqlalchemy import text
from services.db import engine
import os

# ---------------- UTILITY ----------------
def fetch_count(table_name):
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            return result.scalar()
    except:
        return 0

# ---------------- DASHBOARD ----------------
def run():

    st.title("📊 Smart City AI Command Center")

    # ---------- Banner Image ----------
    banner_path = "images/dashboard_banner.jpg"
    if os.path.exists(banner_path):
        st.image(banner_path, use_container_width=True)

    st.divider()

    # -------- LIVE METRICS --------
    col1, col2, col3, col4, col5 = st.columns(5)

    accidents = fetch_count("accident_events")
    traffic = fetch_count("traffic_events")
    aqi = fetch_count("aqi_events")
    complaints = fetch_count("citizen_complaints")
    potholes = fetch_count("pothole_events")

    col1.metric("🚑 Accidents", accidents)
    col2.metric("🚦 Traffic Logs", traffic)
    col3.metric("🌫 AQI Logs", aqi)
    col4.metric("🗣 Complaints", complaints)
    col5.metric("🕳 Potholes", potholes)

    st.divider()

    # -------- RECENT EVENTS --------
    st.subheader("📜 Recent Activity")

    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                text("""
                SELECT 'Accident' AS module, city, event_time 
                FROM accident_events
                UNION ALL
                SELECT 'Traffic', city, event_time 
                FROM traffic_events
                UNION ALL
                SELECT 'AQI', city, event_time 
                FROM aqi_events
                UNION ALL
                SELECT 'Complaint', city, event_time 
                FROM citizen_complaints
                UNION ALL
                SELECT 'Pothole', city, event_time 
                FROM pothole_events
                ORDER BY event_time DESC
                LIMIT 10
                """),
                conn
            )

        st.dataframe(df, use_container_width=True)

    except:
        st.info("No recent activity available.")

    st.divider()
    st.caption("UrbanBot Intelligence • Real-Time Smart City Monitoring")
