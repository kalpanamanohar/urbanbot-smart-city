"""
Analytics Dashboard Module
---------------------------
Streamlit dashboard for Smart City analytics overview.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from services.db import execute_query


# ----------------------------------
# SAFE COUNT FETCHER
# ----------------------------------

def fetch_total(table_name):
    result = execute_query(f"SELECT COUNT(*) AS total FROM {table_name}")
    if isinstance(result, dict) and "error" in result:
        return 0
    return result[0]["total"] if result else 0


def fetch_today_count(table_name):
    result = execute_query(
        f"""
        SELECT COUNT(*) AS total
        FROM {table_name}
        WHERE DATE(event_time) = CURDATE()
        """
    )
    if isinstance(result, dict) and "error" in result:
        return 0
    return result[0]["total"] if result else 0


# ----------------------------------
# MAIN DASHBOARD
# ----------------------------------

def run():

    st.subheader("📊 Smart City Analytics Command Center")
    st.caption("Real-time operational insights across all AI modules")

    tables = {
        "🚑 Accidents": "accident_events",
        "🕳 Potholes": "pothole_events",
        "👥 Crowd": "crowd_events",
        "🗣 Complaints": "citizen_complaints",
        "🚦 Traffic": "traffic_events",
        "🌫 AQI Logs": "aqi_events"
    }

    cols = st.columns(6)

    totals = []
    for idx, (label, table) in enumerate(tables.items()):
        total = fetch_total(table)
        today = fetch_today_count(table)
        cols[idx].metric(label, total, delta=f"{today} today")
        totals.append(total)

    st.divider()

    # -------- BAR CHART --------
    st.subheader("📈 City Event Distribution")

    summary_df = pd.DataFrame({
        "Module": list(tables.keys()),
        "Events": totals
    })

    fig_bar = px.bar(
        summary_df,
        x="Module",
        y="Events",
        color="Module",
        text="Events",
        title="City-wide AI Event Distribution"
    )

    fig_bar.update_layout(
        title_x=0.5,
        xaxis_title="AI Modules",
        yaxis_title="Total Events",
        height=450,
        showlegend=False
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # -------- PIE CHART --------
    st.subheader("📊 Smart City Module Contribution")

    fig_pie = px.pie(
        summary_df,
        values="Events",
        names="Module",
        hole=0.45
    )

    fig_pie.update_layout(height=450)

    st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # -------- SYSTEM STATUS --------
    col1, col2, col3 = st.columns(3)

    col1.success("🚦 Traffic AI: Online")
    col2.success("👁 Vision AI: Active")
    col3.success("🤖 Analytics AI: Running")

    st.info("📡 Live AI Monitoring | Cloud Enabled | Real-time Processing")
    st.caption("UrbanBot Intelligence • Smart City AI Command Platform")
