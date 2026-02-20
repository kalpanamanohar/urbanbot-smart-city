"""
Traffic Prediction Module
--------------------------
AI-based next day traffic prediction using LSTM.
Stores predictions in MySQL and triggers alerts for high congestion.
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
from datetime import datetime
import joblib
from tensorflow.keras.models import load_model
from sqlalchemy import text

from services.db import engine
from services.email_service import send_email_alert
from utils.logger import get_logger


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "traffic_lstm_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "models", "traffic_scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "Banglore_traffic_Dataset.csv")

logger = get_logger()


# --------------------------------------------------
# LOAD MODEL + SCALER
# --------------------------------------------------

@st.cache_resource
def load_traffic_components():
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            st.error("Traffic model files not found.")
            return None, None

        model = load_model(MODEL_PATH, compile=False)
        scaler = joblib.load(SCALER_PATH)

        logger.info("Traffic model loaded successfully.")
        return model, scaler

    except Exception as e:
        logger.error(f"Traffic model loading failed: {e}")
        st.error("Traffic model failed to load.")
        return None, None


# --------------------------------------------------
# CONGESTION CLASSIFICATION
# --------------------------------------------------

def classify_congestion(vehicle_count: int) -> str:
    if vehicle_count < 25000:
        return "LOW 🟢"
    elif vehicle_count < 40000:
        return "MEDIUM 🟡"
    else:
        return "HIGH 🔴"


# --------------------------------------------------
# STORE EVENT IN DATABASE
# --------------------------------------------------

def store_traffic_event(city: str, predicted_traffic: int, level: str):
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO traffic_events
                (city, predicted_vehicles, congestion_level, event_time)
                VALUES (:city, :vehicles, :level, NOW())
            """), {
                "city": city,
                "vehicles": float(predicted_traffic),
                "level": level
            })

        logger.info("Traffic event stored successfully.")

    except Exception as e:
        logger.error(f"Traffic DB error: {e}")
        st.error("Database error occurred while saving traffic prediction.")


# --------------------------------------------------
# MAIN STREAMLIT UI
# --------------------------------------------------

def run():

    st.title("🚦 Smart Traffic Congestion Forecasting")
    st.caption("AI-based next day traffic prediction using LSTM")

    model, scaler = load_traffic_components()

    if model is None:
        return

    city = st.selectbox(
        "🏙 Select City",
        ["Bangalore", "Chennai", "Delhi", "Mumbai", "Hyderabad"]
    )

    if st.button("🔮 Predict Tomorrow Traffic"):

        try:
            # ---------- DATASET CHECK ----------
            if not os.path.exists(DATA_PATH):
                st.error("Traffic dataset not found.")
                return

            df = pd.read_csv(DATA_PATH)

            # ---------- COLUMN VALIDATION ----------
            possible_volume_cols = [
                "Traffic Volume",
                "traffic_volume",
                "Vehicle Count",
                "vehicles"
            ]

            volume_col = None
            for col in possible_volume_cols:
                if col in df.columns:
                    volume_col = col
                    break

            if "Date" not in df.columns or volume_col is None:
                st.error("Dataset format incorrect. Required columns missing.")
                return

            # ---------- PREPROCESS ----------
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values(by="Date")
            df = df[["Date", volume_col]]

            df_daily = df.groupby("Date")[volume_col].mean().reset_index()
            df_daily.set_index("Date", inplace=True)

            if len(df_daily) < 30:
                st.error("Minimum 30 days of data required for prediction.")
                return

            today_traffic = int(df_daily.iloc[-1][volume_col])

            # ---------- SCALING ----------
            scaled_data = scaler.transform(df_daily[[volume_col]])
            last_30 = scaled_data[-30:].reshape(1, 30, 1)

            # ---------- PREDICTION ----------
            prediction = model.predict(last_30, verbose=0)

            if prediction.shape[-1] != 1:
                st.error("Model output shape unexpected.")
                return

            predicted_traffic = scaler.inverse_transform(prediction)[0][0]
            predicted_traffic = int(predicted_traffic)

            # ---------- TREND ----------
            change = predicted_traffic - today_traffic

            if change > 0:
                trend = "📈 Increasing Traffic"
            elif change < 0:
                trend = "📉 Decreasing Traffic"
            else:
                trend = "➡ No Change"

            # ---------- CLASSIFICATION ----------
            level = classify_congestion(predicted_traffic)

            # ---------- STORE IN DATABASE ----------
            store_traffic_event(city, predicted_traffic, level)

            # ---------- EMAIL ALERT ----------
            if "HIGH" in level:
                send_email_alert(
                    module="Traffic",
                    subject=f"🚦 Heavy Traffic Alert - {city}",
                    body=f"""
UrbanBot Smart City Alert

City: {city}
Today's Traffic: {today_traffic}
Tomorrow Prediction: {predicted_traffic}
Congestion Level: {level}
Time: {datetime.now()}

Recommendation:
• Avoid peak travel hours (8 AM – 11 AM)
• Use alternate routes
• Enable live navigation
"""
                )

            # ---------- SAVE SESSION ----------
            st.session_state["traffic_result"] = {
                "today": today_traffic,
                "tomorrow": predicted_traffic,
                "change": change,
                "trend": trend,
                "level": level,
                "time": datetime.now()
            }

            logger.info("Traffic prediction completed successfully.")

        except Exception as e:
            logger.error(f"Traffic prediction error: {e}")
            st.error("Prediction failed due to unexpected error.")


    # --------------------------------------------------
    # RESULT DISPLAY
    # --------------------------------------------------

    if "traffic_result" in st.session_state:

        res = st.session_state["traffic_result"]

        st.subheader("🚦 Traffic Forecast Dashboard")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="🚗 Today's Vehicle Count",
                value=res["today"]
            )

        with col2:
            st.metric(
                label="🔮 Tomorrow Predicted Vehicles",
                value=res["tomorrow"],
                delta=res["change"]
            )

        st.success(f"Congestion Level: {res['level']}")
        st.info(res["trend"])
        st.caption(f"Prediction Generated at: {res['time']}")
