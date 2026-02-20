"""
Air Quality Forecasting Module
--------------------------------
LSTM-based AQI prediction using last 29 days + today manual input.
Stores forecast results and triggers alert for hazardous levels.
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
MODEL_PATH = os.path.join(BASE_DIR, "models", "aqi_lstm_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "models", "aqi_scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "AQI_dataset.csv")

logger = get_logger()


# --------------------------------------------------
# LOAD MODEL + SCALER
# --------------------------------------------------

@st.cache_resource
def load_aqi_components():
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            st.error("AQI model files not found.")
            return None, None

        model = load_model(MODEL_PATH, compile=False)
        scaler = joblib.load(SCALER_PATH)

        logger.info("AQI model loaded successfully.")
        return model, scaler

    except Exception as e:
        logger.error(f"AQI model loading failed: {e}")
        st.error("Failed to load AQI model.")
        return None, None


# --------------------------------------------------
# AQI CATEGORY (CPCB INDIA STANDARD)
# --------------------------------------------------

def classify_aqi(aqi: int) -> str:

    if aqi <= 50:
        return "Good 🟢"
    elif aqi <= 100:
        return "Satisfactory 🟡"
    elif aqi <= 200:
        return "Moderate 🟠"
    elif aqi <= 300:
        return "Poor 🔴"
    elif aqi <= 400:
        return "Very Poor 🟣"
    else:
        return "Severe ⚫"


# --------------------------------------------------
# STORE EVENT IN DATABASE
# --------------------------------------------------

def store_aqi_event(city: str, predicted_aqi: int, level: str):
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO aqi_events
                (city, predicted_aqi, pollution_level, event_time)
                VALUES (:city, :aqi, :level, NOW())
            """), {
                "city": city,
                "aqi": float(predicted_aqi),
                "level": level
            })

        logger.info("AQI event stored successfully.")

    except Exception as e:
        logger.error(f"AQI DB error: {e}")
        st.error("Database error while storing AQI event.")


# --------------------------------------------------
# MAIN STREAMLIT UI
# --------------------------------------------------

def run():

    st.title("🌫 Smart Air Quality Forecasting System")
    st.caption("Manual pollutant input → AI predicts tomorrow AQI (LSTM)")

    model, scaler = load_aqi_components()

    if model is None:
        return

    # -------- CITY --------
    city = st.selectbox(
        "🏙 Select City",
        ["Chennai", "Bangalore", "Delhi", "Mumbai", "Hyderabad"]
    )

    # -------- POLLUTION INPUT --------
    st.subheader("🧪 Enter Today's Pollution Levels")

    col1, col2 = st.columns(2)

    with col1:
        pm25 = st.number_input("PM2.5 (µg/m³)", 0.0, 1000.0, 80.0)
        pm10 = st.number_input("PM10 (µg/m³)", 0.0, 1000.0, 120.0)
        no2  = st.number_input("NO2 (µg/m³)", 0.0, 500.0, 40.0)
        so2  = st.number_input("SO2 (µg/m³)", 0.0, 500.0, 20.0)

    with col2:
        co   = st.number_input("CO (mg/m³)", 0.0, 20.0, 1.2)
        o3   = st.number_input("O3 (µg/m³)", 0.0, 500.0, 30.0)
        nh3  = st.number_input("NH3 (µg/m³)", 0.0, 500.0, 25.0)

    predict_btn = st.button("🔮 Predict Tomorrow AQI")

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    if predict_btn:

        try:
            if not os.path.exists(DATA_PATH):
                st.error("AQI dataset not found.")
                return

            features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'NH3']

            df = pd.read_csv(DATA_PATH)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(by='Date')
            df = df[['Date'] + features]

            df_daily = df.groupby('Date')[features].mean().reset_index()
            df_daily.set_index('Date', inplace=True)

            # Need at least 29 historical days
            if len(df_daily) < 29:
                st.error("Not enough historical data for prediction.")
                return

            last_29_days = df_daily[-29:].copy()

            today_data = pd.DataFrame(
                [[pm25, pm10, no2, so2, co, o3, nh3]],
                columns=features,
                index=[datetime.now()]
            )

            final_input = pd.concat([last_29_days, today_data])

            scaled = scaler.transform(final_input)
            X_input = scaled.reshape(1, 30, len(features))

            prediction = model.predict(X_input, verbose=0)

            # Inverse scaling
            dummy = np.zeros((1, len(features)))
            dummy[0][0] = prediction[0][0]
            prediction = scaler.inverse_transform(dummy)

            predicted_aqi = int(prediction[0][0])
            today_aqi = int(pm25)

            change = predicted_aqi - today_aqi

            if change > 0:
                trend = "📈 Pollution Increasing"
            elif change < 0:
                trend = "📉 Air Quality Improving"
            else:
                trend = "➡ No Change"

            level = classify_aqi(predicted_aqi)

            # Store in DB
            store_aqi_event(city, predicted_aqi, level)

            # Email alert if hazardous
            if predicted_aqi > 200:
                send_email_alert(
                    module="AQI",
                    subject=f"⚠ High Pollution Alert - {city}",
                    body=f"""
UrbanBot Smart City Alert

City: {city}
Today's AQI: {today_aqi}
Tomorrow AQI: {predicted_aqi}
Pollution Level: {level}
Time: {datetime.now()}

Health Advice:
• Avoid outdoor activities
• Wear mask
• Stay indoors
"""
                )

            st.session_state["aqi_result"] = {
                "today": today_aqi,
                "tomorrow": predicted_aqi,
                "change": change,
                "trend": trend,
                "level": level,
                "time": datetime.now()
            }

        except Exception as e:
            logger.error(f"AQI prediction error: {e}")
            st.error("AQI prediction failed.")


    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    if "aqi_result" in st.session_state:

        res = st.session_state["aqi_result"]

        st.subheader("🌫 AQI Forecast Dashboard")

        c1, c2 = st.columns(2)

        with c1:
            st.metric("🌤 Today's AQI", res["today"])

        with c2:
            st.metric("🔮 Tomorrow AQI", res["tomorrow"], delta=res["change"])

        st.success(f"Pollution Level: {res['level']}")
        st.info(res["trend"])
        st.caption(f"Prediction Time: {res['time']}")
