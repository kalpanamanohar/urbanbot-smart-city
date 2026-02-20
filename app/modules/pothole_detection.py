"""
Pothole Detection Module
-------------------------
YOLOv8-based road damage detection.
Stores detection results in MySQL and triggers alerts.
"""

import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO
from sqlalchemy import text

from services.db import engine
from services.email_service import send_email_alert
from utils.logger import get_logger


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "potholess.pt")
IMAGE_SAVE_DIR = os.path.join(BASE_DIR, "images", "potholes")

os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

CONF_THRESHOLD = 0.25
logger = get_logger()


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_pothole_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.error("Pothole model file not found.")
            return None
        return YOLO(MODEL_PATH)
    except Exception as e:
        logger.error(f"Pothole model loading failed: {e}")
        st.error("Failed to load pothole detection model.")
        return None


# --------------------------------------------------
# SEVERITY LOGIC
# --------------------------------------------------

def determine_severity(pothole_count: int) -> str:
    if pothole_count == 0:
        return "No Damage"
    elif pothole_count <= 2:
        return "Minor"
    elif pothole_count <= 5:
        return "Moderate"
    else:
        return "Severe"


# --------------------------------------------------
# STORE EVENT IN DATABASE
# --------------------------------------------------

def store_event(city, area, severity, confidence, image_name):
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO pothole_events
                (city, area, severity, confidence_score, image_name, event_time)
                VALUES (:city, :area, :severity, :confidence, :image_name, NOW())
            """), {
                "city": city,
                "area": area,
                "severity": severity,
                "confidence": float(confidence),
                "image_name": image_name
            })

        logger.info("Pothole event stored successfully.")

    except Exception as e:
        logger.error(f"Pothole DB error: {e}")
        st.error("Database error while saving pothole event.")


# --------------------------------------------------
# MAIN UI
# --------------------------------------------------

def run():

    st.title("🕳 Smart Pothole Detection System")
    st.caption("YOLOv8-Based Road Damage Monitoring Platform")

    model = load_pothole_model()
    if model is None:
        return

    CITY_DATA = {
        "Chennai": ["Guindy", "T Nagar", "Velachery"],
        "Mumbai": ["Bandra", "Dadar", "Andheri"],
        "Delhi": ["Dwarka", "Rohini", "Karol Bagh"],
        "Bangalore": ["Whitefield", "Electronic City", "MG Road"],
        "Hyderabad": ["Gachibowli", "Madhapur", "Secunderabad"],
        "Kolkata": ["Salt Lake", "Howrah", "Park Street"],
        "Pune": ["Hinjewadi", "Kothrud", "Shivaji Nagar"],
        "Ahmedabad": ["Navrangpura", "Satellite", "Maninagar"]
    }

    if "pothole_result" not in st.session_state:
        st.session_state.pothole_result = None

    col1, col2 = st.columns(2)

    with col1:
        city = st.selectbox("🏙 Select City", list(CITY_DATA.keys()))
        area = st.selectbox("📍 Select Area", CITY_DATA[city])
        uploaded_file = st.file_uploader(
            "📷 Upload Road Image",
            type=["jpg", "png", "jpeg"]
        )
        detect_btn = st.button("🔍 Detect Potholes")

    with col2:
        preview = st.empty()

    # --------------------------------------------------
    # DETECTION LOGIC
    # --------------------------------------------------

    if uploaded_file and detect_btn:

        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            results = model.predict(image, conf=CONF_THRESHOLD)

            boxes = results[0].boxes
            pothole_count = 0
            avg_conf = 0.0

            if boxes is not None and len(boxes) > 0:
                pothole_count = len(boxes)
                avg_conf = float(boxes.conf.mean())

            severity = determine_severity(pothole_count)

            annotated_image = results[0].plot()

            image_name = f"pothole_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            save_path = os.path.join(IMAGE_SAVE_DIR, image_name)
            cv2.imwrite(save_path, annotated_image)

            # Store in DB
            store_event(city, area, severity, avg_conf, image_name)

            # Email alert if any pothole detected
            if pothole_count > 0:
                send_email_alert(
                    module="Pothole",
                    subject=f"{severity} Road Damage Detected - {city}",
                    body=f"""
UrbanBot Smart City Alert

City: {city}
Area: {area}
Severity: {severity}
Potholes Detected: {pothole_count}
Confidence Score: {avg_conf:.2f}
Time: {datetime.now()}
"""
                )

            st.session_state.pothole_result = {
                "image": annotated_image,
                "pothole_count": pothole_count,
                "confidence": avg_conf,
                "severity": severity,
                "timestamp": datetime.now()
            }

        except Exception as e:
            logger.error(f"Pothole detection error: {e}")
            st.error("Pothole detection failed.")


    # --------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------

    if st.session_state.pothole_result:

        result = st.session_state.pothole_result

        preview.image(result["image"], channels="BGR", width="stretch")

        c1, c2, c3 = st.columns(3)
        c1.metric("🕳 Potholes", result["pothole_count"])
        c2.metric("📊 Confidence", f"{result['confidence']:.2f}")
        c3.metric("🚧 Severity", result["severity"])

        st.success(f"Detection Time: {result['timestamp']}")