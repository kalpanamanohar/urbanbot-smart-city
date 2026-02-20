import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO

from services.db import execute_query
from services.email_service import send_email_alert


# ---------------- CONFIG ----------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Accident.pt")
IMAGE_SAVE_DIR = os.path.join(BASE_DIR, "images", "accidents")

os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

CONF_THRESHOLD = 0.30


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


# ---------------- SEVERITY ----------------

def determine_severity(count):
    if count == 0:
        return "No Accident"
    elif count == 1:
        return "Minor"
    elif count <= 3:
        return "Moderate"
    else:
        return "Severe"


# ---------------- DATABASE INSERT ----------------

def store_event(city, area, severity, confidence, image_name):
    sql = """
        INSERT INTO accident_events
        (city, area, severity, confidence_score, image_name, event_time)
        VALUES (%s, %s, %s, %s, %s, NOW())
    """
    values = (city, area, severity, confidence, image_name)
    execute_query(sql, values)


# ---------------- MAIN UI ----------------

def run():

    st.title("🚑 Smart Accident Detection")
    st.caption("YOLOv8 Based Accident Detection System")

    model = load_model()

    CITY_DATA = {
        "Chennai": ["Guindy", "T Nagar", "Velachery"],
        "Mumbai": ["Bandra", "Dadar", "Andheri"],
        "Delhi": ["Dwarka", "Rohini", "Karol Bagh"],
    }

    col1, col2 = st.columns(2)

    with col1:
        city = st.selectbox("Select City", list(CITY_DATA.keys()))
        area = st.selectbox("Select Area", CITY_DATA[city])
        file = st.file_uploader("Upload Accident Image", type=["jpg", "png", "jpeg"])
        detect = st.button("Detect Accident")

    with col2:
        preview = st.empty()

    if file and detect:

        image_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        results = model.predict(image, conf=CONF_THRESHOLD)

        boxes = results[0].boxes
        count = len(boxes) if boxes else 0
        confidence = float(boxes.conf.mean()) if boxes else 0

        severity = determine_severity(count)

        output_image = results[0].plot()

        image_name = f"accident_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(os.path.join(IMAGE_SAVE_DIR, image_name), output_image)

        # Save to DB
        store_event(city, area, severity, confidence, image_name)

        # Email alert
        if count > 0:
            send_email_alert(
                module="Accident",
                subject=f"{severity} Accident Detected - {city}",
                body=f"""
City: {city}
Area: {area}
Severity: {severity}
Vehicles Involved: {count}
Confidence: {confidence:.2f}
Time: {datetime.now()}
"""
            )

        preview.image(output_image, channels="BGR", width="stretch")

        st.success(f"Severity: {severity} | Vehicles: {count}")