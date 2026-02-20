"""
Citizen Complaints Module
--------------------------
NLP-based complaint sentiment & priority classification.
Stores complaints in MySQL and triggers alerts for high priority issues.
"""

import streamlit as st
import os
from datetime import datetime
import joblib
from sqlalchemy import text

from services.db import engine
from services.email_service import send_email_alert
from utils.logger import get_logger


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "complaint_ml_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

logger = get_logger()


# --------------------------------------------------
# LOAD NLP MODEL
# --------------------------------------------------

@st.cache_resource
def load_nlp_model():
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            st.error("Complaint model files not found.")
            return None, None

        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)

        logger.info("NLP model loaded successfully")
        return model, vectorizer

    except Exception as e:
        logger.error(f"NLP model load error: {e}")
        st.error("Failed to load NLP model.")
        return None, None


# --------------------------------------------------
# CATEGORY + PRIORITY LOGIC
# --------------------------------------------------

def classify_category(text: str) -> str:
    text = text.lower()

    categories = {
        "Garbage": ["garbage", "waste", "trash", "dump"],
        "Traffic": ["traffic", "jam", "signal", "congestion"],
        "Accident": ["accident", "crash", "collision", "injured"],
        "Electrical": ["electric", "wire", "shock", "power"],
        "Water": ["water", "leak", "pipe", "flood"],
        "Sewage": ["sewage", "drain", "overflow", "gutter"]
    }

    for cat, keywords in categories.items():
        if any(word in text for word in keywords):
            return cat

    return "General"


def classify_priority(text: str, sentiment: str) -> str:
    text = text.lower()

    if any(w in text for w in ["accident", "fire", "electric", "shock", "flood", "collapse", "injury"]):
        return "High"

    if any(w in text for w in ["garbage", "traffic", "pothole", "sewage", "streetlight"]):
        return "Medium"

    if sentiment == "Negative":
        return "Medium"

    return "Low"


# --------------------------------------------------
# STORE COMPLAINT IN DATABASE
# --------------------------------------------------

def store_complaint(city, area, complaint_text, sentiment, priority, category):

    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO citizen_complaints
                (city, area, complaint_text, sentiment, priority, category, status, event_time)
                VALUES (:city, :area, :text, :sentiment, :priority, :category, :status, NOW())
            """), {
                "city": city,
                "area": area,
                "text": complaint_text,
                "sentiment": sentiment,
                "priority": priority,
                "category": category,
                "status": "Pending"
            })

        logger.info("Complaint stored successfully.")

    except Exception as e:
        logger.error(f"Complaint DB error: {e}")
        st.error("Database error while saving complaint.")


# --------------------------------------------------
# MAIN UI
# --------------------------------------------------

def run():

    st.title("📢 Citizen Complaint Analysis System")
    st.caption("NLP-Based Smart City Feedback Monitoring")

    model, vectorizer = load_nlp_model()

    if model is None:
        return

    CITY_DATA = {
        "Chennai": ["Guindy", "T Nagar", "Velachery"],
        "Mumbai": ["Bandra", "Dadar", "Andheri"],
        "Delhi": ["Dwarka", "Rohini", "Karol Bagh"],
        "Bangalore": ["Whitefield", "Electronic City", "MG Road"],
        "Hyderabad": ["Gachibowli", "Madhapur", "Secunderabad"]
    }

    col1, col2 = st.columns(2)

    with col1:
        city = st.selectbox("🏙 Select City", list(CITY_DATA.keys()))
        area = st.selectbox("📍 Select Area", CITY_DATA[city])
        complaint_text = st.text_area("📝 Enter Citizen Complaint", height=160)
        analyze_btn = st.button("🔍 Analyze & Submit")

    with col2:
        result_box = st.empty()

    # --------------------------------------------------
    # ANALYSIS
    # --------------------------------------------------

    if analyze_btn:

        if not complaint_text.strip():
            st.warning("Please enter a complaint before submitting.")
            return

        try:
            vec = vectorizer.transform([complaint_text])
            pred = model.predict(vec)[0]

            sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
            sentiment = sentiment_map.get(pred, "Unknown")

            category = classify_category(complaint_text)
            priority = classify_priority(complaint_text, sentiment)

            # Store in DB
            store_complaint(city, area, complaint_text, sentiment, priority, category)

            # Email alert for high priority
            if priority == "High":
                send_email_alert(
                    module="Citizen Complaint",
                    subject="🚨 High Priority Citizen Complaint",
                    body=f"""
UrbanBot Smart City Alert

City: {city}
Area: {area}
Category: {category}
Sentiment: {sentiment}
Priority: {priority}

Complaint:
{complaint_text}

Time: {datetime.now()}
"""
                )

            st.session_state["complaint_result"] = {
                "sentiment": sentiment,
                "priority": priority,
                "category": category,
                "time": datetime.now()
            }

        except Exception as e:
            logger.error(f"Complaint analysis error: {e}")
            st.error("Complaint analysis failed.")


    # --------------------------------------------------
    # RESULT DISPLAY
    # --------------------------------------------------

    if "complaint_result" in st.session_state:

        res = st.session_state["complaint_result"]

        result_box.metric("📊 Sentiment", res["sentiment"])
        st.success(f"🚦 Priority Level: {res['priority']}")
        st.info(f"📂 Category: {res['category']}")
        st.caption(f"🕒 Logged At: {res['time']}")
