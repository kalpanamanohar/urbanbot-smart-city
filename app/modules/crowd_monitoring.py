"""
Crowd Monitoring Module
------------------------
CNN-based crowd density estimation.
Stores results in MySQL and triggers alerts.
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
from datetime import datetime
from sqlalchemy import text

from services.db import engine
from services.email_service import send_email_alert
from utils.logger import get_logger


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "crowd_density_cc50_v1.pth")
IMAGE_SAVE_DIR = os.path.join(BASE_DIR, "images", "crowd")

os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

logger = get_logger()


# --------------------------------------------------
# MODEL DEFINITION
# --------------------------------------------------

class CrowdNet(nn.Module):
    def __init__(self):
        super().__init__()

        vgg = models.vgg16(weights="IMAGENET1K_V1")
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])

        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2),
            nn.ReLU(),
        )

        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_crowd_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.error("Crowd model file not found.")
            return None

        model = CrowdNet()
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
        model.eval()

        logger.info("Crowd model loaded successfully")
        return model

    except Exception as e:
        logger.error(f"Crowd model loading failed: {e}")
        st.error("Crowd model loading failed.")
        return None


# --------------------------------------------------
# IMAGE TRANSFORM
# --------------------------------------------------

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# CROWD CLASSIFICATION
# --------------------------------------------------

def classify_crowd(count: int):
    if count < 50:
        return "LOW 🟢", "LOW"
    elif count < 150:
        return "MEDIUM 🟡", "MEDIUM"
    else:
        return "HIGH 🔴", "HIGH"


# --------------------------------------------------
# STORE EVENT IN DATABASE
# --------------------------------------------------

def store_crowd_event(city, area, landmark, count, crowd_level, image_name):
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO crowd_events
                (city, area, landmark, crowd_count, crowd_level, image_name, event_time)
                VALUES (:city, :area, :landmark, :count, :level, :image, NOW())
            """), {
                "city": city,
                "area": area,
                "landmark": landmark,
                "count": int(count),
                "level": crowd_level,
                "image": image_name
            })

        logger.info("Crowd event stored successfully.")

    except Exception as e:
        logger.error(f"Crowd DB error: {e}")
        st.error("Database error while saving crowd event.")


# --------------------------------------------------
# MAIN UI
# --------------------------------------------------

def run():

    st.title("👥 Smart Crowd Density Estimation")
    st.caption("AI-based real-time crowd counting using CNN")

    model = load_crowd_model()
    if model is None:
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        city = st.text_input("🏙 City", "Chennai")

    with col2:
        area = st.text_input("📍 Area", "T Nagar")

    with col3:
        landmark = st.text_input("🏛 Landmark", "Bus Stand")

    uploaded_image = st.file_uploader("📷 Upload Crowd Image", type=["jpg", "jpeg", "png"])

    if st.button("🔍 Estimate Crowd") and uploaded_image:

        try:
            image = Image.open(uploaded_image).convert("RGB")

            # Save image locally
            image_name = f"crowd_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            save_path = os.path.join(IMAGE_SAVE_DIR, image_name)
            image.save(save_path)

            img_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(img_tensor)
                density_map = output.squeeze().numpy()

            crowd_count = int(max(density_map.sum(), 0))

            label, raw_level = classify_crowd(crowd_count)

            # Store in DB
            store_crowd_event(city, area, landmark, crowd_count, raw_level, image_name)

            # Email alert
            if raw_level == "HIGH":
                send_email_alert(
                    module="Crowd",
                    subject=f"🚨 Heavy Crowd Alert - {city}",
                    body=f"""
UrbanBot Smart City Alert

City: {city}
Area: {area}
Landmark: {landmark}

Estimated Crowd: {crowd_count}
Crowd Level: {label}
Time: {datetime.now()}

Recommendation:
Deploy crowd control personnel immediately.
"""
                )

            st.session_state["crowd_result"] = {
                "count": crowd_count,
                "level": label,
                "image": image,
                "density": density_map,
                "time": datetime.now()
            }

        except Exception as e:
            logger.error(f"Crowd estimation error: {e}")
            st.error("Crowd estimation failed.")


    # --------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------

    if "crowd_result" in st.session_state:

        res = st.session_state["crowd_result"]

        st.subheader("👥 Crowd Analysis Dashboard")

        col1, col2 = st.columns(2)

        with col1:
            st.image(res["image"], caption="Uploaded Image", use_container_width=True)

        with col2:
            heatmap = cv2.normalize(res["density"], None, 0, 255, cv2.NORM_MINMAX)
            heatmap = heatmap.astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            st.image(heatmap, caption="Density Heatmap", use_container_width=True)

        st.metric("Estimated Crowd Count", res["count"])
        st.success(f"Crowd Level: {res['level']}")
        st.caption(f"Prediction Generated at: {res['time']}")