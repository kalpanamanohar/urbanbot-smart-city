# 🏙️ UrbanBot Intelligence – Smart City AI Platform

UrbanBot Intelligence is an AI-powered Smart City Analytics Platform that integrates Machine Learning, Computer Vision, NLP, and LLM technologies to monitor and analyze urban data in real time.

The system provides intelligent insights on:

- 🚦 Traffic Prediction
- 🚑 Accident Detection
- 🕳 Pothole Detection
- 👥 Crowd Monitoring
- 🌫 Air Quality Forecasting (AQI)
- 🗣 Citizen Complaint Analysis
- 🤖 AI-powered SQL Chatbot

---

## 🚀 Features

### 1️⃣ Traffic Prediction
- LSTM-based time-series forecasting
- Predicts future vehicle volume
- Displays congestion level
- Stores results in MySQL database

### 2️⃣ Accident Detection
- YOLO-based object detection
- Detects accidents from image/video input
- Logs event data with timestamp

### 3️⃣ Pothole Detection
- YOLOv8-based road damage detection
- Automatically stores detected events

### 4️⃣ Crowd Monitoring
- Deep Learning model for crowd density estimation
- Calculates average crowd count per area

### 5️⃣ Air Quality Forecasting
- Predicts AQI levels using ML model
- Identifies best and worst air quality zones

### 6️⃣ Citizen Complaint Analysis
- NLP-based complaint categorization
- Stores and analyzes complaint trends

### 7️⃣ AI SQL Chatbot
- Powered by Groq LLM
- Converts natural language questions into SQL queries
- Fetches real-time insights from database

Example queries:
- "Which city has highest traffic?"
- "Top 3 accident areas last week"
- "Worst AQI city today"
- "Area with most complaints"

---

## 🧠 Technology Stack

| Component | Technology |
|------------|------------|
| Frontend | Streamlit |
| Database | MySQL |
| ORM | SQLAlchemy |
| ML Models | TensorFlow, Scikit-learn |
| Computer Vision | YOLOv8 (Ultralytics), OpenCV |
| Visualization | Plotly, Matplotlib, Folium |
| NLP & LLM | Groq API |
| Environment | Python 3.10+ |

---

## 🗄 Database Schema (Main Tables)

- `traffic_events`
- `accident_events`
- `pothole_events`
- `crowd_events`
- `aqi_events`
- `citizen_complaints`

---

## 📂 Project Structure
