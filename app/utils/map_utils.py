import streamlit as st
import folium
from streamlit_folium import st_folium

def show_city_map(title, center_lat, center_lon, markers):

    st.subheader(f"🗺 {title} Incident Map")

    m = folium.Map(location=[center_lat, center_lon], zoom_start=5)

    for marker in markers:
        folium.Marker(
            location=[marker["lat"], marker["lon"]],
            popup=marker["label"],
            icon=folium.Icon(color=marker.get("color", "red"))
        ).add_to(m)

    st_folium(m, width="stretch")

