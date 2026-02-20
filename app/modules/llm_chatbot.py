"""
LLM Chatbot UI
--------------
Streamlit chat interface for UrbanBot Smart City AI.
"""

import streamlit as st
from modules.chatbot_engine import process_user_query
from utils.logger import get_logger

logger = get_logger()


def run():

    st.subheader("🤖 UrbanBot – Smart City AI Chatbot")
    st.caption("Real-time city analytics powered by AI + SQL Intelligence")

    # ---------- INIT SESSION ----------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ---------- CLEAR BUTTON ----------
    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    # ---------- USER INPUT ----------
    user_input = st.chat_input("Ask your city analytics question...")

    if user_input:

        # Limit chat history (prevent memory growth)
        if len(st.session_state.messages) > 20:
            st.session_state.messages = st.session_state.messages[-20:]

        st.session_state.messages.append(("user", user_input))

        try:
            with st.spinner("UrbanBot is analyzing..."):
                response = process_user_query(user_input)

            if not response:
                response = "Sorry, I couldn't generate a response."

        except Exception as e:
            logger.error(f"Chatbot error: {e}")
            response = "An internal error occurred while processing your request."

        st.session_state.messages.append(("assistant", response))

    # ---------- DISPLAY CHAT ----------
    for role, message in st.session_state.messages:
        with st.chat_message(role):
            st.write(message)
