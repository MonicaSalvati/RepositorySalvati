import streamlit as st
from login import login_page
from chat import chatbot_app

# Inizializza variabili di sessione
for key in ["endpoint", "subscription_key", "model_name", "api_version", "logged_in"]:
    if key not in st.session_state:
        st.session_state[key] = ""

# Routing
if not st.session_state["logged_in"]:
    login_page()
else:
    chatbot_app()

