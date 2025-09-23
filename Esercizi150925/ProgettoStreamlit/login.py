import streamlit as st

def login_page():
    st.title("Login Page")
    st.header("Enter your credentials")

    endpoint = st.text_input("Endpoint URI")
    model_name = st.text_input("Model Name")
    api_version = st.text_input("API Version")
    subscription_key = st.text_input("Subscription Key", type="password")

    if st.button("Log in"):
        if endpoint and subscription_key and model_name and api_version:
            st.session_state["endpoint"] = endpoint
            st.session_state["subscription_key"] = subscription_key
            st.session_state["model_name"] = model_name
            st.session_state["api_version"] = api_version
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            st.error("Please, fill all the fileds.")
