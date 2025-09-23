import streamlit as st
from openai import AzureOpenAI

def chatbot_app():
    st.title("ChatGPT-like app")

    # Set OpenAI API key from Streamlit secrets
    client = AzureOpenAI(
        api_version=st.session_state["api_version"],
        azure_endpoint=st.session_state["endpoint"],
        api_key=st.session_state["subscription_key"],
    )

    # Set a default model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = st.session_state.get("model_name", "gpt-4o")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Enter a request"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": "system", "content": "Answer in english, even if the user write in italian."}
                ] + [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
