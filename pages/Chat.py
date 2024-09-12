import ollama
import streamlit as st

# Configure the Streamlit page layout
st.set_page_config(
    page_title="Local Legal Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)

def get_model_names(models_data: list) -> tuple:
    """
    Retrieves model names from the provided model information.
    """
    return tuple(model["name"] for model in models_data["models"])

def main():
    """
    The main function that runs the application.
    """
    st.subheader("General Chat", divider="red", anchor=False)

    # Fetch available models from Ollama
    models_info = ollama.list()
    available_models = get_model_names(models_info)

    if available_models:
        selected_model = st.selectbox(
            "Pick a model available locally on your system ↓", available_models
        )
    else:
        st.warning("You have not pulled any model from Ollama yet!", icon="⚠️")
        if st.button("Go to settings to download a model"):
            st.page_switch("pages/Model Management.py")

    message_container = st.container(height=500, border=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        avatar_icon = "🤖" if message["role"] == "assistant" else "🧑"
        with message_container.chat_message(message["role"], avatar=avatar_icon):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter a prompt here..."):
        try:
            st.session_state.messages.append({"role": "user", "content": prompt})

            message_container.chat_message("user", avatar="🧑").markdown(prompt)

            with message_container.chat_message("assistant", avatar="🤖"):
                with st.spinner("Model working..."):
                    # Using Ollama model for local chat completions
                    stream = ollama.chat(
                        model=selected_model,
                        messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ],
                        stream=True,
                    )
                # Stream the response
                response = st.write_stream(stream)

            st.session_state.messages.append(
                {"role": "assistant", "content": response})

        except Exception as error:
            st.error(f"Error: {error}", icon="⛔️")


if __name__ == "__main__":
    main()