import streamlit as st
import ollama
from time import sleep

st.set_page_config(
    page_title="Model Management",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.subheader("Model Management", divider="red", anchor=False)

    st.subheader("Download Models", anchor=False)
    model_name = st.text_input(
        "Enter the name of the model to download ↓", placeholder="mistral"
    )
    if st.button(f":green[**Download**] :red[{model_name}]"):
        if model_name:
            try:
                
                ollama.pull(model_name)
                st.success(f"Downloaded model: {model_name}", icon="🎉")
                st.balloons()
                sleep(1)
                st.rerun()
            except Exception as e:
                st.error(
                    f"""Failed to download model: {
                    model_name}. Error: {str(e)}""",
                    icon="😳",
                )
        else:
            st.warning("Please enter a model name.", icon="⚠️")

    st.divider()

    st.subheader("Delete Models", anchor=False)
    models_info = ollama.list()
    available_models = [m["name"] for m in models_info["models"]]

    if available_models:
        selected_models = st.multiselect("Select models to delete", available_models)
        if st.button("🗑️ :red[**Delete Selected Model(s)**]"):
            for model in selected_models:
                try:
                    ollama.delete(model)
                    st.success(f"Deleted model: {model}", icon="🎉")
                    st.balloons()
                    sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(
                        f"""Failed to delete model: {
                        model}. Error: {str(e)}""",
                        icon="😳",
                    )
    else:
        st.info("No models available for deletion.", icon="🦗")


if __name__ == "__main__":
    main()
