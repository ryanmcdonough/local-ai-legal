import ollama
import streamlit as st
import os
from docx import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain

# Configure the Streamlit page layout
st.set_page_config(
    page_title="Local Legal Chatbot with Document Support",
    layout="wide",
    initial_sidebar_state="expanded",
)

def get_model_names(models_data: list) -> tuple:
    """
    Retrieves model names from the provided model information.
    """
    return tuple(model["name"] for model in models_data["models"])

def extract_docx_content(file):
    """
    Extracts content from a .docx file.
    """
    doc = Document(file)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def chunk_data(data, chunk_size=512, chunk_overlap=50):
    """
    Splits the data into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(data)
    return chunks

def create_embeddings_local_model(chunks, model_name):
    """
    Create embeddings with local Ollama model and store them in Chroma vector store.
    """
    embeddings = OllamaEmbeddings(model=model_name)
    vector_store = Chroma.from_texts(chunks, embeddings)
    return vector_store

def ask_and_get_answer(vector_store, full_content, query, model_name, use_full_content=False):
    """
    Handles querying using either embeddings or full content based on the toggle.
    """
    llm = Ollama(model=model_name)
    
    if use_full_content:
        template = """Use the following full document content to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Full Content: {context}

        Question: {question}
        Answer:"""
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(context=full_content, question=query)
    else:
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt}
        )
        response = chain.run(query)
    
    return response

def run_chat_app():
    """
    Core function to run the Streamlit chatbot interface with document support.
    """
    st.subheader("Local AI - Document Chat", divider="red", anchor=False)

    # Fetch available models from Ollama
    models_data = ollama.list()
    model_options = get_model_names(models_data)

    if not model_options:
        st.warning("No local models detected! Please download a model first.", icon="⚠️")
        if st.button("Go to settings to download a model"):
            st.page_switch("pages/Model Management.py")
        return

    # Create two columns for model selection
    col1, col2 = st.columns(2)

    with col1:
        # Model selection dropdown for text generation
        selected_text_model = st.selectbox("Choose a model for text generation ↓", model_options, key="text_model")

    with col2:
        # Model selection dropdown for embeddings
        selected_embedding_model = st.selectbox("Choose a model for embeddings ↓", model_options, key="embedding_model")

    # Upload document file(s) for analysis
    uploaded_files = st.file_uploader("Upload .docx documents for analysis", accept_multiple_files=True, type=['docx'])

    # Chat container with height and border
    chat_container = st.container(height=500, border=True)

    # Initialize session states
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "full_content" not in st.session_state:
        st.session_state.full_content = ""
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    # Handle document processing
    if uploaded_files:
        new_files = [file for file in uploaded_files if file.name not in st.session_state.processed_files]
        if new_files:
            progress_bar = st.progress(0)
            status_text = st.empty()

            all_chunks = []
            for i, uploaded_file in enumerate(new_files):
                status_text.text(f"Processing document {i+1} of {len(new_files)}...")
                
                content = extract_docx_content(uploaded_file)
                st.session_state.full_content += f"\n\n{content}"
                chunks = chunk_data(content)
                all_chunks.extend(chunks)
                
                progress = (i + 1) / len(new_files)
                progress_bar.progress(progress)

                st.session_state.processed_files.add(uploaded_file.name)

            if all_chunks:
                status_text.text("Creating embeddings...")
                if st.session_state.vector_store is None:
                    st.session_state.vector_store = create_embeddings_local_model(all_chunks, selected_embedding_model)
                else:
                    # Add new texts to existing vector store
                    st.session_state.vector_store.add_texts(all_chunks)
                
                progress_bar.progress(1.0)
                status_text.text("Documents processed successfully!")

    # Display chat history
    for msg in st.session_state.chat_history:
        avatar_icon = "🤖" if msg["role"] == "assistant" else "🧑"
        with chat_container.chat_message(msg["role"], avatar=avatar_icon):
            st.markdown(msg["content"])

    # Add toggle for full content
    use_full_content = st.toggle("Use full document content", value=False)

    # Process user input
    user_input = st.chat_input("Type your message or ask about uploaded documents...")
    
    if user_input:
        try:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            chat_container.chat_message("user", avatar="🧑").markdown(user_input)

            if st.session_state.vector_store or st.session_state.full_content:
                with chat_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Generating response..."):
                        answer = ask_and_get_answer(
                            st.session_state.vector_store, 
                            st.session_state.full_content, 
                            user_input, 
                            selected_text_model, 
                            use_full_content=use_full_content
                        )
                        st.markdown(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
            else:
                st.warning("No documents have been uploaded. Please upload a document to analyze.")

        except Exception as error:
            st.error(f"Error: {error}", icon="⛔️")

if __name__ == "__main__":
    run_chat_app()