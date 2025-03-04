# streamlit_app.py
import streamlit as st
from pathlib import Path
import pandas as pd
from datetime import datetime

from rag.ingest import ingest_pdf, get_ingested_docs, delete_document
from rag.chains import RAGChain

# Page config with custom theme
st.set_page_config(
    page_title="Local RAG Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Initialize RAG chain
@st.cache_resource
def get_rag_chain():
    return RAGChain()


rag_chain = get_rag_chain()

# Custom CSS
st.markdown(
    """
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .doc-table {
        font-size: 0.8em;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background: #f0f2f6;
    }
    .assistant-message {
        background: #e8f0fe;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Add this function before the sidebar code
def process_upload(uploaded_file):
    """Process uploaded PDF file"""
    if uploaded_file is None:
        return "No file uploaded"

    try:
        # Save uploaded file temporarily
        temp_path = Path(uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Ingest the PDF
        ingest_pdf(temp_path, uploaded_file.name)

        # Clean up temp file
        temp_path.unlink()

        return f"Successfully ingested {uploaded_file.name}"

    except Exception as e:
        return f"Error processing file: {str(e)}"


# Sidebar for document management
with st.sidebar:
    st.title("📑 Document Management")

    # File uploader with progress
    uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")
    if uploaded_file:
        with st.spinner("Processing document..."):
            result = process_upload(uploaded_file)
            st.success(result)

    # Document list with delete functionality
    st.subheader("Ingested Documents")
    docs = get_ingested_docs()
    if docs:
        df = pd.DataFrame(docs)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(
            df[["filename", "timestamp"]],
            hide_index=True,
            column_config={"filename": "Document", "timestamp": "Ingested On"},
            use_container_width=True,
        )

        # Delete document functionality
        doc_to_delete = st.selectbox(
            "Select document to remove:", [d["filename"] for d in docs]
        )
        if st.button("🗑️ Delete Document", type="secondary"):
            if delete_document(doc_to_delete):
                st.success(f"Deleted {doc_to_delete}")
                st.rerun()

# Main chat interface
st.title("🤖 Local RAG Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(prompt)

            # Format response with model info and references
            response_text = f"""
            **Response**: {response['answer']}
            
            ---
            **Sources**:
            {response['references']}
            
            *Generated using {response['model_used']}*
            """

            st.markdown(response_text)
            st.session_state.messages.append(
                {"role": "assistant", "content": response_text}
            )

# Footer
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        <p>Powered by Local RAG • Models: DeepSeek-R1 & Phi-3.5 • Storage: ChromaDB</p>
    </div>
    """,
    unsafe_allow_html=True,
)
