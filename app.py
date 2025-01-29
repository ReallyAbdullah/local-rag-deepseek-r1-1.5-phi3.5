# app.py (updated)
import gradio as gr
from pathlib import Path
from rag.ingest import ingest_pdf, get_ingested_docs
from rag.chains import RAGChain
import shutil
import os
import json

# Initialize RAG chain
rag_chain = RAGChain()

INGESTED_DOCS = Path("data/ingested.json")


def process_upload(file):
    """Handle PDF upload and ingestion"""
    try:
        upload_dir = Path("data/uploaded")
        upload_dir.mkdir(exist_ok=True)

        filename = Path(file.name).name
        dest_path = upload_dir / filename

        # Verify PDF content
        if dest_path.suffix.lower() != ".pdf":
            return f"❌ Invalid file type: {filename}"

        # Check file size (max 50MB)
        if Path(file.name).stat().st_size > 50 * 1024 * 1024:
            return f"❌ File too large: {filename} (max 50MB)"

        # Copy file if needed
        if not dest_path.exists():
            shutil.copy(file.name, str(dest_path))

        # Verify ingestion
        if ingest_pdf(str(dest_path)):
            try:
                rag = RAGChain()
                count = rag.vector_store._collection.count()
                if count > 0:
                    return f"✅ {filename} processed successfully! ({count} vectors)"
                return f"⚠️ Processed {filename} but no vectors created"
            except Exception as e:
                return f"⚠️ Verification failed: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


def respond(message, history):
    result = rag_chain.invoke(message)

    # Simple text-based model indicator
    model_name = "Phi-3.5" if result["model_used"] == "simple" else "DeepSeek-R1:1.5B"
    response = (
        f"[{model_name}]\n{result['answer']}\n\nReferences:\n{result['references']}"
    )

    return response


examples = [
    # Phi-3.5 example (simple)
    "List 3 key points from section 2.3",
    # DeepSeek example (complex)
    "Explain the implications of the main findings in the context of current industry trends",
    # Phi-3.5 example
    "What date was the experiment conducted?",
    # DeepSeek example
    "Compare the methodology used here with best practices in the field",
]


# app.py (UI improvements)
with gr.Blocks(title="Local RAG", theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
    <div style="text-align: center;">
        <h1>📚 Document Intelligence Assistant</h1>
        <p>Upload PDF documents and ask questions about their content</p>
    </div>
    """
    )

    with gr.Row(equal_height=False):
        # Document Upload Section
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("## 📤 Document Management")
            with gr.Group():
                upload = gr.UploadButton(
                    "Upload PDF Document",
                    file_types=[".pdf"],
                    variant="primary",
                    size="sm",
                )
                upload_result = gr.Markdown()

            doc_table = gr.Dataframe(
                headers=["Filename", "Upload Date"],
                datatype=["str", "str"],
                interactive=False,
                every=1,
            )

            # Refresh documents list
            def update_doc_list():
                docs = get_ingested_docs()
                return [[d["filename"], d["timestamp"]] for d in docs]

            doc_table.change(update_doc_list, inputs=None, outputs=doc_table, every=1)

        # Chat Interface Section
        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(
                respond,
                examples=examples,
                fill_height=True,
                clear_btn="🗑️ Clear History",
                submit_btn="🚀 Submit",
            )
            chatbot.chatbot.height = 800
            chatbot.textbox_lines = 3

    # Event handling
    upload.upload(process_upload, inputs=[upload], outputs=[upload_result])

    # Additional UI components
    gr.Markdown(
        """
    <div style="text-align: center; color: #666; margin-top: 20px;">
        <hr>
        <p>Powered by Local RAG • Models: DeepSeek-R1 & Phi-3.5 • Storage: ChromaDB</p>
    </div>
    """
    )

if __name__ == "__main__":
    app.launch(server_port=7860, share=False)
