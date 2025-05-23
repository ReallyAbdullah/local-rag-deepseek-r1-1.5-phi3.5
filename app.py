# app.py (updated)
import gradio as gr
import logging
from pathlib import Path
from rag.ingest import ingest_pdf, get_ingested_docs, delete_document
from rag.chains import RAGChain
from rag.agents import ProgressCallback
from config import UI_CONFIG, UPLOAD_CONFIG, UPLOAD_DIR
import shutil
import os
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize RAG chain
rag_chain = RAGChain()

INGESTED_DOCS = Path("data/ingested.json")


def process_upload(file):
    """Handle PDF upload and ingestion"""
    try:
        filename = Path(file.name).name

        # Validate file type
        if Path(file.name).suffix.lower() not in UPLOAD_CONFIG["allowed_types"]:
            return gr.Warning(
                f"Invalid file type: {filename}. Only PDF files are allowed."
            )

        # Check file size
        if Path(file.name).stat().st_size > UPLOAD_CONFIG["max_file_size"]:
            return gr.Warning(f"File too large: {filename}. Maximum size is 50MB.")

        # Process file
        dest_path = UPLOAD_DIR / filename
        if not dest_path.exists():
            shutil.copy(file.name, str(dest_path))

        # Ingest file
        if ingest_pdf(str(dest_path)):
            logger.info(f"Successfully processed {filename}")
            return gr.Info(
                f"✅ Successfully processed {filename} ({dest_path.stat().st_size / 1024:.1f} KB)"
            )
        else:
            return gr.Warning(f"Failed to process {filename}")

    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return gr.Error(f"Error processing {filename}: {str(e)}")


def delete_file(filename):
    """Delete a file from the system"""
    try:
        if delete_document(filename):
            logger.info(f"Successfully deleted {filename}")
            return gr.Info(f"✅ {filename} deleted successfully!")
        return gr.Warning(f"Failed to delete {filename}")
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        return gr.Error(f"Error: {str(e)}")


def get_chat_response(message, history):
    """Handle chat interactions"""
    try:
        # Initialize progress list for this interaction
        progress_updates = []

        def progress_callback(update):
            progress_updates.append(update)
            # Create an interim response with all progress so far
            updates_text = "  \n".join(progress_updates)
            interim_response = (
                "### 🤖 Agent Progress:\n"
                f"{updates_text}\n\n"
                "---\n"
                "Processing your request..."
            )
            return interim_response

        # Set up progress callback
        callback = ProgressCallback(progress_callback)
        rag_chain.agents.set_progress_callback(callback)

        # Get the final result
        result = rag_chain.invoke(message)

        # Clean up the answer
        answer = result["answer"]
        if "<think>" in answer:
            answer = answer.split("</think>")[-1].strip()

        # Format agent information
        agent_info = ""
        if "agent_info" in result:
            agent_info = f"""
### 🤖 Agent Information
- Crew Size: {result['agent_info']['crew_size']}
- Tasks Completed: {result['agent_info']['tasks_completed']}
"""

        # Format the final response
        response = f"""### Response from {result['model_used'].title()}

{answer}

{agent_info}
---
**Source Documents:**
{result['references']}

<div style="font-size: 0.8em; color: #666; margin-top: 1em">
Model: {result['model_used'].title()} • Response Time: {result.get('time_taken', 'N/A')}
</div>"""

        return response

    except Exception as e:
        logger.error(f"Error in chat response: {str(e)}")
        return f"""### ⚠️ Error

I encountered an error while processing your request:
{str(e)}

Please try again or rephrase your question."""


def create_ui():
    """Create the Gradio UI"""
    with gr.Blocks(
        title="Local Agentic RAG Assistant",
        theme=gr.themes.Soft(),
        css=".gradio-container {max-width: 1200px}",
    ) as app:
        gr.Markdown(
            """
            <div style="text-align: center; margin-bottom: 2rem">
                <h1 style="margin-bottom: 0.5rem">🤖 Local RAG Assistant</h1>
                <p style="color: #666">Powered by local LLMs and vector search</p>
            </div>
            """
        )

        with gr.Tabs() as tabs:
            # Chat Tab
            with gr.Tab("💬 Chat", id="chat"):
                chatbot = gr.Chatbot(
                    height=UI_CONFIG["chat_height"],
                    show_copy_button=True,
                    show_share_button=False,
                    avatar_images=["assets/human.webp", "assets/ai.png"],
                    type="messages",
                )
                msg = gr.Textbox(
                    placeholder="Ask me anything about your documents...",
                    container=False,
                    lines=UI_CONFIG["textbox_lines"],
                    scale=7,
                )
                with gr.Row():
                    submit = gr.Button("Send", variant="primary", scale=2)
                    clear = gr.Button("Clear", variant="secondary", scale=1)

                def user_message(message, history):
                    if message:
                        return "", history + [{"role": "user", "content": message}]
                    return "", history

                def bot_message(history):
                    if not history:
                        return history

                    user_message = history[-1]["content"]
                    bot_response = get_chat_response(user_message, history[:-1])

                    history.append({"role": "assistant", "content": bot_response})
                    return history

                msg.submit(user_message, [msg, chatbot], [msg, chatbot]).then(
                    bot_message, chatbot, chatbot
                )
                submit.click(user_message, [msg, chatbot], [msg, chatbot]).then(
                    bot_message, chatbot, chatbot
                )
                clear.click(lambda: None, None, chatbot, queue=False)

            # Document Management Tab
            with gr.Tab("📑 Documents", id="docs"):
                status_box = gr.Textbox(label="Status", interactive=False, visible=True)

                with gr.Row():
                    with gr.Column(scale=2):
                        upload = gr.UploadButton(
                            "📤 Upload PDF",
                            file_types=UPLOAD_CONFIG["allowed_types"],
                            variant="primary",
                        )

                    with gr.Column(scale=1):
                        refresh_btn = gr.Button("🔄 Refresh List", variant="secondary")

                def update_doc_list():
                    try:
                        docs = get_ingested_docs()
                        return [[d["filename"], d["timestamp"]] for d in docs]
                    except Exception as e:
                        logger.error(f"Error updating document list: {str(e)}")
                        return []

                # Document list with delete buttons
                doc_list = gr.Dataframe(
                    headers=["Filename", "Upload Date"],
                    datatype=["str", "str"],
                    interactive=False,
                    wrap=True,
                    value=update_doc_list(),  # Call the function directly instead of using lambda
                )

                with gr.Row():
                    delete_input = gr.Textbox(
                        label="Enter filename to delete",
                        placeholder="example.pdf",
                        interactive=True,
                    )
                    delete_btn = gr.Button("🗑️ Delete", variant="secondary")

                def handle_upload_complete(file):
                    try:
                        status = process_upload(file)
                        docs = update_doc_list()
                        return status, docs
                    except Exception as e:
                        return f"Error: {str(e)}", update_doc_list()

                def handle_delete(filename):
                    try:
                        status = delete_file(filename)
                        docs = update_doc_list()
                        return status, docs, ""
                    except Exception as e:
                        return f"Error: {str(e)}", update_doc_list(), filename

                # Wire up events
                upload.upload(
                    handle_upload_complete,
                    inputs=[upload],
                    outputs=[status_box, doc_list],
                )

                refresh_btn.click(
                    lambda: (None, update_doc_list()), outputs=[status_box, doc_list]
                )

                delete_btn.click(
                    handle_delete,
                    inputs=[delete_input],
                    outputs=[status_box, doc_list, delete_input],
                )

        gr.Markdown(
            """
            <div style="text-align: center; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #eee;">
                <p style="color: #666; font-size: 0.9rem">
                    Built with Gradio • Powered by Local LLMs • Vector Search by ChromaDB
                </p>
            </div>
            """
        )

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(server_port=UI_CONFIG["port"], share=UI_CONFIG["share"])
