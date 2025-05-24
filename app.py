# app.py (Updated with streaming and preserved Document Management)
import gradio as gr
import logging
from pathlib import Path
from rag.ingest import ingest_pdf, get_ingested_docs, delete_document # Preserved
from rag.chains import RAGChain # Preserved
from rag.agents import ProgressCallback # Ensured import
from config import UI_CONFIG, UPLOAD_CONFIG, UPLOAD_DIR # Preserved
import shutil
import os # Preserved (though not explicitly in snippet, it's common)
import json # Preserved (though not explicitly in snippet, it's common)

# Configure logging (Preserved from original)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize RAG chain (Preserved from original)
rag_chain = RAGChain()

# INGESTED_DOCS path (Preserved from original)
INGESTED_DOCS = Path("data/ingested.json")


# process_upload and delete_file functions (Preserved from original app.py)
def process_upload(file):
    """Handle PDF upload and ingestion"""
    try:
        # file is a Gradio FileData object, file.name is the temp path to the uploaded file
        filename = Path(file.name).name # Get the original filename

        # Validate file type
        if Path(filename).suffix.lower() not in UPLOAD_CONFIG["allowed_types"]:
            # Use filename for user-facing messages
            return gr.Warning(
                f"Invalid file type: {filename}. Only PDF files are allowed."
            )

        # Check file size (use file.name which is the temp path for stat)
        if Path(file.name).stat().st_size > UPLOAD_CONFIG["max_file_size"]:
            return gr.Warning(f"File too large: {filename}. Maximum size is {UPLOAD_CONFIG['max_file_size'] // (1024*1024)}MB.")

        # Process file: Copy from temp path (file.name) to UPLOAD_DIR
        dest_path = UPLOAD_DIR / filename
        if not dest_path.exists(): # Check if file with same name already exists in UPLOAD_DIR
            shutil.copy(file.name, str(dest_path))
        else:
            logger.info(f"File {filename} already exists in {UPLOAD_DIR}. Using existing file for ingestion.")


        # Ingest file using its path in UPLOAD_DIR
        if ingest_pdf(str(dest_path)):
            logger.info(f"Successfully processed and ingested {filename}")
            # Use dest_path.stat() as file.name is a temp file
            return gr.Info(
                f"✅ Successfully processed {filename} ({dest_path.stat().st_size / 1024:.1f} KB)"
            )
        else:
            return gr.Warning(f"Failed to process {filename}")

    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}", exc_info=True)
        # Use filename if available, else a generic message
        fn = Path(file.name).name if hasattr(file, 'name') and file.name else "the uploaded file"
        return gr.Error(f"Error processing {fn}: {str(e)}")


def delete_file(filename_to_delete: str):
    """Delete a file from the system"""
    try:
        if not filename_to_delete or not filename_to_delete.strip():
            return gr.Warning("Please provide a filename to delete.")
        
        # delete_document function is from rag.ingest
        if delete_document(filename_to_delete):
            logger.info(f"Successfully deleted {filename_to_delete}")
            return gr.Info(f"✅ {filename_to_delete} deleted successfully!")
        # If delete_document returns False, it means it couldn't find/delete
        return gr.Warning(f"Failed to delete {filename_to_delete}. It might not exist or is protected.")
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}", exc_info=True)
        return gr.Error(f"Error deleting {filename_to_delete}: {str(e)}")


# Modified get_chat_response as per subtask
def get_chat_response(message: str, history: list):
    """
    Handles chat interactions and streams responses. This function is a generator.
    """
    logger.info(f"User query for streaming: {message}")
    
    _yield_buffer = [] 

    def backend_callback_appender(message_from_backend: str):
        _yield_buffer.append(message_from_backend)

    current_call_progress_callback = ProgressCallback(backend_callback_appender)
    rag_chain.set_progress_callback(current_call_progress_callback)

    yield "🤖 Assistant is processing your request..." 

    try:
        final_result_dict = rag_chain.invoke(message) 
        
        accumulated_response_text = ""
        # Track if the last yielded item was a status message to avoid double newlines or weird spacing
        last_yielded_was_status = False 

        for idx, item_from_buffer in enumerate(_yield_buffer):
            is_status_message = any(item_from_buffer.startswith(prefix) for prefix in 
                                    ["🚀", "▶️", "🧠", "✅", "🏁", "📝", "✔️", "🔍", "ℹ️", "⚠️", "❌", "🤖"])
            
            if not is_status_message:
                # If the previous yield was a status, and this is text, prepend a newline for readability if needed.
                # However, simple concatenation of stream might be better.
                # accumulated_response_text += ("\n" if last_yielded_was_status else "") + item_from_buffer
                accumulated_response_text += item_from_buffer
                yield accumulated_response_text 
                last_yielded_was_status = False
            else:
                if accumulated_response_text and not last_yielded_was_status : # Yield pending text before status
                    yield accumulated_response_text
                    accumulated_response_text = "" # Reset for next text part
                yield item_from_buffer 
                last_yielded_was_status = True
        
        # After buffer is processed, handle the main answer from final_result_dict
        answer = final_result_dict.get("answer", "No answer found.")
        if "<think>" in answer: 
            answer = answer.split("</think>")[-1].strip()

        # If the accumulated text from streaming isn't the full final answer, yield the full answer.
        # This handles cases where streaming might not capture everything or for non-streamed parts.
        if accumulated_response_text != answer :
             # If there was pending text, and it's different from the final answer, yield it.
             # Or if the final answer is just different from the last text chunk.
            if accumulated_response_text and not last_yielded_was_status:
                 yield accumulated_response_text # yield remaining part of stream if any
            # Yield the definitive final answer if it wasn't fully represented by the stream
            yield answer


        agent_info_str = ""
        if "agent_info" in final_result_dict and final_result_dict["agent_info"]:
            agent_info = final_result_dict["agent_info"]
            agent_info_str = f"\n\n### 🤖 Agent Information\n- Crew Size: {agent_info.get('crew_size', 'N/A')}\n- Tasks Completed: {agent_info.get('tasks_completed', 'N/A')}"

        references_str = final_result_dict.get("references", "")
        if references_str: # Ensure it's not empty
            references_str = f"\n\n---\n**Source Documents:**\n{references_str}"
        else: # Provide a default message if no references
            references_str = "\n\n---\n**Source Documents:**\nNo specific documents referenced for this query."


        model_used = final_result_dict.get('model_used', 'N/A')
        footer = f"\n\n<div style='font-size: 0.8em; color: #666; margin-top: 1em'>Model: {model_used.title()}</div>"
        
        final_formatted_message = f"### Response from {model_used.title()}\n\n{answer}{agent_info_str}{references_str}{footer}"
        yield final_formatted_message

    except Exception as e:
        logger.error(f"Error in chat response generation: {str(e)}", exc_info=True)
        yield f"### ⚠️ Error\n\nI encountered an error: {str(e)}\nPlease try again or check the application logs."
    finally:
        logger.info(f"Finished streaming response for query: {message}")


# Modified create_ui as per subtask
def create_ui():
    with gr.Blocks(
        title="Local Agentic RAG Assistant",
        theme=gr.themes.Soft(),
        css=".gradio-container {max-width: 1200px}",
    ) as app_ui: 
        gr.Markdown(
            """
            <div style="text-align: center; margin-bottom: 2rem">
                <h1 style="margin-bottom: 0.5rem">🤖 Local RAG Assistant</h1>
                <p style="color: #666">Powered by local LLMs and vector search</p>
            </div>
            """
        )

        with gr.Tabs() as tabs:
            with gr.Tab("💬 Chat", id="chat"):
                chatbot = gr.Chatbot(
                    label="Chat Window",
                    height=UI_CONFIG["chat_height"],
                    show_copy_button=True,
                    show_share_button=False,
                    avatar_images=(Path("assets/human.webp").as_posix(), Path("assets/ai.png").as_posix()),
                    bubble_full_width=False,
                )
                msg = gr.Textbox(
                    label="Your Message",
                    placeholder="Ask me anything about your documents...",
                    container=False,
                    lines=UI_CONFIG["textbox_lines"],
                    scale=7, # From original snippet
                )
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary", scale=2) 
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1) 

                def user_message_fn(message_text: str, history: list):
                    if message_text and message_text.strip(): 
                        return "", history + [{"role": "user", "content": message_text}]
                    return "", history 

                def bot_message_fn(history: list):
                    if not history or not history[-1]["content"] or history[-1]["role"] != "user":
                        # Yield original history if no action needed or if last message isn't user's.
                        # Gradio expects a generator, so ensure we yield something.
                        yield history
                        return

                    user_input = history[-1]["content"]
                    history.append({"role": "assistant", "content": ""}) 

                    for response_part in get_chat_response(user_input, history[:-1]): # Pass history without the current user message
                        history[-1]["content"] = response_part 
                        yield history 

                msg.submit(user_message_fn, [msg, chatbot], [msg, chatbot], queue=False).then(
                    bot_message_fn, chatbot, chatbot
                )
                submit_btn.click(user_message_fn, [msg, chatbot], [msg, chatbot], queue=False).then(
                    bot_message_fn, chatbot, chatbot
                )
                clear_btn.click(lambda: (None, []), None, [msg, chatbot], queue=False)


            with gr.Tab("📑 Documents", id="docs"):
                status_box = gr.Textbox(label="Status", interactive=False, visible=True)
                with gr.Row():
                    with gr.Column(scale=2):
                        upload_btn = gr.UploadButton(
                            "📤 Upload PDF",
                            file_types=UPLOAD_CONFIG["allowed_types"], # Ensure this is like ['.pdf']
                            variant="primary",
                        )
                    with gr.Column(scale=1):
                        refresh_btn = gr.Button("🔄 Refresh List", variant="secondary")
                
                doc_list_df = gr.Dataframe( 
                    label="Ingested Documents",
                    headers=["Filename", "Upload Date"],
                    datatype=["str", "str"],
                    interactive=False,
                    wrap=True,
                )
                
                def update_doc_list_ui():
                    try:
                        docs = get_ingested_docs() 
                        return [[d["filename"], d["timestamp"]] for d in docs]
                    except Exception as e:
                        logger.error(f"Error updating document list for UI: {str(e)}", exc_info=True)
                        return [] # Return empty list on error
                
                app_ui.load(update_doc_list_ui, None, [doc_list_df]) 
                refresh_btn.click(update_doc_list_ui, None, [doc_list_df])

                with gr.Row():
                    delete_input_tb = gr.Textbox( 
                        label="Enter filename to delete",
                        placeholder="example.pdf",
                        interactive=True,
                    )
                    delete_btn = gr.Button("🗑️ Delete", variant="secondary")

                def handle_upload_complete_ui(file_obj_or_list):
                    # Gradio upload can return a single FileData or a list of FileData
                    # We'll assume single file upload for simplicity as per original design
                    if not file_obj_or_list: return "No file provided for upload.", update_doc_list_ui()
                    
                    file_obj = file_obj_or_list
                    if isinstance(file_obj_or_list, list):
                        if not file_obj_or_list: return "No file provided for upload.", update_doc_list_ui()
                        file_obj = file_obj_or_list[0] # Take the first file if it's a list

                    status_message = process_upload(file_obj) # process_upload is top-level
                    
                    # Convert Gradio specific status (gr.Info etc.) to string for status_box
                    if hasattr(status_message, 'name') and status_message.name in ["info", "warning", "error"]:
                         status_text = str(status_message) 
                    else: # If it's already a string or other type
                        status_text = str(status_message)

                    return status_text, update_doc_list_ui()

                def handle_delete_ui(filename_to_delete: str): 
                    if not filename_to_delete: return "No filename entered.", update_doc_list_ui(), ""
                    status_message = delete_file(filename_to_delete) # delete_file is top-level
                    
                    if hasattr(status_message, 'name') and status_message.name in ["info", "warning", "error"]:
                        status_text = str(status_message)
                    else:
                        status_text = str(status_message)
                    return status_text, update_doc_list_ui(), "" 

                upload_btn.upload(
                    handle_upload_complete_ui,
                    inputs=[upload_btn],
                    outputs=[status_box, doc_list_df],
                )
                delete_btn.click(
                    handle_delete_ui,
                    inputs=[delete_input_tb],
                    outputs=[status_box, doc_list_df, delete_input_tb],
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
    return app_ui


if __name__ == "__main__":
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True) 
    
    app_instance = create_ui() 
    app_instance.launch(server_port=UI_CONFIG["port"], share=UI_CONFIG["share"])
