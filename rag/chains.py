# rag/chains.py (updated)
import logging
from pathlib import Path
from typing import List, Dict, Any, Callable # Added Callable for ProgressCallback typing
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage
from .models import LocalModels, ModelError
from config import VECTOR_STORE_DIR, RAG_CONFIG, MODEL_CONFIG
from .agents import RAGAgents, AgentError, ProgressCallback # Added ProgressCallback

# from autogen import UserProxyAgent
# from pathlib import Path
# from agents import PlanningAgent, TaskAutomator

# Configure logging
# logging.basicConfig( # BasicConfig should ideally be called only once at application entry point
#     level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )
logger = logging.getLogger(__name__) # Get logger instance


class RAGError(Exception):
    """Custom exception for RAG-related errors"""

    pass


class RAGChain:
    def __init__(self):
        try:
            self.models = LocalModels()
            self.agents = RAGAgents() # RAGAgents is initialized here

            # Initialize vector store
            self.vector_store = Chroma(
                persist_directory=str(VECTOR_STORE_DIR),
                collection_name="rag_collection",
                embedding_function=self.models.embeddings,
            )

            # Configure retriever
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": RAG_CONFIG["k_retrieval"]}
            )

            logger.info("RAG chain initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG chain: {str(e)}", exc_info=True)
            raise RAGError(f"RAG chain initialization failed: {str(e)}")

    # New method to set progress callback
    def set_progress_callback(self, callback: ProgressCallback):
        """Sets the progress callback for the RAGAgents instance."""
        if self.agents:
            self.agents.set_progress_callback(callback)
            logger.info("Progress callback set for RAGChain and propagated to RAGAgents.")
        else:
            logger.warning("Attempted to set progress callback, but RAGAgents not initialized.")

    def _format_references(self, docs: List[Dict]) -> str:
        """Format retrieved documents into reference string"""
        try:
            references = []
            for doc_obj in docs: # Changed variable name to avoid conflict with docs parameter name
                source = doc_obj.metadata.get("source", "unknown")
                page = doc_obj.metadata.get("page", "N/A")
                try:
                    filename = Path(source).name
                except TypeError: # Handle cases where source might not be a Path-like object
                    filename = str(source) 
                references.append(f"{filename} (Page {page})")
            # Return as a single string with each reference on a new line, prefixed by a dash
            return "\n".join([f"- {ref}" for ref in references]) if references else "No specific documents referenced."
        except Exception as e:
            logger.error(f"Error formatting references: {str(e)}", exc_info=True)
            return "Error retrieving references"

    def _route_query(self, query: str) -> str:
        """Route query to appropriate model based on complexity"""
        try:
            # Emit progress before routing - decided against this to reduce noise
            # if self.agents and hasattr(self.agents, '_emit_progress'):
            # self.agents._emit_progress("🚦 Routing query...")
            
            llm = self.models.get_llm("simple") # Uses simple model for routing
            prompt = f"""Classify this query as either 'simple' or 'complex' based on its intent and the information it seeks. 
            Simple queries are typically straightforward questions that can be answered directly from the text.
            Complex queries may require multi-step reasoning, synthesis of information from multiple parts of documents, or a deeper analysis.
            Respond only with the single word 'simple' or 'complex'. Query: {query}"""

            response = llm.invoke(prompt)
            response_text = (
                response.content if hasattr(response, 'content') else str(response) # AIMessage vs str
            )

            query_type = (
                "complex" if "complex" in response_text.strip().lower() else "simple"
            )
            logger.info(f"Query classified as: {query_type}")
            # if self.agents and hasattr(self.agents, '_emit_progress'):
            # self.agents._emit_progress(f"🚦 Query routed as: {query_type}")
            return query_type

        except Exception as e:
            logger.error(f"Query routing failed: {str(e)}", exc_info=True)
            # if self.agents and hasattr(self.agents, '_emit_progress'):
            # self.agents._emit_progress("🚦 Query routing failed, defaulting to simple.")
            return "simple"  # Default to simple model on error

    def invoke(self, query: str) -> dict:
        """Main RAG chain execution"""
        try:
            # Emit progress before document retrieval
            if self.agents and hasattr(self.agents, '_emit_progress') and self.agents.progress_callback:
                self.agents._emit_progress("🔍 Retrieving relevant documents...")
            else:
                logger.info("Progress callback not set, skipping 'Retrieving documents' message.")


            docs = self.retriever.invoke(query)
            if not docs:
                logger.warning("No relevant documents found for query")
                if self.agents and hasattr(self.agents, '_emit_progress') and self.agents.progress_callback:
                    self.agents._emit_progress("ℹ️ No relevant documents found.")
                return {
                    "answer": "I couldn't find any relevant information in the documents to answer your question.",
                    "references": "",
                    "model_used": "N/A", # No model used if no docs
                }

            # Route query to appropriate model
            model_type = self._route_query(query)
            logger.info(f"Query routed to model type: {model_type}")

            if model_type == "complex":
                try:
                    context = [
                        {"content": doc.page_content, "metadata": doc.metadata}
                        for doc in docs
                    ]
                    logger.info("Processing complex query with agents")
                    # Progress for agent processing will be handled by CustomCrewEventListener via RAGAgents
                    result = self.agents.process_query(query, context)
                    return {
                        "answer": result["answer"],
                        "references": self._format_references(docs),
                        "model_used": result["model_used"],
                        "agent_info": result.get("agent_info", {}),
                    }
                except AgentError as e:
                    logger.warning(
                        f"Agent processing failed, falling back to standard processing: {str(e)}", exc_info=True
                    )
                    if self.agents and hasattr(self.agents, '_emit_progress') and self.agents.progress_callback:
                        self.agents._emit_progress("⚠️ Agent processing failed. Falling back to simple response mode.")
                    model_type = "simple" # Fallback to simple

            # Standard processing for simple queries or fallback
            if self.agents and hasattr(self.agents, '_emit_progress') and self.agents.progress_callback:
                self.agents._emit_progress(f"🧠 Generating response using {model_type} model...")
            
            llm = self.models.get_llm(model_type)
            logger.info(f"Using model: {MODEL_CONFIG['llm'][model_type]}")
            context_str = "\n\n".join([d.page_content for d in docs])
            logger.info(f"Context length: {len(context_str)} characters")

            # Generate prompts based on model type
            # Emphasize using ONLY the provided context for all answers.
            if model_type == "complex": # Fallback from agent error
                prompt = f"""**Context:**\n{context_str}\n\n**Question:** {query}\n\nAnalyze the question based *only* on the provided context. Present your answer as:\n**Reasoning Process:** (brief step-by-step analysis based *only* on the context)\n**Final Answer:** (concise conclusion based *only* on the context)"""
                logger.info("Using complex prompt template for fallback.")
            else: # Simple query
                prompt = f"""Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer clearly and concisely based *only* on the provided context."""
                logger.info("Using simple prompt template.")

            logger.info("Requesting response from model...")
            
            answer = ""
            # Check if streaming is supported (method exists)
            if hasattr(llm, 'stream'):
                logger.info(f"Streaming response for {model_type} model...")
                full_answer_chunks = []
                for chunk in llm.stream(prompt):
                    # Chunks from OllamaLLM.stream are strings directly.
                    # If they were AIMessageChunk, it would be chunk.content
                    chunk_content = str(chunk) # Ensure it's a string
                    if self.agents and hasattr(self.agents, '_emit_progress') and self.agents.progress_callback:
                        self.agents._emit_progress(chunk_content)
                    full_answer_chunks.append(chunk_content)
                answer = "".join(full_answer_chunks)
            else:
                logger.info(f"Non-streaming response for {model_type} model.")
                response = llm.invoke(prompt)
                answer = (
                    response.content if hasattr(response, 'content') else str(response)
                )
                # If not streaming, send the whole answer as one progress update.
                if self.agents and hasattr(self.agents, '_emit_progress') and self.agents.progress_callback:
                    self.agents._emit_progress(answer)

            logger.info(f"Processed answer length: {len(answer)} characters")

            return {
                "answer": answer,
                "references": self._format_references(docs),
                "model_used": model_type,
            }

        except Exception as e:
            logger.error(f"Error in RAG chain: {str(e)}", exc_info=True)
            if self.agents and hasattr(self.agents, '_emit_progress') and self.agents.progress_callback:
                self.agents._emit_progress(f"❌ Error in RAG chain processing: {str(e)}")
            raise RAGError(f"Failed to process query: {str(e)}") # Re-raise to be caught by UI

    # def execute_query(self, query):
    #     try:
    #         # Step 1: Create plan
    #         plan = self.planner.create_plan(query)

    #         # Step 2: Execute plan
    #         result = self._execute_plan(plan, query)

    #         # Step 3: Save results
    #         save_result = self.automator.save_content(result["answer"])

    #         return {**result, "automation_result": save_result}
    #     except Exception as e:
    #         return {"error": str(e)}

    # def _execute_plan(self, plan, query):
    #     # Simplified execution flow
    #     docs = self.retriever.invoke(query)
    #     context = "\n\n".join([d.page_content for d in docs])

    #     # Get answer from LLM
    #     llm = self.models.get_llm("deepseek-r1:1.5b")
    #     answer = llm.invoke(f"Context: {context}\n\nQuestion: {query}")

    #     return {"answer": answer, "references": self._format_references(docs)}
