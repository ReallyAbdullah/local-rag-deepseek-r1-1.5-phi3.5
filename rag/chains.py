# rag/chains.py (updated)
import logging
from pathlib import Path
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage
from .models import LocalModels, ModelError
from config import VECTOR_STORE_DIR, RAG_CONFIG, MODEL_CONFIG
from .agents import RAGAgents, AgentError

# from autogen import UserProxyAgent
# from pathlib import Path
# from agents import PlanningAgent, TaskAutomator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RAGError(Exception):
    """Custom exception for RAG-related errors"""

    pass


class RAGChain:
    def __init__(self):
        try:
            self.models = LocalModels()
            self.agents = RAGAgents()

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
            logger.error(f"Failed to initialize RAG chain: {str(e)}")
            raise RAGError(f"RAG chain initialization failed: {str(e)}")

    def _format_references(self, docs: List[Dict]) -> str:
        """Format retrieved documents into reference string"""
        try:
            references = []
            for doc in docs:
                source = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "N/A")
                try:
                    filename = Path(source).name
                except TypeError:
                    filename = str(source)
                references.append(f"{filename} (Page {page})")
            return "\n".join([f"- {ref}" for ref in references])
        except Exception as e:
            logger.error(f"Error formatting references: {str(e)}")
            return "Error retrieving references"

    def _route_query(self, query: str) -> str:
        """Route query to appropriate model based on complexity"""
        try:
            llm = self.models.get_llm("simple")
            prompt = f"""Classify this query as either 'simple' or 'complex'. 
            Respond only with the single word. Query: {query}"""

            response = llm.invoke(prompt)
            response_text = (
                response.content if isinstance(response, AIMessage) else str(response)
            )

            query_type = (
                "complex" if "complex" in response_text.strip().lower() else "simple"
            )
            logger.info(f"Query classified as: {query_type}")
            return query_type

        except Exception as e:
            logger.error(f"Query routing failed: {str(e)}")
            return "simple"  # Default to simple model on error

    def invoke(self, query: str) -> dict:
        """Main RAG chain execution"""
        try:
            # Get relevant documents
            docs = self.retriever.invoke(query)
            if not docs:
                logger.warning("No relevant documents found for query")
                return {
                    "answer": "I couldn't find any relevant information in the documents to answer your question.",
                    "references": "",
                    "model_used": "simple",
                }

            # Route query to appropriate model
            model_type = self._route_query(query)
            logger.info(f"Query routed to model type: {model_type}")

            # For complex queries, use agent-based processing
            if model_type == "complex":
                try:
                    # Convert documents to context format
                    context = [
                        {"content": doc.page_content, "metadata": doc.metadata}
                        for doc in docs
                    ]

                    # Process with agents
                    logger.info("Processing complex query with agents")
                    result = self.agents.process_query(query, context)

                    return {
                        "answer": result["answer"],
                        "references": self._format_references(docs),
                        "model_used": result["model_used"],
                        "agent_info": result.get("agent_info", {}),
                    }

                except AgentError as e:
                    logger.warning(
                        f"Agent processing failed, falling back to standard processing: {str(e)}"
                    )
                    # Fall back to standard processing
                    model_type = "simple"

            # Standard processing for simple queries or fallback
            llm = self.models.get_llm(model_type)
            logger.info(f"Using model: {MODEL_CONFIG['llm'][model_type]}")

            # Prepare context
            context = "\n\n".join([d.page_content for d in docs])
            logger.info(f"Context length: {len(context)} characters")

            # Generate prompts based on model type
            if model_type == "complex":
                prompt = f"""**Context:** {context}
                
                **Question:** {query}
                
                Analyze step-by-step considering:
                1. Document evidence
                2. Logical connections
                3. Potential implications
                
                Present your answer as:
                **Reasoning Process:**
                - [Step-by-step analysis]
                
                **Final Answer:**
                - [Concise conclusion]"""
                logger.info("Using complex prompt template")
            else:
                prompt = f"""Context: {context}
                
                Question: {query}
                Answer clearly and concisely."""
                logger.info("Using simple prompt template")

            # Get response
            logger.info("Requesting response from model...")
            response = llm.invoke(prompt)
            logger.info(f"Raw response type: {type(response)}")
            logger.info(f"Raw response: {response}")

            answer = (
                response.content if isinstance(response, AIMessage) else str(response)
            )
            logger.info(f"Processed answer length: {len(answer)} characters")

            return {
                "answer": answer,
                "references": self._format_references(docs),
                "model_used": model_type,
            }

        except Exception as e:
            logger.error(f"Error in RAG chain: {str(e)}")
            raise RAGError(f"Failed to process query: {str(e)}")

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
