# rag/chains.py (updated)
from pathlib import Path
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage
from .models import LocalModels


class RAGChain:
    def __init__(self):
        self.models = LocalModels()
        vector_store_path = Path(__file__).parent.parent / "vector_store"

        # Ensure persistence happens immediately
        self.vector_store = Chroma(
            persist_directory=str(vector_store_path),
            collection_name="rag_collection",
            embedding_function=self.models.embeddings,
        )

        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

    def _format_references(self, docs: List[Dict]) -> str:
        """Format retrieved documents into reference string"""
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

    def _route_query(self, query: str) -> str:
        """Route query to appropriate model"""
        llm = self.models.get_llm("phi3.5")
        prompt = f"""Classify this query as either 'simple' or 'complex'. 
        Respond only with the single word. Query: {query}"""

        # Handle both string and AIMessage responses
        response = llm.invoke(prompt)
        if isinstance(response, AIMessage):
            response_text = response.content.strip().lower()
        else:
            response_text = str(response).strip().lower()

        return "complex" if "complex" in response_text else "simple"

    def invoke(self, query: str) -> dict:
        """Main RAG chain execution"""
        docs = self.retriever.invoke(query)

        model_type = self._route_query(query)
        llm = self.models.get_llm(
            "deepseek-r1:1.5b" if model_type == "complex" else "phi3.5"
        )

        context = "\n\n".join([d.page_content for d in docs])

        # Different prompts for different models
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
        else:
            prompt = f"""Context: {context}
            
            Question: {query}
            Answer clearly and concisely."""

        response = llm.invoke(prompt)
        answer = response.content if isinstance(response, AIMessage) else str(response)

        return {
            "answer": answer,
            "references": self._format_references(docs),
            "model_used": model_type,
        }
