# rag/models.py
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama


class LocalModels:
    def __init__(self):
        try:
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
            # Test embedding
            test_embed = self.embeddings.embed_query("test")
            if len(test_embed) < 10:
                raise ValueError("Invalid embeddings received")
        except Exception as e:
            print(f"🚨 Embedding Error: {str(e)}")
            raise

    def get_llm(self, model_name):
        return ChatOllama(model=model_name)
