# rag/models.py
import logging
from langchain_ollama import OllamaEmbeddings, ChatOllama
from config import MODEL_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelError(Exception):
    """Custom exception for model-related errors"""

    pass


class LocalModels:
    def __init__(self):
        self._init_embeddings()
        # Remove immediate verification to speed up startup
        self.models = {}

    def _init_embeddings(self):
        """Initialize embedding model"""
        try:
            self.embeddings = OllamaEmbeddings(
                model=MODEL_CONFIG["embeddings"]["model"]
            )
            logger.info("Embedding model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise ModelError(f"Embedding model initialization failed: {str(e)}")

    def get_llm(self, model_type: str = "simple"):
        """Get LLM model based on type (simple/complex) with lazy loading"""
        try:
            if model_type not in self.models:
                model_name = MODEL_CONFIG["llm"].get(model_type)
                if not model_name:
                    raise ModelError(f"Invalid model type: {model_type}")
                self.models[model_type] = ChatOllama(model=model_name)
                logger.info(f"Initialized {model_type} model: {model_name}")
            return self.models[model_type]
        except Exception as e:
            logger.error(f"Failed to get LLM model: {str(e)}")
            raise ModelError(f"Failed to get LLM model: {str(e)}")
