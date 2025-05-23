from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
UPLOAD_DIR = DATA_DIR / "uploaded"
INGESTED_DOCS_FILE = DATA_DIR / "ingested.json"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# Model configurations
MODEL_CONFIG = {
    "embeddings": {"model": "nomic-embed-text", "dimension": 768},
    "llm": {"simple": "phi3.5", "complex": "deepseek-r1:1.5b"},
}

# RAG configurations
RAG_CONFIG = {"chunk_size": 800, "chunk_overlap": 100, "k_retrieval": 3}

# File upload configurations
UPLOAD_CONFIG = {"max_file_size": 50 * 1024 * 1024, "allowed_types": [".pdf"]}  # 50MB

# UI configurations
UI_CONFIG = {
    "theme": "soft",
    "port": 7860,
    "share": False,
    "chat_height": 800,
    "textbox_lines": 3,
}
