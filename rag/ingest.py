# rag/ingest.py (updated)
from datetime import datetime
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from .models import LocalModels, ModelError
from config import DATA_DIR, VECTOR_STORE_DIR, RAG_CONFIG
import hashlib
import json
import logging
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IngestError(Exception):
    """Custom exception for ingestion-related errors"""

    pass


INGESTED_DOCS = DATA_DIR / "ingested.json"


def get_ingested_docs():
    """Load ingested documents registry"""
    try:
        if not INGESTED_DOCS.exists():
            # Initialize with empty array if file doesn't exist
            INGESTED_DOCS.parent.mkdir(parents=True, exist_ok=True)
            with open(INGESTED_DOCS, "w") as f:
                json.dump([], f)
            return []

        with open(INGESTED_DOCS, "r") as f:
            content = f.read().strip()
            if not content:  # Handle empty file
                return []
            return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Corrupted ingested docs registry: {str(e)}")
        # Backup corrupted file and create new one
        backup_path = INGESTED_DOCS.with_suffix(".json.bak")
        if INGESTED_DOCS.exists():
            shutil.copy2(INGESTED_DOCS, backup_path)
            logger.info(f"Backed up corrupted registry to {backup_path}")
        with open(INGESTED_DOCS, "w") as f:
            json.dump([], f)
        return []
    except Exception as e:
        logger.error(f"Error loading ingested docs registry: {str(e)}")
        return []


def update_ingested_docs(docs):
    """Save ingested documents registry"""
    try:
        # Ensure parent directory exists
        INGESTED_DOCS.parent.mkdir(parents=True, exist_ok=True)

        # Pretty print JSON for better readability
        with open(INGESTED_DOCS, "w") as f:
            json.dump(docs, f, indent=2, sort_keys=True)
        logger.info("Updated ingested documents registry")
    except Exception as e:
        logger.error(f"Error updating ingested docs registry: {str(e)}")
        raise IngestError(f"Failed to update document registry: {str(e)}")


def get_pdf_text(file_path: str) -> str:
    """Verify PDF contains extractable text"""
    try:
        from pypdf import PdfReader

        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            if not text.strip():
                logger.warning(f"No text extracted from {file_path}")
            return text
    except Exception as e:
        logger.error(f"PDF text extraction failed: {str(e)}")
        return ""


def smart_pdf_loader(file_path: str):
    """Intelligent PDF loader with fallback strategies"""
    abs_path = Path(file_path).resolve()
    logger.info(f"Processing PDF: {abs_path.name}")

    # Strategy 1: Try standard text extraction with error suppression
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            loader = PyPDFLoader(str(abs_path))
            pages = loader.load_and_split()
            if pages and len(pages[0].page_content.strip()) > 100:
                logger.info(f"Using text-based loader for {abs_path.name}")
                return loader
    except Exception as e:
        logger.warning(f"Standard PDF loader failed: {str(e)}")

    # Strategy 2: Try OCR-based extraction
    try:
        logger.info(f"Attempting OCR for {abs_path.name}")
        loader = UnstructuredFileLoader(
            str(abs_path),
            strategy="ocr_only",
            ocr_languages="eng",
            post_processors=["remove_extra_whitespace", "remove_empty_lines"],
        )
        # Verify OCR content
        docs = loader.load()
        if docs and len(docs[0].page_content.strip()) > 100:
            logger.info(f"OCR successful for {abs_path.name}")
            return loader
    except Exception as e:
        logger.warning(f"OCR failed: {str(e)}")

    # Strategy 3: Manual text fallback with additional cleaning
    raw_text = get_pdf_text(str(abs_path))
    if len(raw_text.strip()) > 100:
        logger.info(f"Using raw text extraction fallback for {abs_path.name}")
        from langchain_community.docstore.document import Document

        # Clean the text
        clean_text = "\n".join(
            line.strip() for line in raw_text.splitlines() if line.strip()
        )
        return [Document(page_content=clean_text, metadata={"source": str(abs_path)})]

    error_msg = f"Failed to extract text from {abs_path.name}"
    logger.error(error_msg)
    raise IngestError(error_msg)


def ingest_pdf(file_path: str, filename: str = None):
    """Ingest PDF document into vector store"""
    try:
        abs_path = Path(file_path).resolve()
        doc_filename = filename or abs_path.name

        # 1. Verify PDF exists
        if not abs_path.exists():
            raise IngestError(f"File not found: {abs_path}")

        # 2. Load documents with diagnostics
        loader = smart_pdf_loader(str(abs_path))
        pages = loader.load()
        logger.info(f"Loaded {len(pages)} pages from {doc_filename}")

        # 3. Verify meaningful content
        if not pages or len(pages[0].page_content.strip()) < 50:
            raise IngestError(f"No meaningful text in PDF: {doc_filename}")

        # 4. Configure text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAG_CONFIG["chunk_size"],
            chunk_overlap=RAG_CONFIG["chunk_overlap"],
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,
            add_start_index=True,
        )

        chunks = text_splitter.split_documents(pages)
        logger.info(f"Split into {len(chunks)} chunks")

        # 5. Initialize models and verify embeddings
        models = LocalModels()
        test_embed = models.embeddings.embed_query("test embedding")
        if len(test_embed) < 100:
            raise IngestError("Invalid embedding dimension")

        # 6. Create vector store
        db = Chroma.from_documents(
            documents=chunks,
            embedding=models.embeddings,
            persist_directory=str(VECTOR_STORE_DIR),
            collection_name="rag_collection",
        )

        # 7. Verify storage
        if db._collection.count() < 1:
            raise IngestError(f"No vectors stored for {doc_filename}")

        logger.info(f"Vector store updated: {db._collection.count()} vectors")

        # Record ingestion
        ingested = get_ingested_docs()
        ingested.append(
            {
                "filename": Path(file_path).name,
                "path": str(file_path),
                "timestamp": datetime.now().isoformat(),
                "chunks": len(chunks),
            }
        )
        update_ingested_docs(ingested)

        return True

    except Exception as e:
        logger.error(f"Critical ingestion error: {str(e)}")
        raise IngestError(f"Failed to ingest document: {str(e)}")


def delete_document(filename: str):
    """Remove document and its vectors"""
    try:
        ingested = get_ingested_docs()
        doc = next((d for d in ingested if d["filename"] == filename), None)

        if not doc:
            logger.warning(f"Document not found in registry: {filename}")
            return False

        # Delete from vector store
        vector_store = Chroma(
            persist_directory=str(VECTOR_STORE_DIR),
            embedding_function=LocalModels().embeddings,
            collection_name="rag_collection",
        )
        vector_store.delete(where={"source": str(doc["path"])})
        logger.info(f"Removed vectors for {filename}")

        # Delete from registry
        ingested = [d for d in ingested if d["filename"] != filename]
        update_ingested_docs(ingested)
        logger.info(f"Removed {filename} from registry")

        # Delete original file
        file_path = Path(doc["path"])
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted file: {filename}")

        return True

    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise IngestError(f"Failed to delete document: {str(e)}")
