# rag/ingest.py (updated)
from datetime import datetime
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from .models import LocalModels
import hashlib
import json

INGESTED_DOCS = Path(__file__).parent.parent / "data" / "ingested.json"


def get_ingested_docs():
    """Load ingested documents registry"""
    if not INGESTED_DOCS.exists():
        return []
    with open(INGESTED_DOCS, "r") as f:
        return json.load(f)


def update_ingested_docs(docs):
    """Save ingested documents registry"""
    with open(INGESTED_DOCS, "w") as f:
        json.dump(docs, f)


def get_pdf_text(file_path: str) -> str:
    """Verify PDF contains extractable text"""
    try:
        from pypdf import PdfReader

        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            return "\n".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        print(f"⚠️ PDF text verification failed: {str(e)}")
        return ""


def smart_pdf_loader(file_path: str):
    """Intelligent PDF loader with fallback strategies"""
    abs_path = Path(file_path).resolve()
    print(f"\n🔍 Processing PDF: {abs_path.name}")

    # Strategy 1: Try standard text extraction
    try:
        loader = PyPDFLoader(str(abs_path))
        pages = loader.load_and_split()
        if pages and len(pages[0].page_content) > 100:
            print(f"✅ Using text-based loader for {abs_path.name}")
            return loader
    except Exception as e:
        print(f"⚠️ Standard PDF loader failed: {str(e)}")

    # Strategy 2: Try OCR-based extraction
    try:
        print(f"🔄 Attempting OCR for {abs_path.name}")
        loader = UnstructuredFileLoader(
            str(abs_path),
            strategy="ocr_only",
            ocr_languages="eng",
            post_processors=["remove_extra_whitespace"],
        )
        # Verify OCR content
        docs = loader.load()
        if docs and len(docs[0].page_content) > 100:
            print(f"✅ OCR successful for {abs_path.name}")
            return loader
    except Exception as e:
        print(f"⚠️ OCR failed: {str(e)}")

    # Strategy 3: Manual text fallback
    raw_text = get_pdf_text(str(abs_path))
    if len(raw_text) > 100:
        print(f"⚠️ Using raw text extraction fallback for {abs_path.name}")
        from langchain_community.docstore.document import Document

        return [Document(page_content=raw_text, metadata={"source": str(abs_path)})]

    raise ValueError(f"Failed to extract text from {abs_path.name}")


def ingest_pdf(file_path: str):
    try:
        abs_path = Path(file_path).resolve()
        vector_store_path = Path(__file__).parent.parent / "vector_store"

        # 1. Verify PDF exists
        if not abs_path.exists():
            print(f"🚨 File not found: {abs_path}")
            return False

        # 2. Load documents with diagnostics
        loader = smart_pdf_loader(str(abs_path))
        pages = loader.load()
        print(f"📄 Loaded {len(pages)} pages from PDF")

        # 3. Verify meaningful content
        if not pages or len(pages[0].page_content.strip()) < 50:
            print(f"🚨 No meaningful text in PDF: {abs_path.name}")
            return False

        # 4. Configure text splitter with validation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,
            add_start_index=True,
        )

        chunks = text_splitter.split_documents(pages)
        print(f"✂️ Split into {len(chunks)} chunks")
        print("Sample chunk:", chunks[0].page_content[:100] + "...")

        # 5. Verify embeddings
        models = LocalModels()
        test_embed = models.embeddings.embed_query("test embedding")
        print(f"🔢 Embedding test: Vector length {len(test_embed)}")

        # 6. Create vector store with verification
        vector_store_path.mkdir(parents=True, exist_ok=True)
        db = Chroma.from_documents(
            documents=chunks,
            embedding=models.embeddings,
            persist_directory=str(vector_store_path),
            collection_name="rag_collection",
        )

        # 7. Verify storage
        if db._collection.count() < 1:
            print(f"🚨 No vectors stored for {abs_path.name}")
            return False

        print(f"💾 Vector store updated: {db._collection.count()} vectors")

        # Record ingestion
        ingested = get_ingested_docs()
        ingested.append(
            {
                "filename": Path(file_path).name,
                "path": str(file_path),
                "timestamp": datetime.now().isoformat(),
            }
        )
        update_ingested_docs(ingested)

        return True

    except Exception as e:
        print(f"🔥 Critical ingestion error: {str(e)}")
        return False


def delete_document(filename: str):
    """Remove document and its vectors"""
    try:
        ingested = get_ingested_docs()
        doc = next((d for d in ingested if d["filename"] == filename), None)

        if doc:
            # Delete from vector store
            vector_store = Chroma(
                persist_directory="../vector_store",
                embedding_function=LocalModels().embeddings,
                collection_name="rag_collection",
            )
            vector_store.delete(ids=[str(doc["path"])])

            # Delete from registry
            ingested = [d for d in ingested if d["filename"] != filename]
            update_ingested_docs(ingested)

            # Delete original file
            Path(doc["path"]).unlink(missing_ok=True)

        return True
    except Exception as e:
        print(f"Deletion error: {str(e)}")
        return False
