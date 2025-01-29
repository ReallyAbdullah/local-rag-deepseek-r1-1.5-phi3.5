# Local RAG Application 🧠📚

A privacy-focused Retrieval-Augmented Generation (RAG) system that runs entirely on your local machine. Built for researchers, students, and professionals who need document analysis without cloud dependencies.

![RAG Architecture Diagram](https://admin.bentoml.com/uploads/medium_simple_rag_workflow_091648ef39.png)

## Key Features ✨

- **Local First** - No data leaves your machine
- **Dual-Model System** - Smart routing between:
  - 🚀 Phi-3.5 (3.8B) for factual queries
  - 🧠 DeepSeek-R1 (1.5B) for complex reasoning
- **Document Management**:
  - PDF ingestion with text/OCR support
  - Vector storage using ChromaDB
  - Full document lifecycle management
- **Privacy Focused** - 100% offline operation
- **Responsive UI** - Gradio-based chat interface

## Tech Stack ⚙️

| Component           | Technology                        |
| ------------------- | --------------------------------- |
| Language Models     | Ollama (Phi-2, DeepSeek-R1)       |
| Embeddings          | Nomic Embed Text                  |
| Vector Store        | ChromaDB                          |
| Document Processing | LangChain, PyPDF, Unstructured.IO |
| UI Framework        | Gradio                            |
| OCR Engine          | Tesseract                         |

## Installation 🛠️

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) installed
- Tesseract OCR (for image-based PDFs):
  ```bash
  # MacOS
  brew install tesseract
  # Ubuntu
  sudo apt install tesseract-ocr
  ```

### Setup

1. Clone repository:

   ```bash
   git clone https://github.com/yourusername/local-rag.git
   cd local-rag
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull phi2
   ollama pull deepseek-r1:1.5b
   ```

## Usage 🖥️

1. Start the application:

   ```bash
   python app.py
   ```

2. Access the UI at `http://localhost:7860`

3. Workflow:

   - **Upload Documents**:

     - Supported format: PDF
     - Max size: 50MB
     - Both text-based and image PDFs supported

   - **Chat Interface**:

     - Ask natural language questions
     - Automatic model selection based on complexity
     - Source references with page numbers

   - **Document Management**:
     - View ingested documents
     - Delete documents and associated vectors

## Configuration ⚙️

Environment variables (`.env`):

```ini
# Optional advanced configuration
EMBEDDING_MODEL=nomic-embed-text
SIMPLE_MODEL=phi3.5
COMPLEX_MODEL=deepseek-r1:1.5b
CHUNK_SIZE=800
```

## Project Structure 📂

```bash
.
├── app.py               # Main application entry
├── rag/                 # Core RAG components
│   ├── chains.py        # Processing pipelines
│   ├── ingest.py        # Document ingestion logic
│   └── models.py        # Model management
├── data/                # Document storage
│   ├── uploaded/        # Original documents
│   └── ingested.json    # Ingestion registry
├── vector_store/        # ChromaDB storage
├── requirements.txt     # Dependencies
└── README.md            # This file
```

## License 📜

MIT License - See [LICENSE](LICENSE) for details

## Contributing 🤝

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## Acknowledgments 🙏

- Ollama team for local LLM management
- LangChain for RAG infrastructure
- ChromaDB for vector storage
- Gradio for UI framework

---

**Note**: This project is under active development. For feature requests or bug reports, please open an issue.
