# RAG Document Research & Theme Identification Chatbot

A sophisticated RAG (Retrieval-Augmented Generation) pipeline for document research and theme identification, built for the Wasserstoff AI Internship Project.

## 🎯 Project Overview

This system processes 75+ documents and creates an advanced chatbot that can:

- 📄 **Ingest Documents**: Process PDFs, scanned images, and text files with OCR
- 🔍 **Answer Questions**: Extract precise answers with accurate citations
- 🎨 **Identify Themes**: Analyze and synthesize common themes across documents
- 📊 **Provide Citations**: Return answers with page, paragraph, and sentence-level citations

## 🏗️ Architecture

```
Documents → OCR/Parsing → Chunking → Embeddings → Vector DB → Chatbot
```

### Core Components:
1. **OCR & Parsing** (`src/ocr_parsing.py`) - Extract text from documents
2. **Chunking** (`src/chunking.py`) - Break documents into searchable chunks
3. **Embeddings** (`src/embeddings.py`) - Create vector embeddings and store in Qdrant
4. **Pipeline** (`src/main_pipeline.py`) - Orchestrate the entire process

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone the repository
git clone <repository-url>
cd rag-document-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup project structure
python scripts/setup_environment.py
```

### 2. Configure Environment
Copy the `.env.example` to `.env` and update with your API keys:
```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Add Documents
Place your PDF documents in the `docs/` folder:
```bash
# Add 75+ PDF documents to the docs/ directory
cp /path/to/your/documents/*.pdf docs/
```

### 4. Run the Pipeline
```bash
# Run the complete pipeline
python src/main_pipeline.py

# Or run individual steps
python src/ocr_parsing.py      # Step 1: OCR & Parsing
python src/chunking.py         # Step 2: Chunking
python src/embeddings.py       # Step 3: Embeddings
```

## 📁 Project Structure

```
rag-document-chatbot/
├── .env                      # Environment configuration
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
│
├── config/
│   └── settings.py          # Configuration management
│
├── src/
│   ├── ocr_parsing.py       # Document processing & OCR
│   ├── chunking.py          # Text chunking
│   ├── embeddings.py        # Vector embeddings
│   ├── main_pipeline.py     # Pipeline orchestration
│   └── utils/               # Utility functions
│
├── docs/                    # Input PDF documents
├── data/                    # Processed data
│   ├── processed_documents/ # OCR outputs
│   ├── chunked_documents/   # Chunking outputs
│   └── embeddings_storage/  # Embeddings backup
│
├── tests/                   # Unit tests
├── notebooks/               # Jupyter notebooks
├── web_interface/           # Chatbot interface
└── scripts/                 # Utility scripts
```

## ⚙️ Configuration

The system uses environment variables for configuration:

### API Keys
- `OPENAI_API_KEY` - OpenAI API key for embeddings
- `GEMINI_API_KEY` - Google Gemini API key
- `QDRANT_API_KEY` - Qdrant Cloud API key

### Processing Settings
- `PDF_DIRECTORY` - Input documents folder
- `USE_OCR` - Enable OCR for scanned documents
- `MIN_PARAGRAPH_LENGTH` - Minimum paragraph length
- `BATCH_SIZE` - Processing batch size

### Vector Database
- `QDRANT_URL` - Qdrant server URL
- `COLLECTION_NAME` - Vector collection name
- `EMBEDDING_MODEL` - Embedding model to use

## 📊 Pipeline Stages

### Stage 1: OCR & Document Parsing
- Extracts text from PDFs using PyMuPDF
- Performs OCR on scanned documents with Tesseract
- Enhances image quality for better OCR results
- Splits content into paragraphs with filtering

### Stage 2: Document Chunking
- Creates chunks with metadata structure:
  ```json
  {
    "doc_id": "document-uuid",
    "page": 1,
    "para_id": 2,
    "text": "paragraph content",
    "keywords": ["key", "terms"],
    "cluster_id": 1
  }
  ```
- Extracts relevant keywords
- Performs topic clustering

### Stage 3: Vector Embeddings
- Creates semantic embeddings using SentenceTransformers or OpenAI
- Stores vectors in Qdrant with metadata
- Enables semantic search capabilities

## 🔍 Usage Examples

### Search Documents
```python
from src.embeddings import EmbeddingProcessor

processor = EmbeddingProcessor()
results = processor.search_similar_chunks("regulatory compliance", limit=5)

for result in results:
    print(f"Document: {result['doc_name']}")
    print(f"Page: {result['page']}, Para: {result['para_id']}")
    print(f"Text: {result['text']}")
    print(f"Score: {result['score']}")
```

### Process New Documents
```python
from src.ocr_parsing import DocumentProcessor

processor = DocumentProcessor()
docs = processor.process_document_folder()
print(f"Processed {len(docs)} documents")
```

## 🧪 Testing

Run tests to validate the pipeline:
```bash
# Run all tests
python -m pytest tests/

# Test individual components
python -m pytest tests/test_ocr.py
python -m pytest tests/test_chunking.py
python -m pytest tests/test_embeddings.py
```

## 📈 Performance Monitoring

The system provides comprehensive logging and monitoring:
- Processing logs in `pipeline.log`
- Summary files for each stage
- Performance metrics and statistics
- Error handling and retry logic


- Paragraph/sentence-level citations
- Filtering and sorting capabilities
- Visual citation mapping
- Document inclusion/exclusion controls
