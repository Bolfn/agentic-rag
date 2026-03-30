import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
IMPORT_DIR = PROJECT_ROOT / "imports"


def _resolve_import_dir() -> Path:
    configured_path = os.getenv("RAG_IMPORT_DIR")
    if configured_path:
        return Path(configured_path).expanduser()
    return IMPORT_DIR


SOURCE_DOCUMENTS_DIR = _resolve_import_dir()

EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"
GENERATION_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

ENCODING_NAME = "cl100k_base"
CHUNK_SIZE = 350
CHUNK_OVERLAP = 50
MIN_CHUNK_TOKENS = 90
MIN_PARAGRAPH_TOKENS = 50

COLLECTION_NAME = "agentic_rag_documents"
TOP_K = 4
MAX_CONTEXT_DISTANCE = 1.1
EMBEDDING_CANDIDATES = 12
BM25_CANDIDATES = 12
RERANK_CANDIDATES = 8

EXTRACTED_PATH = DATA_DIR / "extracted_pages.json"
CLEANED_PATH = DATA_DIR / "cleaned_pages.json"
CHUNKED_PATH = DATA_DIR / "chunked_documents.json"
EMBEDDED_PATH = DATA_DIR / "embedded_chunks.json"
CHROMA_DIR = DATA_DIR / "chroma_db"


def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
