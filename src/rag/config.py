from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_DOCUMENT = PROJECT_ROOT / "sample_policy_and_procedures_manual.pdf"

EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"
GENERATION_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

ENCODING_NAME = "cl100k_base"
CHUNK_SIZE = 350
CHUNK_OVERLAP = 70
MIN_CHUNK_TOKENS = 120
MIN_PARAGRAPH_TOKENS = 80

COLLECTION_NAME = "policy_manual_agentic"
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
