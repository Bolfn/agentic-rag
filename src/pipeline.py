"""Pipeline entrypoint for the first working pass of the agentic RAG rewrite."""

import json

from rag.chunking import run_chunking
from rag.clean import run_clean
from rag.config import (
    CHUNKED_PATH,
    CHROMA_DIR,
    CLEANED_PATH,
    EMBEDDED_PATH,
    EXTRACTED_PATH,
    SOURCE_DOCUMENTS_DIR,
    ensure_data_dirs,
)
from rag.embed import run_embed
from rag.index import run_index
from rag.loaders import run_extract


def write_json(path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ensure_data_dirs()

    extracted_pages = run_extract(SOURCE_DOCUMENTS_DIR)
    write_json(EXTRACTED_PATH, extracted_pages)

    cleaned_pages = run_clean(extracted_pages)
    write_json(CLEANED_PATH, cleaned_pages)

    chunked_documents = run_chunking(cleaned_pages)
    write_json(CHUNKED_PATH, chunked_documents)

    embedded_chunks = run_embed(chunked_documents)
    write_json(EMBEDDED_PATH, embedded_chunks)

    run_index(embedded_chunks)

    print("Agentic RAG base pipeline finished.")
    print(f"Extracted pages: {len(extracted_pages)}")
    print(f"Cleaned pages: {len(cleaned_pages)}")
    print(f"Chunked documents: {len(chunked_documents)}")
    print(f"Embedded chunks: {len(embedded_chunks)}")
    print(f"Artifacts written to: {EXTRACTED_PATH.parent}")
    print(f"Chroma index written to: {CHROMA_DIR}")
    print(f"Imported PDFs from: {SOURCE_DOCUMENTS_DIR}")


if __name__ == "__main__":
    main()
