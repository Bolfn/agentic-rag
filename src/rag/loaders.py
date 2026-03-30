from pathlib import Path

import fitz


def list_pdf_documents(source_dir: Path) -> list[Path]:
    return sorted(path for path in source_dir.rglob("*.pdf") if path.is_file())


def extract_pages(pdf_path: Path, document_id: str, relative_path: str) -> list[dict]:
    """Extract raw page text from a PDF while preserving document and page metadata."""
    document = fitz.open(pdf_path)
    pages = []

    for page_index, page in enumerate(document, start=1):
        text = page.get_text("text")
        if not text or not text.strip():
            continue

        pages.append(
            {
                "document_id": document_id,
                "document_name": pdf_path.name,
                "document_path": relative_path,
                "page_number": page_index,
                "text": text,
            }
        )

    return pages


def extract_all_pages(source_dir: Path) -> list[dict]:
    pdf_paths = list_pdf_documents(source_dir)
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in import directory: {source_dir}"
        )

    pages: list[dict] = []
    for index, pdf_path in enumerate(pdf_paths, start=1):
        document_id = f"doc-{index}"
        relative_path = str(pdf_path.relative_to(source_dir))
        pages.extend(extract_pages(pdf_path, document_id, relative_path))

    return pages


def run_extract(source_dir: Path) -> list[dict]:
    return extract_all_pages(source_dir)
