from pathlib import Path

import fitz


def extract_pages(pdf_path: Path) -> list[dict]:
    """Extract raw page text from a PDF while preserving page numbers."""
    document = fitz.open(pdf_path)
    pages = []

    for page_index, page in enumerate(document, start=1):
        text = page.get_text("text")
        if not text or not text.strip():
            continue

        pages.append(
            {
                "page_number": page_index,
                "text": text,
            }
        )

    return pages


def run_extract(pdf_path: Path) -> list[dict]:
    return extract_pages(pdf_path)
