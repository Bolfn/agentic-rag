import re

import tiktoken

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    ENCODING_NAME,
    MIN_CHUNK_TOKENS,
    MIN_PARAGRAPH_TOKENS,
)


def token_count(text: str, encoding: tiktoken.Encoding) -> int:
    return len(encoding.encode(text))


def split_by_separator(text: str, separator: str) -> list[str]:
    if separator == "":
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    return [part.strip() for part in text.split(separator) if part.strip()]


def recursive_split(
    text: str,
    encoding: tiktoken.Encoding,
    separators: list[str] | None = None,
) -> list[str]:
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    if token_count(text, encoding) <= CHUNK_SIZE:
        return [text.strip()]

    if not separators:
        return [text.strip()]

    separator = separators[0]
    parts = split_by_separator(text, separator)

    if len(parts) <= 1:
        return recursive_split(text, encoding, separators[1:])

    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = f"{current}{separator if current and separator else ''}{part}".strip()

        if token_count(candidate, encoding) <= CHUNK_SIZE:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if token_count(part, encoding) <= CHUNK_SIZE:
            current = part
            continue

        chunks.extend(recursive_split(part, encoding, separators[1:]))
        current = ""

    if current:
        chunks.append(current)

    return [chunk.strip() for chunk in chunks if chunk.strip()]


def add_overlap(chunks: list[str], encoding: tiktoken.Encoding) -> list[dict]:
    chunk_records = []

    for index, chunk in enumerate(chunks):
        tokens = encoding.encode(chunk)
        overlap_prefix = []

        if index > 0 and CHUNK_OVERLAP > 0:
            previous_tokens = encoding.encode(chunks[index - 1])
            overlap_prefix = previous_tokens[-CHUNK_OVERLAP:]

        combined_tokens = overlap_prefix + tokens
        text = encoding.decode(combined_tokens).strip()

        chunk_records.append(
            {
                "page_chunk_index": index + 1,
                "token_count": len(combined_tokens),
                "text": text,
            }
        )

    return chunk_records


def is_heading(paragraph: str) -> bool:
    text = paragraph.strip()
    if not text or "\n" in text:
        return False
    if len(text) > 90:
        return False
    if re.match(r"^(?:[a-z]\)|\d+\.|[ivxlcdm]+\.)\s+", text, flags=re.IGNORECASE):
        return False
    word_count = len(text.replace("/", " ").split())

    if text.endswith(":") and word_count <= 5:
        return True

    alpha_chars = [char for char in text if char.isalpha()]
    if not alpha_chars:
        return False

    uppercase_ratio = sum(1 for char in alpha_chars if char.isupper()) / len(alpha_chars)
    titleish = text == text.title() or uppercase_ratio >= 0.75
    sentence_like = any(mark in text for mark in [". ", "? ", "! "])

    return titleish and not sentence_like and word_count <= 8


def is_list_block(paragraph: str) -> bool:
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    if not lines:
        return False

    list_line_count = sum(
        1
        for line in lines
        if re.match(r"^(?:[a-z]\)|\d+\.|[ivxlcdm]+\.)\s+", line, flags=re.IGNORECASE)
        or line.startswith("•")
        or line.startswith("")
        or line.startswith("")
    )

    return list_line_count >= 1


def starts_with_heading(body_block: str) -> tuple[str | None, str]:
    parts = [part.strip() for part in body_block.split("\n\n") if part.strip()]
    if not parts:
        return None, ""

    first = parts[0]
    if is_heading(first):
        remainder = "\n\n".join(parts[1:]).strip()
        return first, remainder

    return None, body_block


def merge_small_paragraphs(paragraphs: list[str], encoding: tiktoken.Encoding) -> list[str]:
    if not paragraphs:
        return []

    merged: list[str] = []

    for paragraph in paragraphs:
        if not merged:
            merged.append(paragraph)
            continue

        if is_heading(paragraph):
            merged.append(paragraph)
            continue

        if is_list_block(paragraph):
            merged.append(paragraph)
            continue

        if token_count(paragraph, encoding) < MIN_PARAGRAPH_TOKENS and not is_heading(merged[-1]):
            candidate = f"{merged[-1]}\n\n{paragraph}".strip()
            if token_count(candidate, encoding) <= CHUNK_SIZE:
                merged[-1] = candidate
                continue

        merged.append(paragraph)

    return merged


def build_sections(paragraphs: list[str]) -> list[dict]:
    sections: list[dict] = []
    current_heading: str | None = None
    current_body: list[str] = []

    for paragraph in paragraphs:
        if is_heading(paragraph):
            if current_heading or current_body:
                sections.append(
                    {
                        "heading": current_heading,
                        "body": current_body[:],
                    }
                )
                current_body = []

            current_heading = paragraph
            continue

        current_body.append(paragraph)

    if current_heading or current_body:
        sections.append(
            {
                "heading": current_heading,
                "body": current_body[:],
            }
        )

    return sections


def split_section(section: dict, encoding: tiktoken.Encoding) -> list[str]:
    heading = section["heading"]
    body = section["body"]

    if not body:
        return []

    chunks: list[str] = []
    current = heading.strip() if heading else ""

    for block in body:
        nested_heading, nested_body = starts_with_heading(block)
        if nested_heading:
            if current and current.strip() != (heading.strip() if heading else ""):
                chunks.append(current)
            elif current and not nested_body:
                chunks.append(current)

            heading = nested_heading
            current = heading

            if not nested_body:
                continue

            block = nested_body

        candidate = f"{current}\n\n{block}".strip() if current else block

        if token_count(candidate, encoding) <= CHUNK_SIZE:
            current = candidate
            continue

        if current:
            chunks.append(current)

        block_with_heading = f"{heading}\n\n{block}".strip() if heading else block
        if token_count(block_with_heading, encoding) <= CHUNK_SIZE:
            current = block_with_heading
            continue

        subchunks = recursive_split(block_with_heading, encoding)
        chunks.extend(subchunks[:-1])
        current = subchunks[-1]

    if current:
        chunks.append(current)

    return chunks


def is_heading_only_chunk(text: str) -> bool:
    parts = [part.strip() for part in text.split("\n\n") if part.strip()]
    if len(parts) != 1:
        return False
    return is_heading(parts[0])


def chunk_pages(pages: list[dict]) -> list[dict]:
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    chunked_documents = []

    for page in pages:
        paragraphs = [part.strip() for part in page["text"].split("\n\n") if part.strip()]
        merged_paragraphs = merge_small_paragraphs(paragraphs, encoding)
        sections = build_sections(merged_paragraphs)

        page_chunks: list[str] = []
        for section in sections:
            section_chunks = split_section(section, encoding)
            section_records = add_overlap(section_chunks, encoding)
            page_chunks.extend(record["text"] for record in section_records)
        chunk_records = [
            {
                "page_chunk_index": index + 1,
                "token_count": token_count(text, encoding),
                "text": text,
            }
            for index, text in enumerate(page_chunks)
        ]

        for chunk in chunk_records:
            if is_heading_only_chunk(chunk["text"]):
                continue

            if chunk["token_count"] < MIN_CHUNK_TOKENS and chunked_documents:
                previous = chunked_documents[-1]
                if previous["page_number"] == page["page_number"]:
                    previous["text"] = f"{previous['text']}\n\n{chunk['text']}".strip()
                    previous["token_count"] = token_count(previous["text"], encoding)
                    continue

            chunked_documents.append(
                {
                    "chunk_id": f"chunk-{len(chunked_documents) + 1}",
                    "page_number": page["page_number"],
                    "page_chunk_index": chunk["page_chunk_index"],
                    "token_count": chunk["token_count"],
                    "text": chunk["text"],
                }
            )

    return chunked_documents


def run_chunking(pages: list[dict]) -> list[dict]:
    return chunk_pages(pages)
