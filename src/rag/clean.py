import re


PAGE_MARKER_PATTERN = re.compile(r"Page\s+\d+\s+of\s+\d+", re.IGNORECASE)
MANUAL_TITLE_PATTERN = re.compile(
    r"Sample\s+Policies\s+and\s+Procedures\s+Manual(?:\s+v\.?\s*\d+(?:\.\d+)?)?",
    re.IGNORECASE,
)
VERSION_PATTERN = re.compile(r"MANUAL\s+v\.?\s*\d+(?:\.\d+)?", re.IGNORECASE)
DOT_LEADER_PATTERN = re.compile(r"\.{4,}")
MULTISPACE_PATTERN = re.compile(r"[ \t]{2,}")
MULTIBLANK_PATTERN = re.compile(r"\n{3,}")


def normalize_line(line: str) -> str:
    return MULTISPACE_PATTERN.sub(" ", line.strip())


def is_heading_like_line(line: str) -> bool:
    if not line:
        return False
    if len(line) > 90:
        return False
    if re.match(r"^[a-z0-9]", line):
        return False
    if re.match(r"^(?:[a-z]\)|\d+\.|[ivxlcdm]+\.)\s+", line, flags=re.IGNORECASE):
        return False
    word_count = len(line.replace("/", " ").split())

    if line.endswith(":") and word_count <= 5:
        return True

    alpha_chars = [char for char in line if char.isalpha()]
    if not alpha_chars:
        return False

    uppercase_ratio = sum(1 for char in alpha_chars if char.isupper()) / len(alpha_chars)
    titleish = line == line.title() or uppercase_ratio >= 0.75
    sentence_like = any(mark in line for mark in [". ", "? ", "! "])

    return titleish and not sentence_like and word_count <= 8


def is_list_item_line(line: str) -> bool:
    return bool(
        re.match(r"^(?:[a-z]\)|\d+\.|[ivxlcdm]+\.)\s+", line, flags=re.IGNORECASE)
        or line.startswith("•")
        or line.startswith("")
        or line.startswith("")
    )


def is_noise_line(line: str) -> bool:
    if not line:
        return True
    if PAGE_MARKER_PATTERN.fullmatch(line):
        return True
    if MANUAL_TITLE_PATTERN.fullmatch(line):
        return True
    if VERSION_PATTERN.fullmatch(line):
        return True
    if re.fullmatch(r"\d+\s+of\s+\d+", line, flags=re.IGNORECASE):
        return True
    return False


def rebuild_paragraphs(lines: list[str]) -> str:
    paragraphs: list[str] = []
    current: list[str] = []

    for line in lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current).strip())
                current = []
            continue

        if is_heading_like_line(line):
            if current:
                paragraphs.append(" ".join(current).strip())
                current = []
            paragraphs.append(line)
            continue

        if is_list_item_line(line):
            if current and not is_list_item_line(current[0]):
                paragraphs.append(" ".join(current).strip())
                current = []
            current.append(line)
            continue

        if current and is_list_item_line(current[0]):
            paragraphs.append(" ".join(current).strip())
            current = []

        current.append(line)

    if current:
        paragraphs.append(" ".join(current).strip())

    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def is_toc_page(text: str) -> bool:
    if "table of contents" in text.lower():
        return True

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    dot_leader_lines = sum(1 for line in lines if DOT_LEADER_PATTERN.search(line))
    numbered_tail_lines = sum(
        1 for line in lines if re.search(r"(?:\.{2,}|\s)\d{1,3}\s*$", line)
    )
    return dot_leader_lines >= 5 or numbered_tail_lines >= 8


def is_table_like_page(text: str) -> bool:
    lowered = text.lower()
    if "business record retention schedule" in lowered:
        return True

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        return False

    short_lines = sum(1 for line in lines if len(line) <= 24)
    numeric_or_code_lines = sum(
        1
        for line in lines
        if re.fullmatch(r"(?:\d+|p|term|corporate|personnel|taxation|accounting and fiscal)", line.lower())
    )
    return short_lines >= 12 and numeric_or_code_lines >= 4


def should_drop_page(page_number: int, text: str) -> bool:
    lowered = text.lower()

    if page_number == 1 and "sample cdc" in lowered:
        return True
    if is_toc_page(text):
        return True
    if is_table_like_page(text):
        return True

    return False


def should_drop_page_record(page: dict, cleaned_text: str) -> bool:
    raw_text = page["text"]

    if should_drop_page(page["page_number"], raw_text):
        return True
    if should_drop_page(page["page_number"], cleaned_text):
        return True

    return False


def clean_page_text(text: str) -> str:
    cleaned_lines = []

    for raw_line in text.splitlines():
        line = normalize_line(raw_line)
        if is_noise_line(line):
            continue
        cleaned_lines.append(line)

    cleaned_text = rebuild_paragraphs(cleaned_lines)
    cleaned_text = PAGE_MARKER_PATTERN.sub(" ", cleaned_text)
    cleaned_text = MANUAL_TITLE_PATTERN.sub(" ", cleaned_text)
    cleaned_text = VERSION_PATTERN.sub(" ", cleaned_text)
    cleaned_text = MULTIBLANK_PATTERN.sub("\n\n", cleaned_text)
    return cleaned_text.strip()


def clean_pages(pages: list[dict]) -> list[dict]:
    cleaned_records = []

    for page in pages:
        cleaned_text = clean_page_text(page["text"])
        if not cleaned_text:
            continue
        if should_drop_page_record(page, cleaned_text):
            continue

        cleaned_records.append(
            {
                "page_number": page["page_number"],
                "text": cleaned_text,
            }
        )

    return cleaned_records


def run_clean(pages: list[dict]) -> list[dict]:
    return clean_pages(pages)
