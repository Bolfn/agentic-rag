import io
import os
import warnings
from contextlib import redirect_stderr, redirect_stdout
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import logging as transformers_logging

from .config import GENERATION_MODEL_NAME

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

MAX_INPUT_TOKENS = 2048
MAX_NEW_TOKENS = 256
HOLIDAY_MARKERS = [
    "new year",
    "martin luther king",
    "president",
    "memorial",
    "independence",
    "labor day",
    "columbus",
    "veteran",
    "thanksgiving",
    "christmas",
]

transformers_logging.set_verbosity_error()
warnings.filterwarnings(
    "ignore",
    message=r".*_check_is_size will be removed.*",
    category=FutureWarning,
)


def _run_quietly(func):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return func()


def _resolve_local_model_path(model_name: str) -> str:
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = cache_dir / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
        if snapshots:
            return str(snapshots[-1])
    return model_name


@lru_cache(maxsize=1)
def load_generation_model():
    """Load the agreed local instruct model."""
    model_path = _resolve_local_model_path(GENERATION_MODEL_NAME)
    tokenizer = _run_quietly(
        lambda: AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    )
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = _run_quietly(
        lambda: AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            device_map="auto",
            quantization_config=quantization_config,
        )
    )
    return tokenizer, model, GENERATION_MODEL_NAME


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk.get("metadata", {})
        page_number = metadata.get("page_number", chunk.get("page_number", "unknown"))
        chunk_index = metadata.get(
            "page_chunk_index",
            chunk.get("page_chunk_index", chunk.get("chunk_index", "unknown")),
        )
        context_parts.append(
            "\n".join(
                [
                    f"Context {index}",
                    f"Page: {page_number}",
                    f"Chunk: {chunk_index}",
                    chunk.get("text", ""),
                ]
            )
        )

    context = "\n\n".join(context_parts)

    return "\n\n".join(
        [
            "Answer the user's question using only the provided context.",
            "If the answer is not clearly supported by the context, say that the answer was not found in the retrieved context.",
            "Prefer concise factual answers. When the context contains policy conditions or lists, include those details.",
            context,
            f"Question: {question}",
            "Answer:",
        ]
    )


def _is_heading_like(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 70:
        return False
    if stripped.endswith((".", ";", ":")):
        return False
    normalized = (
        stripped.replace("(", " ")
        .replace(")", " ")
        .replace("/", " ")
        .replace("-", " ")
    )
    words = normalized.split()
    if len(words) > 8:
        return False
    alpha_words = [word for word in words if any(char.isalpha() for char in word)]
    if not alpha_words:
        return False
    return all(word[:1].isupper() for word in alpha_words if word[:1].isalpha())


def _extract_relevant_passage(question: str, text: str) -> str:
    question_lower = question.lower()
    text = text.strip()

    if "holiday" in question_lower:
        start_index = text.lower().find("holidays")
        if start_index != -1:
            candidate = text[start_index:]
        else:
            candidate = text

        for marker in ["Leave of Absence", "Personal Sick Leave", "Bereavement", "Vacation Days"]:
            marker_index = candidate.find(marker)
            if marker_index > 0:
                candidate = candidate[:marker_index].strip()
                break

        lines = [line.strip() for line in candidate.splitlines() if line.strip()]
        kept: list[str] = []
        for index, line in enumerate(lines):
            if index > 0 and _is_heading_like(line):
                break
            kept.append(line)
        clipped = "\n".join(kept).strip()
        return clipped or candidate[:900]

    return text[:900]


def extractive_fallback(question: str, retrieved_chunks: list[dict]) -> str:
    if not retrieved_chunks:
        return "I could not find a sufficiently relevant passage in the retrieved context."

    top_chunk = _extract_relevant_passage(question, retrieved_chunks[0]["text"])
    if len(top_chunk) > 900:
        top_chunk = top_chunk[:897] + "..."

    return "\n".join(
        [
            "I could not generate a full grounded answer with the local generation model.",
            "Most relevant retrieved passage:",
            top_chunk,
        ]
    )


def generate_answer(question: str, retrieved_chunks: list[dict]) -> tuple[str, str]:
    if not retrieved_chunks:
        return (
            "I could not find a sufficiently relevant passage in the retrieved context.",
            "none",
        )

    tokenizer, model, model_name = load_generation_model()
    prompt = build_prompt(question, retrieved_chunks)

    try:
        messages = [
            {
                "role": "system",
                "content": "Answer using only the provided context. If the context is insufficient, say so clearly.",
            },
            {"role": "user", "content": prompt},
        ]
        rendered_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            rendered_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.no_grad():
            output_ids = _run_quietly(
                lambda: model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )
            )
        generated_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        if not answer:
            return extractive_fallback(question, retrieved_chunks), f"{model_name}-fallback"
        return answer, model_name
    except Exception:
        return extractive_fallback(question, retrieved_chunks), f"{model_name}-fallback"
