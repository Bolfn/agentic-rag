import json
import io
import math
import re
from contextlib import redirect_stderr, redirect_stdout
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import chromadb
from chromadb.config import Settings
from sentence_transformers import CrossEncoder, SentenceTransformer

from .config import (
    BM25_CANDIDATES,
    CHROMA_DIR,
    CHUNKED_PATH,
    COLLECTION_NAME,
    EMBEDDING_CANDIDATES,
    EMBEDDING_MODEL_NAME,
    MAX_CONTEXT_DISTANCE,
    RERANK_CANDIDATES,
    RERANKER_MODEL_NAME,
    TOP_K,
)


def _run_quietly(func):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return func()


@lru_cache(maxsize=1)
def get_collection():
    client = _run_quietly(
        lambda: chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    )
    return _run_quietly(lambda: client.get_collection(COLLECTION_NAME))


@lru_cache(maxsize=1)
def load_retrieval_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True, device="cpu")


@lru_cache(maxsize=1)
def load_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL_NAME, local_files_only=True, device="cpu")


@lru_cache(maxsize=1)
def load_chunk_documents() -> list[dict]:
    payload = json.loads(Path(CHUNKED_PATH).read_text(encoding="utf-8"))
    return cast(list[dict], payload)


def tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class BM25Index:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.tokenized_docs = [tokenize_for_bm25(doc["text"]) for doc in documents]
        self.doc_count = len(self.tokenized_docs)
        self.avg_doc_len = (
            sum(len(tokens) for tokens in self.tokenized_docs) / self.doc_count
            if self.doc_count
            else 0.0
        )
        self.doc_freqs: dict[str, int] = {}
        self.term_freqs: list[dict[str, int]] = []

        for tokens in self.tokenized_docs:
            frequencies: dict[str, int] = {}
            seen = set()
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1
                if token not in seen:
                    self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
                    seen.add(token)
            self.term_freqs.append(frequencies)

    def get_scores(self, query: str, k1: float = 1.5, b: float = 0.75) -> list[float]:
        query_tokens = tokenize_for_bm25(query)
        scores = [0.0] * self.doc_count

        for token in query_tokens:
            doc_freq = self.doc_freqs.get(token, 0)
            if doc_freq == 0:
                continue

            idf = math.log(1 + (self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5))

            for index, frequencies in enumerate(self.term_freqs):
                term_freq = frequencies.get(token, 0)
                if term_freq == 0:
                    continue

                doc_len = len(self.tokenized_docs[index])
                denominator = term_freq + k1 * (
                    1 - b + b * (doc_len / self.avg_doc_len if self.avg_doc_len else 0.0)
                )
                scores[index] += idf * ((term_freq * (k1 + 1)) / denominator)

        return scores


def retrieve_embedding_candidates(
    query: str,
    model: SentenceTransformer,
    candidate_count: int = EMBEDDING_CANDIDATES,
) -> list[dict]:
    collection = get_collection()
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()
    results = _run_quietly(
        lambda: collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_count,
        )
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    return [
        {
            "chunk_id": chunk_id,
            "text": document,
            "metadata": metadata,
            "distance": distance,
            "source": "embedding",
        }
        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        )
    ]


def retrieve_bm25_candidates(
    query: str,
    documents: list[dict],
    candidate_count: int = BM25_CANDIDATES,
) -> list[dict]:
    bm25 = BM25Index(documents)
    scores = bm25.get_scores(query)
    ranked = sorted(
        zip(documents, scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )

    candidates = []
    for document, score in ranked[:candidate_count]:
        candidates.append(
            {
                "chunk_id": document["chunk_id"],
                "text": document["text"],
                "metadata": {
                    "page_number": document["page_number"],
                    "page_chunk_index": document["page_chunk_index"],
                    "token_count": document["token_count"],
                },
                "bm25_score": score,
                "source": "bm25",
            }
        )

    return candidates


def reciprocal_rank_fusion(result_sets: list[list[dict]], k: int = 60) -> list[dict]:
    fused: dict[str, dict] = {}

    for result_set in result_sets:
        for rank, item in enumerate(result_set, start=1):
            chunk_id = item["chunk_id"]
            entry = fused.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "rrf_score": 0.0,
                    "sources": set(),
                    "distance": item.get("distance"),
                    "bm25_score": item.get("bm25_score"),
                },
            )
            entry["rrf_score"] += 1.0 / (k + rank)
            entry["sources"].add(item["source"])
            if item.get("distance") is not None:
                entry["distance"] = item["distance"]
            if item.get("bm25_score") is not None:
                entry["bm25_score"] = item["bm25_score"]

    ranked = sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)
    for item in ranked:
        item["sources"] = sorted(item["sources"])
    return ranked


def rerank_results(
    query: str,
    results: list[dict],
    reranker: CrossEncoder,
    top_k: int = TOP_K,
) -> list[dict]:
    candidates = results[:RERANK_CANDIDATES]
    if not candidates:
        return []

    pairs = [(query, item["text"]) for item in candidates]
    scores = reranker.predict(pairs)

    reranked = []
    for item, score in zip(candidates, scores, strict=True):
        reranked.append(
            {
                **item,
                "rerank_score": float(score),
            }
        )

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return reranked[:top_k]


def retrieve_chunks(
    query: str,
    model: SentenceTransformer,
    top_k: int = TOP_K,
    use_reranker: bool = True,
    reranker: CrossEncoder | None = None,
) -> list[dict]:
    chunk_documents = load_chunk_documents()
    embedding_candidates = retrieve_embedding_candidates(query, model)
    bm25_candidates = retrieve_bm25_candidates(query, chunk_documents)
    fused_results = reciprocal_rank_fusion([embedding_candidates, bm25_candidates])

    if use_reranker:
        active_reranker = reranker or load_reranker()
        return rerank_results(query, fused_results, active_reranker, top_k=top_k)

    return fused_results[:top_k]


def format_results(results: list[dict[str, Any]]) -> list[dict]:
    formatted = []

    for item in results:
        formatted.append(
            {
                "text": item["text"],
                "metadata": item["metadata"],
                "distance": item.get("distance"),
                "bm25_score": item.get("bm25_score"),
                "rrf_score": item.get("rrf_score"),
                "rerank_score": item.get("rerank_score"),
                "sources": item.get("sources", []),
            }
        )

    return formatted


def filter_results(results: list[dict], max_distance: float = MAX_CONTEXT_DISTANCE) -> list[dict]:
    filtered = []

    for result in results:
        distance = result.get("distance")
        if distance is None or distance <= max_distance:
            filtered.append(result)

    return filtered
