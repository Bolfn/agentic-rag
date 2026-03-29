import argparse
import json
from pathlib import Path

from rag.config import DATA_DIR
from rag.retrieve import (
    filter_results,
    format_results,
    load_reranker,
    load_retrieval_model,
    retrieve_chunks,
)


DEFAULT_QUERY_PATH = DATA_DIR / "test_queries.json"


def load_queries(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple retrieval evaluation over a fixed query set.")
    parser.add_argument(
        "--query-file",
        default=str(DEFAULT_QUERY_PATH),
        help="Path to a JSON file containing evaluation queries.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of results to display per query.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable reranking and evaluate hybrid retrieval only.",
    )
    args = parser.parse_args()

    query_path = Path(args.query_file)
    queries = load_queries(query_path)

    model = load_retrieval_model()
    reranker = None if args.no_rerank else load_reranker()

    print(f"Loaded {len(queries)} evaluation queries from {query_path}")
    print(f"Mode: {'hybrid-only' if args.no_rerank else 'hybrid+rerank'}")
    print()

    for item in queries:
        print(f"=== {item['id']} | {item['category']} ===")
        print(item["query"])

        results = retrieve_chunks(
            item["query"],
            model,
            top_k=args.top_k,
            use_reranker=not args.no_rerank,
            reranker=reranker,
        )
        formatted = format_results(results)
        filtered = filter_results(formatted)

        if not filtered:
            print("No sufficiently relevant chunks were found.")
            print()
            continue

        for index, result in enumerate(filtered[: args.top_k], start=1):
            metadata = result["metadata"]
            text_preview = result["text"].replace("\n", " ").strip()
            text_preview = " ".join(text_preview.split())
            if len(text_preview) > 240:
                text_preview = text_preview[:237] + "..."

            score_parts = []
            if result.get("distance") is not None:
                score_parts.append(f"distance={result['distance']:.4f}")
            if result.get("bm25_score") is not None:
                score_parts.append(f"bm25={result['bm25_score']:.4f}")
            if result.get("rrf_score") is not None:
                score_parts.append(f"rrf={result['rrf_score']:.4f}")
            if result.get("rerank_score") is not None:
                score_parts.append(f"rerank={result['rerank_score']:.4f}")

            print(
                f"{index}. page={metadata['page_number']} chunk={metadata['page_chunk_index']} "
                f"{' '.join(score_parts)}"
            )
            print(text_preview)

        print()


if __name__ == "__main__":
    main()
