import argparse

from rag.retrieve import filter_results, format_results, load_retrieval_model, retrieve_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve top matching chunks from the agentic RAG index.")
    parser.add_argument("query", help="User query")
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable reranking and use hybrid retrieval only.",
    )
    args = parser.parse_args()

    model = load_retrieval_model()
    results = retrieve_chunks(args.query, model, use_reranker=not args.no_rerank)
    formatted_results = format_results(results)
    filtered_results = filter_results(formatted_results)

    if not filtered_results:
        print("No sufficiently relevant chunks were found.")
        return

    for index, item in enumerate(filtered_results, start=1):
        metadata = item["metadata"]
        print(f"[{index}]")
        if item.get("distance") is not None:
            print(f"distance={item['distance']:.4f}")
        if item.get("bm25_score") is not None:
            print(f"bm25_score={item['bm25_score']:.4f}")
        if item.get("rrf_score") is not None:
            print(f"rrf_score={item['rrf_score']:.4f}")
        if item.get("rerank_score") is not None:
            print(f"rerank_score={item['rerank_score']:.4f}")
        if item.get("sources"):
            print(f"sources={','.join(item['sources'])}")
        print(f"document={metadata['document_name']}")
        print(f"page={metadata['page_number']} chunk={metadata['page_chunk_index']}")
        print(item["text"])
        print()


if __name__ == "__main__":
    main()
