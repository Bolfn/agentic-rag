"""CLI entrypoint for the LangGraph-based agentic RAG chat."""

from rag.graph import build_graph


def print_sources(retrieved_chunks: list[dict]) -> None:
    if not retrieved_chunks:
        return

    print("\nSources:")
    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk["metadata"]
        print(f"{index}. page={metadata['page_number']} chunk={metadata['page_chunk_index']}")


def main() -> None:
    graph = build_graph()
    chat_history: list[dict] = []

    print("Agentic RAG chat ready. Type 'exit' or 'quit' to stop.")

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        result = graph.invoke(
            {
                "question": question,
                "chat_history": chat_history,
            }
        )

        print(f"\nAnswer:\n{result.get('answer', 'No answer generated.')}")
        print_sources(result.get("retrieved_chunks", []))

        trace = result.get("agent_trace", [])
        if trace:
            print("\nAgent Trace:")
            for step in trace:
                print(f"- {step}")

        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": result.get("answer", "")})


if __name__ == "__main__":
    main()
