# Policy Manual Agentic RAG

This project builds a fully local RAG assistant over a policy manual PDF. The current implementation lives in `src/` and uses a LangGraph-based agent flow with local retrieval, reranking, and answer generation.

## Project Origin

This repository builds on an earlier version of my policy-manual RAG project:

- `https://github.com/Bolfn/policy-manual-rag.git`

The earlier repository served as the starting point, while this repository contains the current agentic RAG implementation with LangGraph orchestration, hybrid retrieval, reranking, follow-up handling, and local Qwen generation.

## Current Stack

- PDF extraction: `PyMuPDF`
- Cleaning and normalization: Python `re` + custom heuristics
- Chunking: heading/list-aware recursive chunking
- Embeddings: `intfloat/e5-base-v2`
- Vector database: `Chroma`
- Lexical retrieval: local `BM25`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Generation model: `Qwen/Qwen2.5-3B-Instruct`
- Agent orchestration: `LangGraph`

## Project Structure

- `src/pipeline.py`: end-to-end preprocessing and indexing pipeline
- `src/retrieve.py`: retrieval CLI
- `src/chat.py`: terminal chat interface
- `src/evaluate.py`: fixed query evaluation runner
- `src/rag/loaders.py`: PDF loading
- `src/rag/clean.py`: page cleaning and noise filtering
- `src/rag/chunking.py`: heading/list-aware chunking
- `src/rag/embed.py`: embedding generation
- `src/rag/index.py`: Chroma indexing
- `src/rag/retrieve.py`: hybrid retrieval and reranking
- `src/rag/llm.py`: local generation and extractive fallback
- `src/rag/graph.py`: LangGraph workflow
- `data/`: generated intermediate files and Chroma index

## Setup

```text
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Demo Notebook

For the assignment presentation, the main walkthrough notebook is:

- `agentic_rag_demo.ipynb`

The notebook explains the pipeline units, retrieval strategy, LangGraph flow, follow-up handling, smalltalk routing, bottlenecks, and next steps.

## Input Document

The system is currently built around:

- `sample_policy_and_procedures_manual.pdf`

## Build The Agentic Index

Run the preprocessing and indexing pipeline:

```text
python src/pipeline.py
```

This produces:

- `data/extracted_pages.json`
- `data/cleaned_pages.json`
- `data/chunked_documents.json`
- `data/embedded_chunks.json`
- `data/chroma_db/`

## Run Retrieval

To inspect retrieved chunks only:

```text
python src/retrieve.py "What is the holiday policy?"
```

## Run Chat

To run the terminal-based agentic RAG chat:

```text
cd src
python chat.py
```

Type `exit` or `quit` to stop.

## Run Evaluation

To run the fixed evaluation query set:

```text
python src/evaluate.py
```

The evaluation queries are stored in:

- `data/test_queries.json`

## Agent Flow

The current LangGraph workflow does the following:

1. classify the user question
2. detect follow-up questions
3. rewrite follow-up queries when needed
4. run hybrid retrieval
5. rerank candidate chunks
6. generate a grounded answer with the local Qwen model
7. verify the answer and fall back to extractive output when generation is weak
8. short-circuit smalltalk/acknowledgement inputs without retrieval

## Notes

- The current implementation is the code under `src/`.
- Document chunk embeddings are generated once during indexing; only query embeddings are computed at retrieval time.
- GPU access is required for acceptable local Qwen response times. If `nvidia-smi` or `torch.cuda.is_available()` fails, generation may fall back to CPU and become very slow.
