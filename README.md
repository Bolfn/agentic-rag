# Agentic RAG

This project builds a fully local agentic RAG assistant for question answering over one or more PDF documents. The current implementation lives in `src/` and uses a LangGraph-based control flow with hybrid retrieval, reranking, local generation, and verification.

The system is designed as a general PDF RAG pipeline:

- put PDF files into `imports/`
- run the indexing pipeline
- ask questions over the indexed document set

For demonstration, the repository currently uses a sample policy manual PDF as the example corpus, and the included evaluation query set is written for that example corpus.

## Project Origin

This repository builds on an earlier version of my policy-manual RAG project:

- `https://github.com/Bolfn/policy-manual-rag.git`

The earlier repository served as the starting point. This repository contains the generalized agentic rewrite with LangGraph orchestration, multi-PDF ingestion, hybrid retrieval, reranking, follow-up handling, and local Qwen generation.

## Current Stack

- PDF extraction: `PyMuPDF`
- Cleaning and normalization: Python `re` + custom heuristics
- Chunking: heading/list-aware recursive chunking
- Embeddings: `intfloat/e5-base-v2`
- Vector database: `Chroma`
- Lexical retrieval: local BM25
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Generation model: `Qwen/Qwen2.5-3B-Instruct`
- Agent orchestration: `LangGraph`

## Project Structure

- `imports/`: PDF files to ingest
- `src/pipeline.py`: end-to-end preprocessing and indexing pipeline
- `src/retrieve.py`: retrieval CLI
- `src/chat.py`: terminal chat interface
- `src/evaluate.py`: fixed query evaluation runner
- `src/rag/loaders.py`: multi-PDF loading
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

The notebook explains the pipeline units, retrieval strategy, LangGraph flow, follow-up handling, smalltalk routing, bottlenecks, and next steps. It presents the current example corpus based on the sample policy manual in `imports/`.

## Input Documents

The pipeline indexes every PDF found under:

- `imports/`

You can replace the example PDF with your own documents, or add multiple PDFs. The pipeline will ingest all of them into a shared local index.

Optional override:

```text
RAG_IMPORT_DIR=/path/to/pdf-folder python src/pipeline.py
```

## Build The Index

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
python src/retrieve.py "What does the document say about direct deposit?"
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

The current evaluation queries are stored in:

- `data/test_queries.json`

Important:

- the evaluation file is corpus-specific
- the included query set is meant for the current example policy manual corpus
- if you change the indexed PDFs, you should also update `data/test_queries.json`
- the current runner is mainly a retrieval-focused sanity check, not a full automatic answer benchmark

## Agent Flow

The current LangGraph workflow does the following:

1. classify the input as question or smalltalk
2. detect follow-up questions
3. rewrite contextual follow-up queries when needed
4. run hybrid retrieval
5. rerank candidate chunks
6. broaden retrieval when the first pass looks weak
7. generate a grounded answer with the local Qwen model
8. verify the answer and fall back to extractive output when generation is weak
9. short-circuit acknowledgements without retrieval

## Notes

- The current implementation is the code under `src/`.
- Document chunk embeddings are generated once during indexing; only query embeddings are computed at retrieval time.
- GPU access is required for acceptable local Qwen response times. If `nvidia-smi` or `torch.cuda.is_available()` fails, generation may fall back to CPU and become very slow.
- The system is intended as a general local PDF RAG pipeline, while the sample policy manual is only the current demonstration corpus.
