# Assignment Report

## 1. Business Context

This project implements a local question-answering assistant over uploaded PDF documents. A user can place one or more PDFs into the import folder, build the local index, and ask natural-language questions against the indexed document set without relying on paid APIs.

The current demonstration corpus uses a sample policy manual PDF, because it is a realistic long-form document with sections, lists, repeated terms, and procedural content. The notebook and the fixed evaluation query set therefore use that sample corpus as an example, but the codebase itself is written as a general PDF RAG pipeline.

The main presentation artifact for the assignment is the notebook `agentic_rag_demo.ipynb`, which walks through the pipeline units, retrieval logic, and agentic control flow step by step.

## 2. System Architecture

![system diagram](system-diagram.svg)

The current implementation is an agentic RAG pipeline in `src/`:

1. load all PDFs from `imports/`
2. extract raw page text with document metadata
3. clean repeated headers, footers, and noisy pages
4. build heading/list-aware chunks
5. index chunk embeddings in Chroma
6. retrieve with hybrid search
7. rerank candidates
8. generate a grounded answer with a local instruct model
9. verify the answer and fall back to extractive output when needed

## 3. Key Technical Decisions

### 3.1 LLM Selection

The current local generation model is `Qwen/Qwen2.5-3B-Instruct`. It is stronger than very small CPU-friendly baselines, but still realistic to run locally on a modest consumer GPU. Larger local models were tested, but stability became an issue on the available hardware.

### 3.2 Embedding Model Selection

The project uses `intfloat/e5-base-v2` for dense retrieval. Compared to lighter sentence-transformer baselines, it gave better semantic matching on long-form document QA tasks without relying on external APIs.

### 3.3 Retrieval Strategy

The retrieval layer is hybrid:

- dense retrieval with Chroma + `e5-base-v2`
- lexical retrieval with BM25
- result fusion with reciprocal rank fusion
- reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`

This combination worked better than pure embedding search, especially when the query required both semantic matching and exact term overlap.

### 3.4 Chunking Strategy

The first block-based PDF extraction produced inconsistent layout fragments. The final solution moved to page text extraction and heading/list-aware chunking. The system now:

- filters noisy pages such as title pages, table-of-contents pages, and table-like pages
- preserves heading-like lines and lists
- builds section-like chunks before token-based splitting

This improved retrieval quality significantly over naive block extraction.

### 3.5 Framework Choice

The orchestration layer uses `LangGraph`. This allowed the project to move beyond a simple `retrieve -> answer` flow and implement agent-style decisions such as:

- smalltalk short-circuiting
- follow-up query rewriting
- broader retry search
- answer verification
- extractive fallback when generation is weak

## 4. Agentic Behavior

The agentic workflow contains the following stages:

1. **Input classification**
   - distinguishes normal questions from short acknowledgements such as `cool` or `thanks`
2. **Follow-up handling**
   - rewrites short contextual follow-up questions using chat history
3. **Hybrid retrieval**
   - combines semantic and lexical candidates
4. **Reranking**
   - improves top-k relevance for ambiguous questions
5. **Broader retry search**
   - expands retrieval when the first pass looks weak
6. **Answer generation**
   - uses the local Qwen instruct model on retrieved context
7. **Answer verification**
   - checks for weak or unsupported answers and falls back to extractive output

## 5. How I Would Measure Success In Production

I would track:

- top-1 and top-3 retrieval relevance
- groundedness of generated answers
- latency of retrieval and generation
- frequency of extractive fallback
- GPU and memory stability
- user satisfaction across different document sets

## 6. Demo Corpus And Examples

The current demo corpus is the sample policy manual placed in `imports/`. This is only the present example corpus used for demonstration and evaluation.

### Example 1

- **Question:** When do I get salary paid?
- **Generated answer:** Salary is paid bi-weekly. If pay day falls on a holiday or weekend, employees are paid on the last working day before the holiday or weekend. Employees can also use direct deposit.
- **Result:** Correct and grounded on the current demo corpus.

### Example 2

- **Question:** What is the holiday policy?
- **Generated answer:** The retrieved context describes the recognized holidays and election-day time-off conditions.
- **Result:** Correct for the sample policy-manual corpus, with extractive fallback available if generation becomes too vague.

### Example 3

- **Question:** cool
- **Generated answer:** Understood. Ask the next question when you're ready.
- **Result:** Correctly routed as smalltalk without retrieval.

## 7. Test / Evaluation Suite

The project includes:

- a fixed evaluation query set in `data/test_queries.json`
- an evaluation runner in `src/evaluate.py`

Important qualification:

- this evaluation set is **not universal**
- it is written for the **current demo corpus**
- when a different set of PDFs is indexed, the evaluation query file should also be updated

The current evaluation runner is primarily a retrieval-focused sanity check. It shows the top retrieved chunks and their scores for each test query. This is useful for debugging retrieval quality on a known corpus, but it is not yet a full automatic benchmark with expected-answer or expected-page scoring.

On the current sample corpus, the observed outcome was:

- retrieval became clearly stronger after hybrid search and reranking
- direct factual questions work well
- the retrieval layer is often better than the final generated answer
- concrete questions such as grievance procedure, harassment policy, direct deposit, travel reimbursement, and computer security produce strong grounded answers
- abstract summary questions are less stable, even when retrieval is good
- some generated answers still mix adjacent sections or over-summarize partially related chunks
- template-style source text can leak into answers as vague placeholders such as unresolved `days`, `weeks`, or `hours`
- generation quality with Qwen 3B is usable, but guardrails are still important

The most important practical takeaway from the evaluation is that the current system is already good at finding the right part of the document, but the smaller local generator is still the weaker link in the end-to-end chain.

## 8. Limitations

The current system still has limitations:

- local generation quality depends on GPU availability
- if CUDA is unavailable, response time becomes poor
- some answers still require extractive fallback
- retrieval is good but not perfect for repeated concepts
- abstract or high-level questions can still receive weaker answers than direct factual questions
- adjacent chunks can still be blended into one answer when the retrieved context is semantically close but not fully aligned
- placeholder-like source phrasing can be repeated too confidently by the generation model
- the current fixed evaluation suite is corpus-specific
- mixed-topic, larger multi-document corpora still need broader testing

The most likely latency bottleneck is hardware, especially GPU availability and local inference throughput. The most likely answer-quality bottleneck is the relatively small local generation model rather than the retrieval stack itself.

## 9. What Did Not Work Well

Several approaches were less successful:

- raw PDF block extraction produced inconsistent chunks
- pure embedding retrieval was weaker than hybrid retrieval
- larger local generation models caused stability problems on the available hardware
- weaker local models sometimes produced vague answers and needed stricter verification
- domain-specific hardcoded routing rules made the system less general and were removed

## 10. Next Steps

The next practical improvements would be:

- stronger structured evaluation with expected-document and expected-page annotations
- stricter answer verification for weak, overconfident, or poorly grounded generations
- more structured answer modes for list, threshold, and definition-style questions
- cleaner and smaller generation context to reduce cross-section mixing
- shorter, more factual prompting for local generation
- cleaner answer formatting for extractive fallback
- local model serving in a dedicated process for more stable startup behavior
- stronger local hardware or a more capable but still stable local model
- broader mixed-corpus testing with multiple unrelated PDFs
- stronger retrieval score analysis on multi-document corpora

## 11. Reflection

The strongest lesson from this project was that retrieval quality and system stability matter at least as much as the raw generation model. The final agentic version is much stronger than the original baseline because it combines better chunking, hybrid retrieval, reranking, and a graph-based control flow instead of relying on a single direct retrieval step.

At the same time, another important lesson was that a reusable RAG system should avoid corpus-specific hardcoded rules in the core logic. The current version is therefore framed as a general local PDF RAG pipeline, while the sample policy manual remains only the present demonstration corpus.
