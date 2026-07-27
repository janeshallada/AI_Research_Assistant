# AI Research & Knowledge Assistant

A production-oriented backend that lets users upload PDF research/technical documents,
search across them semantically, ask grounded questions (RAG) with citations, compare
and summarize documents, and get documents auto-classified into technical domains by a
custom-trained TensorFlow model.

## 1. Project Overview

Organizations accumulate large volumes of PDFs (papers, specs, internal docs) that are
hard to search with plain keyword tools and risky to query with a generic LLM (which may
hallucinate). This project implements a Retrieval-Augmented Generation (RAG) system that:

- Ingests PDFs, extracts text with page-level metadata, and chunks it intelligently.
- Embeds chunks and indexes them in a vector database for semantic/keyword/hybrid search.
- Answers questions **only** from retrieved context, always citing document + page.
- Compares and summarizes documents on demand.
- Auto-classifies every uploaded document into a technical domain using a TensorFlow
  text classifier trained specifically for this project.
- Maintains conversational memory so follow-up questions ("What are *its* limitations?")
  resolve correctly.
- Exposes system analytics (documents, chunks, embeddings, most-queried documents).

## 2. Architecture Diagram

```
                    ┌────────────────┐
                    │   PDF Upload   │  (FastAPI /documents/upload)
                    └───────┬────────┘
                            │
                            ▼
                ┌───────────────────────┐        ┌─────────────────────────┐
                │ PDF Parser (PyMuPDF)  │───────▶│ TensorFlow Domain        │
                │ page-level metadata   │        │ Classifier (.h5 model)   │
                └───────┬───────────────┘        └────────────┬─────────────┘
                        │                                      │
                        ▼                                      ▼
                ┌───────────────────────┐          category + confidence
                │ Recursive Chunking     │          stored on Document row
                │ (size=1000, overlap=150)
                └───────┬───────────────┘
                        │
                        ▼
                ┌───────────────────────┐        ┌─────────────────────────┐
                │ Embedding Engine       │───────▶│ ChromaDB Vector Index    │
                │ (sentence-transformers │        │ (doc_id, file_name,      │
                │  or OpenAI embeddings) │        │  page_number metadata)   │
                └───────────────────────┘        └────────────┬─────────────┘
                                                                 │
                                query ──────────────────────────┤
                                                                 ▼
                                                  ┌───────────────────────────┐
                                                  │ Semantic / Keyword /       │
                                                  │ Hybrid Retrieval (top-K)   │
                                                  └────────────┬───────────────┘
                                                                 │
                          Conversation History ──────────────────┤
                                                                 ▼
                                                  ┌───────────────────────────┐
                                                  │ RAG Prompt + Citations     │
                                                  │  (LLM: OpenAI or Mock)     │
                                                  └────────────┬───────────────┘
                                                                 ▼
                                                        Final grounded answer
```

SQLite (via SQLAlchemy) stores document metadata, chunk records, chat sessions/messages,
and query logs used for analytics — independent of, but kept consistent with, the vector
store.

## 3. Technology Stack

| Layer | Technology |
|---|---|
| Backend Framework | FastAPI + Uvicorn |
| Document Processing | PyMuPDF (fitz) |
| Text Chunking / RAG | Custom recursive chunker + LangChain-style prompt templates |
| Vector Database | ChromaDB (local, file-persisted) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (default, local/offline) or OpenAI embeddings |
| LLM Engine | OpenAI (`gpt-4o`) or a deterministic offline **Mock** provider for local/dev/test use |
| Machine Learning | TensorFlow / Keras (`TextVectorization` + Dense classifier), scikit-learn (train/val split) |
| Metadata Database | SQLite via SQLAlchemy ORM (PostgreSQL-ready — swap the connection URL) |
| Testing | pytest |

## 4. Project Structure

```
ai-research-assistant/
├── config/settings.py            # Environment-based settings (pydantic-settings)
├── data/
│   ├── raw_documents/            # Uploaded PDFs
│   ├── vector_db/                # ChromaDB persistence
│   └── dataset/sample_training_data.csv  # Labelled data for the TF classifier
├── models/                       # tf_classifier.h5, tokenizer.pickle, labels.json (generated)
├── src/
│   ├── database/                 # SQLAlchemy engine/session + ORM models
│   ├── document_processing/      # pdf_parser.py, chunker.py, pipeline.py
│   ├── ml/                       # dataset_prep.py, train_classifier.py, predictor.py
│   ├── vector_store/             # embeddings.py, manager.py (semantic/keyword/hybrid)
│   ├── rag/                      # llm.py, qa_chain.py, summarizer.py, comparator.py
│   ├── analytics/metrics.py
│   └── schemas.py                # Pydantic request/response models
├── routes/                       # document_routes, search_routes, analysis_routes, analytics_routes
├── tests/                        # test_parser.py, test_rag.py, test_ml.py
├── main.py                       # FastAPI entry point
├── requirements.txt
├── .env.example
└── postman_collection.json
```

## 5. Setup Instructions

```bash
# 1. Clone and enter the project
git clone <repo-url> && cd ai-research-assistant

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — set OPENAI_API_KEY and LLM_PROVIDER=openai to use a real LLM,
# or leave LLM_PROVIDER=mock to run fully offline (see Section 9).

# 5. Train the TensorFlow classifier (writes models/tf_classifier.h5)
python -m src.ml.train_classifier

# 6. Run the API
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 7. Open Swagger UI
# http://localhost:8000/docs
```

Run tests:
```bash
pytest -v
```

## 6. Environment Variables

See `.env.example` for the full list. Key variables:

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `openai` or `mock` |
| `OPENAI_API_KEY` | Required only if `LLM_PROVIDER=openai` |
| `EMBEDDING_PROVIDER` | `local` (sentence-transformers) or `openai` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunking parameters (default 1000 / 150) |
| `RETRIEVAL_TOP_K` | Number of chunks retrieved per query (default 4) |
| `VECTOR_DB_DIR` / `SQLITE_DB_PATH` | Storage locations |

## 7. API Documentation (Summary)

Full interactive documentation is auto-generated by FastAPI at `/docs` (Swagger) and
`/redoc`. Summary of endpoints:

| Method | Endpoint | Description |
|---|---|---|
| POST | `/documents/upload` | Upload a PDF; triggers background ingestion pipeline |
| GET | `/documents` | List all documents + metadata + status |
| GET | `/documents/{doc_id}` | Get a single document's metadata |
| DELETE | `/documents/{doc_id}` | Delete a document (file, DB row, vector entries) |
| POST | `/documents/{doc_id}/reprocess` | Re-run the ingestion pipeline |
| POST | `/search` | Semantic / keyword / hybrid search (`mode` param) |
| POST | `/qa` | RAG question answering with citations + conversation memory (`session_id`) |
| POST | `/analysis/summarize` | Executive / Technical / Bullet / Key-Takeaway summary |
| POST | `/analysis/compare` | Compare 2+ documents |
| POST | `/analysis/classify` | Return the TensorFlow-predicted category for a document |
| GET | `/analytics/summary` | Document/chunk/embedding/question counts, category distribution |
| GET | `/analytics/top-documents` | Most frequently retrieved documents |

A ready-to-import `postman_collection.json` is included at the project root.

## 8. Chunking Strategy & Justification

Recursive, overlap-aware chunking (`src/document_processing/chunker.py`):

- **Chunk size**: 1000 characters (configurable, ~800–1000 recommended range).
- **Chunk overlap**: 150 characters.
- **Splitting priority**: paragraph breaks → line breaks → sentence breaks → hard
  character window, so chunks break on natural language boundaries whenever possible
  instead of mid-sentence.
- **Why overlap matters**: without it, a fact or clause that straddles the boundary
  between chunk *N* and chunk *N+1* can be truncated in both chunks and become
  effectively unretrievable/unusable in isolation. The 150-character overlap keeps
  enough trailing context from the previous chunk to preserve continuity.
- Page number is preserved per chunk (not per document), so every retrieved chunk can
  be cited back to an exact `(file_name, page_number)` pair.

## 9. Search Modes — When to Use Each

- **Semantic search**: best for conceptual, paraphrased, or "meaning" questions where
  the exact wording in the question won't match the source text (e.g. "what are the
  drawbacks of this approach?").
- **Keyword search**: best for precise lookups — exact identifiers, acronyms, numbers,
  or rare technical terms — where embedding similarity can under-rank an exact match.
- **Hybrid search** *(default)*: fuses normalized semantic + keyword scores. Recommended
  as the general-purpose default because it combines the recall of semantic search with
  the precision of exact keyword hits.

## 10. LLM & Embedding Modes (Design Decision)

To make the project runnable and testable **without requiring paid API access**, the LLM
layer (`src/rag/llm.py`) is pluggable:

- `LLM_PROVIDER=openai` → calls the OpenAI Chat Completions API (`gpt-4o` by default).
- `LLM_PROVIDER=mock` *(default)* → a deterministic, fully offline extractive responder
  that still enforces the "answer only from retrieved context" contract and returns the
  same fallback message when context is insufficient. This lets the RAG pipeline,
  citation logic, and conversation memory be exercised and unit-tested end-to-end with
  zero external dependencies or cost.

Similarly, `EMBEDDING_PROVIDER=local` uses `sentence-transformers/all-MiniLM-L6-v2`
(downloaded once, then fully offline) so the vector search pipeline does not require an
API key either. Both providers implement the same interface, so switching to a fully
managed OpenAI stack in production is a one-line `.env` change.

## 11. Assumptions

- A single labelled CSV (`data/dataset/sample_training_data.csv`) with `text,label`
  columns is used to train the TensorFlow classifier; it can be freely extended or
  replaced with a larger public dataset.
- "Total embeddings generated" is reported as the current vector-store chunk count
  (one embedding per indexed chunk).
- Authentication/multi-user support is out of scope for the core deliverable and listed
  under Future Improvements (it was offered as a bonus feature in the assignment).
- Conversation memory is session-scoped (via `session_id`) rather than user-scoped,
  since no auth layer is required by the core assignment.

## 12. Design Decisions

- **FastAPI** was chosen over Flask for native async support (non-blocking PDF
  processing via `BackgroundTasks`) and automatic OpenAPI/Swagger generation.
- **ChromaDB** was chosen over FAISS/Qdrant for zero-infrastructure, file-based
  persistence that keeps the project runnable with a single `pip install`.
- **SQLite** is the default metadata store; the SQLAlchemy layer means switching to
  PostgreSQL only requires changing `SQLITE_DB_PATH`/connection URL, no code changes.
- **Modular layout** (`src/document_processing`, `src/ml`, `src/vector_store`, `src/rag`,
  `src/analytics`) mirrors the pipeline stages so each concern is independently testable
  and swappable.

## 13. Limitations

- The bundled training dataset is small (course/demo scale); real deployments should use
  a larger, more diverse labelled corpus for the classifier.
- Scanned/image-only PDFs are not OCR'd (see Future Improvements).
- The keyword-search implementation is a simplified term-frequency scorer rather than a
  full BM25 implementation with IDF weighting, to avoid a separate search-engine
  dependency; it can be swapped for `rank-bm25` or an external engine if needed.
- The Mock LLM provider produces extractive (not generative) answers; it is intended for
  offline development/testing, not production-quality synthesis.

## 14. Future Improvements

- OCR for scanned PDFs (e.g. `pytesseract`).
- Authentication & multi-user support with per-user document isolation.
- Streaming LLM responses over SSE/WebSockets.
- Reranking model (cross-encoder) applied after hybrid retrieval.
- Dockerization and a CI/CD pipeline for automated testing/deployment.
- True BM25 (`rank-bm25`) for the keyword-search path.

## 15. Sample Documents

`data/sample_documents/` contains three ready-to-use demo PDFs (generated by
`generate_sample_pdfs.py`), one per domain, for exercising every feature end-to-end:

- `retrieval_augmented_generation.pdf` → classifies as **Artificial Intelligence**
- `cloud_native_microservices.pdf` → classifies as **Cloud Computing**
- `network_intrusion_detection.pdf` → classifies as **Cyber Security**

Upload all three via `POST /documents/upload` (or the Swagger UI at `/docs`) to demo
classification, then use `/analysis/compare` on any two of them, `/analysis/summarize`
on one, and `/qa` with a follow-up question to demo conversation memory. You can
regenerate/extend these with `python generate_sample_pdfs.py`.

## 16. Trained Model Artifacts

Running `python -m src.ml.train_classifier` produces:
- `models/tf_classifier.keras` — the trained Keras model, saved in the native Keras
  format rather than legacy `.h5`. **Design decision:** Keras 3's HDF5 (`.h5`) saver does
  not reliably persist the `TextVectorization` layer's lookup-table state, causing a
  "Table not initialized" error at inference after reload; the native `.keras` format
  serializes stateful preprocessing layers correctly, so it is used here. `MODEL_PATH` in
  `.env` can be repointed if your TensorFlow version differs.
- `models/tokenizer.pickle` — the TextVectorization vocabulary (inspection artifact)
- `models/labels.json` — the ordered category label list used at inference time

## 17. Running with Docker

```bash
cp .env.example .env
docker compose up --build
# API available at http://localhost:8000/docs
```

The container trains the classifier on first boot (using `data/dataset/sample_training_data.csv`)
and then starts the API. `./data` and `./models` are mounted as volumes so uploaded
documents, the vector index, and the trained model persist across container restarts.
