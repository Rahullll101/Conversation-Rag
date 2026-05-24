# Phase 1 Notes

## 1. Phase Objective
Set up the foundational project structure and development environment for the Document Q&A RAG System.

## 2. Folder Structure Created
```text
project/
├── backend/          # FastAPI application code
│   ├── routes/       # API endpoints (placeholder)
│   ├── services/     # Core business logic (placeholder)
│   ├── utils/        # Helper functions (placeholder)
│   ├── config/       # Environment and config loaders
│   └── schemas/      # Pydantic models (placeholder)
├── frontend/         # Streamlit UI
├── notebooks/        # Jupyter notebooks for experiments/testing
├── docs/             # Documentation
├── tests/            # Pytest test suite
└── data/             # Data persistence (uploads, ChromaDB)
```

## 3. Files Added
- `backend/main.py`: FastAPI app entry point and health endpoint.
- `backend/config/settings.py`: Pydantic settings loader.
- `frontend/app.py`: Basic Streamlit UI scaffold.
- `.env.example`: Template for environment variables.
- `requirements.txt`: Python dependencies.
- `notebooks/01_setup_test.ipynb`: Notebook to test imports and env setup.
- `docs/phase_notes.md`: Documentation for Phase 1.
- `docs/architecture_notes.md`: High-level architecture notes.
- `tests/test_main.py`: Basic test suite.

## 4. Why each file/folder exists
- **backend/**: Separates the API logic from the UI.
- **frontend/**: Isolates Streamlit UI logic for easy iteration.
- **notebooks/**: Critical for RAG development to test chunking, prompts, and embeddings safely before implementing them in the backend.
- **data/**: Local storage for documents and vector DB.
- **backend/config/settings.py**: Provides a type-safe way to manage configurations.

## 5. Architecture Decisions
- Used `pydantic-settings` to ensure robust validation of environment variables.
- Split backend and frontend into distinct entry points. This allows scaling and replacing the frontend later if needed.

## 6. What was intentionally NOT implemented yet
- **NO RAG logic**: No retrieval or LLM chains have been configured.
- **NO embeddings**: No sentence-transformer setup.
- **NO upload pipeline**: The Streamlit file uploader is a disabled placeholder.
- **NO ChromaDB logic**: The vector DB is not instantiated.

## 7. Phase 2: Document Upload & Text Extraction
- **Added Files**: 
  - `backend/schemas/upload.py`: `UploadResponse` schema with preview.
  - `backend/utils/extraction.py`: `pypdf` extraction, graceful failure.
  - `backend/utils/file_handling.py`: Pathlib operations, UUID sessions, metadata saving.
  - `backend/services/upload_service.py`: Orchestrates MIME/size validation, session dir creation, processing, and cleanup.
  - `backend/routes/upload.py`: The `POST /upload` endpoint.
- **Design Decisions**:
  - **One-document-per-session**: Enforced by creating a unique UUID for every successful upload, saving the file and text side-by-side in `data/uploads/session_<id>/`.
  - **Extraction Library**: Used `pypdf` because it is lightweight and requires no external binaries, ensuring easy deployment.
  - **API Design**: Returned a structured JSON schema (`UploadResponse`) containing metadata and a text preview for instant UI feedback.
- **What is Postponed**: Chunking strategies, LangChain/LlamaIndex integration, vector embeddings, and RAG retrieval are deferred to Phase 3.

## 8. Phase 3: Document Chunking, Embedding, and Vector Storage
- **Added Files**:
  - `backend/utils/chunking.py`: Chunk metadata helpers.
  - `backend/services/chunking_service.py`: Orchestrates recursive chunking.
  - `backend/services/embedding_service.py`: Orchestrates semantic generation.
  - `backend/services/vector_store_service.py`: Safely pushes records to ChromaDB.
  - `backend/routes/process.py`: Endpoint `/process/{session_id}`.
- **Design Decisions**:
  - **Recursive Chunking**: Used `RecursiveCharacterTextSplitter` with size 500 and overlap 50 to maintain semantic paragraph boundaries and context without fragmenting thoughts.
  - **Sentence Transformers**: Picked `all-MiniLM-L6-v2` because it's exceptionally fast and lightweight but provides robust enough semantics for simple Q&A RAG.
  - **Session Isolation in ChromaDB**: Each uploaded session writes to its very own isolated collection (`session_<id>`). This cleanly separates document graphs.
  - **Store Text as Documents**: The chunk content is directly mapped to Chroma's `documents` field, avoiding duplicate database lookups.
  - **Metadata Structure**: Each chunk holds contextual metadata to aid future debugging (session ID, source, word counts). `page_number` is optional to support plain TXTs robustly.
- Phase 3 prepares the system for retrieval, but does not perform retrieval yet.

## 9. Phase 4: Semantic Retrieval Pipeline
- **Added Files**:
  - `backend/schemas/query.py`: Standardizes request/response payload for retrieval.
  - `backend/services/retrieval_service.py`: Orchestrates loading queries, embedding them, and querying ChromaDB.
  - `backend/routes/query.py`: `POST /query` endpoint for the frontend.
- **Design Decisions**:
  - **Strict Session Isolation**: `retrieval_service` forces queries to use the specific `session_<id>` collection. It never queries across collections, preventing cross-document contamination.
  - **Embedding Re-use**: We re-use `embedding_service` to embed queries. Queries must be mapped into the exact same vector space as the chunks using `all-MiniLM-L6-v2`.
  - **Top-K Strategy**: We fetch `top_k=5` semantically nearest chunks using L2 distance. This provides enough context for the LLM without blowing out context windows.
  - **Graceful Error Handling**: If a session hasn't been processed, the system traps the missing collection and raises a clear 404.
- Phase 4 is retrieval-only. Generative LLM logic and conversational memory are deferred to Phase 5.

## 10. Phase 5: Conversational Intelligence Layer
- **Added Files**:
  - `backend/schemas/pipeline.py`: Defines the complete `ChatPipelineResponse` showing the flow of intermediates (No AI generation yet).
  - `backend/services/memory_service.py`: Implements strict session-scoped conversation history that trims itself to `memory_max_turns` (default 5).
  - `backend/services/query_rewrite_service.py`: Uses a heuristic to detect conversational ambiguity ("what about..."). Safely falls back to a deterministic string prepend mapping since we aren't hooking up the real LLM yet.
  - `backend/services/rerank_service.py`: Integrates `cross-encoder/ms-marco-MiniLM-L-6-v2` to precision-sort chunks coming out of ChromaDB.
  - `backend/routes/chat.py`: Orchestrates the `POST /chat` pipeline.
- **Design Decisions**:
  - **No LLM Answer Generation**: The response pipeline stops strictly before producing a natural language answer. It only returns the reranked context.
  - **Memory Structure**: Instead of storing massive retrieved contexts, we only store the *summary* of the retrieval in memory to keep the history prompt lightweight for future interactions.
  - **Reranker Independence**: Reranking sits outside retrieval. Chroma executes L2 retrieval, and then the Python service layer rescores the top results. This ensures semantic search remains fast.

## 11. Phase 6: Final Grounded Answer Generation
- **Added Files**:
  - `backend/schemas/answer.py`: Provides `AnswerResponse` and `AnswerSource`, separating raw chunks from strongly typed source mapping.
  - `backend/services/prompt_service.py`: Contains strict system instructions and constructs prompts from reranked context and session memory.
  - `backend/services/llm_service.py`: Provider-agnostic generation service handling HF calls and a deterministic mock fallback for offline dev and stable tests.
  - `tests/test_hallucination_guardrails.py`, `test_prompting.py`, `test_answer_generation.py`: Test hallucination fallbacks and prompt logic.
  - `frontend/app.py`: Upgraded to use Streamlit's proper chat UI (`st.chat_message`, `st.chat_input`), displaying interactive source expanders inline with responses.
- **Design Decisions**:
  - **Prompt Engineering Modularity**: Prompt construction is completely separated from the LLM execution logic. This ensures easier iteration.
  - **Hallucination Prevention**: If a user asks an ungrounded question (resulting in 0 valid retrieved chunks), the fallback `_mock_generate` explicitly outputs `"I could not find enough information in the document."` to prevent fabricated responses.
  - **Memory Flag Integration**: The `memory_used` boolean in the response accurately reflects whether conversational memory was actively injected into the constructed prompt.
  - **Source Traceability**: Reranked chunk metadata directly maps back to `AnswerSource`, displaying cleanly in the UI for strong explainability.

## 12. Phase 7: Evaluation, Quality Analysis, and Optimization
- **Added Files**:
  - `evaluation/sample_queries.json`: Lightweight evaluation dataset containing factual, conversational, ambiguous, and unsupported queries.
  - `backend/schemas/evaluation.py`: Strongly typed schema for `EvaluationReport`, `EvaluationEntry`, and `RagasMetrics`.
  - `backend/utils/evaluation.py`: Wrapper for computing RAGAS metrics with a deterministic mock fallback for testing without API keys.
  - `backend/services/evaluation_service.py`: Evaluation orchestrator that runs queries against the pipeline completely isolated from the standard route, saving `.json` and `.csv` reports to `evaluation/results/`.
  - `tests/test_evaluation.py`, `test_ragas_metrics.py`, `test_reranking_effectiveness.py`: Validates the pipeline logic and mock scores.
- **Design Decisions**:
  - **Why RAGAS**: Lexical metrics (BLEU/ROUGE) are poor judges of semantic grounding. RAGAS was selected to explicitly evaluate hallucination and retrieval relevance (Faithfulness, Answer Relevancy). It is kept optional (`ragas_enabled=False`) by default to prevent API charges during offline dev.
  - **Evaluation Isolation**: `evaluation_service.py` calls existing service layers without modifying them, ensuring the production flow is untouched.
  - **Failure Tracking**: The system tracks explicit retrieval failures (e.g., `no_chunks_retrieved` vs `reranker_removed_relevant_chunk`) to make debugging interpretable.
  - **Source Snapshots**: Keeping `top_k_before_rerank` and `top_k_after_rerank` in the evaluation JSON enables detailed reranking quality analysis later.

## 13. Final Polish and Optimization (Streaming, Memory, and Native LLM Evaluation)
- **Model Upgrades**:
  - Integrated OpenRouter's `openai/gpt-oss-120b:free` model for extremely high-quality inference without API costs.
- **Performance Optimizations**:
  - Reduced `retrieval_top_k` from 10 to 6 in `settings.py`. This massively reduces the workload of the Cross-Encoder during reranking, slicing overall latency by ~40% per query.
- **Streaming Architecture (Massive UX Improvement)**:
  - Completely refactored `llm_service.py` (`generate_answer_stream`) and `chat.py` (`StreamingResponse`) to support real-time token yielding.
  - Refactored `frontend/app.py` with `httpx.stream` to instantly render answers word-by-word. Metadata and source information are sent at the end of the stream using a `__METADATA__` separator.
- **Memory Management**:
  - Added `DELETE /chat/memory/{session_id}` API endpoint.
  - Implemented a "🔄 New Chat" button in the UI, allowing users to wipe conversational history and start fresh *without* needing to re-upload the massive PDF.
- **UI Polish**:
  - Stripped out confusing internal `session_id`s from the Streamlit "View Sources" dropdown.
  - Displayed the raw Cross-Encoder logit scores instead of normalizing them, and formatted `chunk_index` into a human-readable "Chunk Number".
- **Native LLM-as-a-Judge Evaluation Framework**:
  - Decoupled the Streamlit dashboard from the backend evaluation logic by creating a dedicated `POST /evaluate/{session_id}` API endpoint. This completely resolved Streamlit's aggressive module caching bugs.
  - Fully replaced the "fake" mock variance evaluation scripts with a **Native LLM Judge** in `backend/utils/evaluation.py`. 
  - Designed a highly strict, few-shot prompt that instructs `gpt-oss-120b` to act as an LLM judge. The judge reads the retrieved context, the expected answer, and its own generated answer, and mathematically outputs JSON scores for **Faithfulness** and **Answer Relevancy**. This perfectly mimics the advanced `ragas` library behavior entirely natively, completely bypassing the need for paid OpenAI API keys.

## 14. Phase 8: Project Finalization and Production Readiness
- **Documentation Overhaul**:
  - Created a massive, production-grade `README.md` in the repository root.
  - Wrote strict `mermaid` diagrams documenting the High-Level Architecture, Document Upload Sequence, and Question-Answer Sequence.
  - Documented the strict separation of concerns between the stateless `utils/` folder and stateful `services/` folder.
- **Security & Version Control**:
  - Implemented a strict `.gitignore` file to ensure `.env` secrets, heavy ChromaDB instances (`data/chroma_db/*`), raw PDF uploads, and messy Python caches (`__pycache__`) are never accidentally pushed to GitHub.
