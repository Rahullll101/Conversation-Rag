# Conversational Document Q&A RAG System Documentation

Welcome to the documentation for the Conversational Document Q&A RAG (Retrieval-Augmented Generation) System. This solution implements a robust, production-aligned pipeline that combines semantic retrieval, chunking, Cross-Encoder reranking, and LLM-based streaming, with a strict focus on grounded, explainable answers.

---

# Traditional Rag Architecture
<img width="2000" height="1126" alt="image" src="https://github.com/user-attachments/assets/5e2f7f60-50ff-43e1-9a66-e0d7580c4efa" />

# 🏗️ Architecture Overview

This system answers user questions using only the information found in a set of provided PDF documents. The process consists of:

- PDF loading and text extraction
- Recursive character semantic chunking
- Embedding generation and isolated vector storage (ChromaDB)
- Conversational query rewriting for ambiguous pronouns
- Semantic retrieval (L2 Distance) and Cross-Encoder precision reranking
- Prompt construction with retrieved context
- LLM streaming strictly over the retrieved chunks

## System Architecture

```mermaid
flowchart TD
    A[User Query] 
        --> B[Query Rewrite]
        --> C[Query Embedding]
        --> D[Vector Database ChromaDB]
        --> E[Top-K Relevant Chunks]
        --> F[Cross-Encoder Reranker]
        --> G[Prompt Construction]
        --> H[LLM openai/gpt-oss-120b:free]
        --> I[Streaming Answer + Source Attribution]
```

> **Key Principle:**  
> The LLM *never* answers from its own internal knowledge. Answers are always grounded in the retrieved document context.

---

# 🔄 Workflow Diagram

This step-by-step workflow visualizes the dual upload and chat pipeline:

```mermaid
flowchart TD
    S[Upload PDF] 
        --> T[Extract Text & Metadata]
        --> U[Recursive Semantic Chunking]
        --> V[Generate Embeddings]
        --> W[Store in Isolated DB Collection]
    X[User Chat Message] 
        --> Y[Rewrite Query using Memory]
        --> Z[Retrieve Similar Chunks]
        --> AA[Rescore with Cross-Encoder]
        --> AB[LLM Streams Answer]
        --> AC[Return Answer + Source Percents]
```

---

# 📦 Project Structure & File Responsibilities

| File / Folder                        | Purpose                                                        |
|--------------------------------------|----------------------------------------------------------------|
| `.env`                               | Stores OpenRouter API token                                    |
| `requirements.txt`                   | Lists required Python dependencies                             |
| `backend/config/settings.py`         | Pydantic environment loaders and configuration                 |
| `backend/routes/`                    | FastAPI endpoint controllers (chat, process, upload)           |
| `backend/services/chunking_service.py`| Splits documents into semantically meaningful chunks           |
| `backend/services/embedding_service.py`| Embedding model loader and semantic generation              |
| `backend/services/retrieval_service.py`| Executes L2 semantic search against ChromaDB                |
| `backend/services/rerank_service.py` | Rescores chunks using Cross-Encoders                         |
| `backend/services/llm_service.py`    | Streaming Text Generation for the retrieved context            |
| `backend/utils/evaluation.py`        | Native LLM-as-a-judge strict grading prompts                   |
| `frontend/app.py`                    | Streamlit application entry point and chat UI                  |

---

# 📂 .env

This file stores your OpenRouter API token, required for accessing hosted LLMs.

```env
OPENROUTER_API_KEY=sk-or-v1-...
```

- **Purpose:** Securely store credentials.
- **Usage:** Loaded by `pydantic-settings` in `backend/config/settings.py`, required for LLM endpoint access.

---

# ⚙️ requirements.txt

Lists all dependencies needed to run the system.

```txt
fastapi
uvicorn
streamlit
pydantic
pydantic-settings
python-multipart
pypdf
langchain-text-splitters
chromadb
sentence-transformers
huggingface-hub
httpx
```

- **Purpose:** Declarative environment setup.
- **Notable Packages:**
  - `fastapi` & `streamlit`: Core API and UI frameworks.
  - `sentence-transformers`: For embeddings and Cross-Encoder reranking.
  - `chromadb`: Fast vector DB with metadata support.
  - `pypdf`: Lightweight offline PDF extraction.

---

# ✂️ backend/services/chunking_service.py

Provides semantic chunking logic using LangChain's text splitters.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_document(text: str, session_id: str) -> List[ChunkMetadata]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    # Splits while preserving paragraph boundaries
```

- **Purpose:** Splits documents into semantically meaningful text chunks.
- **Why recursive chunking?**
  - Preserves meaning across sentence and paragraph boundaries.
  - Reduces chances of splitting explanations or tables.

---

# 🧬 backend/services/embedding_service.py

Loads the embedding model for ChromaDB vector generation.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings(chunks: List[str]) -> List[List[float]]:
    """
    Generate dense vectors for document chunks.
    """
    return model.encode(chunks).tolist()
```

- **Purpose:** 
  - Loads the fast `all-MiniLM-L6-v2` transformer for semantic embeddings.
  - Converts text into 384-dimensional arrays for similarity search.

---

# 🎯 backend/services/rerank_service.py

Applies a Cross-Encoder to drastically improve retrieval precision.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_chunks(query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    # Calculates exact relevance between the query and each individual chunk
    scores = reranker.predict([[query, c.text] for c in chunks])
    # Returns sorted chunks
```

- **Purpose:** 
  - Overcomes the limitations of standard L2 Distance vector search.
  - Acts as a highly accurate filter before passing context to the LLM.

---

# 🧠 backend/services/llm_service.py

Interacts with OpenRouter for streaming generation.

```python
import httpx
import json

def generate_answer_stream(prompt: str):
    """
    Yields tokens as they stream in from the OpenRouter API.
    """
    with httpx.stream("POST", url, headers=headers, json=payload) as r:
        for chunk in r.iter_lines():
            # Parse JSON and yield raw text tokens
            yield token
```

- **Purpose:** 
  - Generates natural language answers based on retrieved context.
  - Streams tokens instantly for a highly responsive UI.
- **Notable:** 
  - Connects to the `openai/gpt-oss-120b:free` model.
  - Falls back to deterministic local mock generation if keys are missing.

---

# 🚀 frontend/app.py

Entry point for the Streamlit UI application.

```python
import streamlit as st
import httpx

st.title("Conversational RAG Assistant")

if prompt := st.chat_input("Ask a question about your document"):
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        # Streams the answer word-by-word via HTTPX
        for chunk in httpx.stream("POST", "http://127.0.0.1:8000/chat"):
            answer_placeholder.markdown(full_answer)
```

- **Purpose:** 
  - Provides a beautiful, interactive web UI.
  - Handles real-time HTTP streaming to render responses word-by-word.

---

# 🖥️ API Endpoints

The system is fully decoupled. The backend operates as a pure REST API.

## /chat (POST) – Ask a Question

### "Chat Pipeline" Endpoint (POST /chat)

```api
{
    "title": "Ask a Question (Streaming)",
    "description": "Submit a question to the RAG system. Returns a streaming response yielding tokens, followed by a __METADATA__ separator with sources.",
    "method": "POST",
    "baseUrl": "http://127.0.0.1:8000",
    "endpoint": "/chat",
    "headers": [
        {
            "key": "Content-Type",
            "value": "application/json",
            "required": true
        }
    ],
    "bodyType": "json",
    "requestBody": "{\n  \"session_id\": \"uuid-1234\",\n  \"message\": \"What is the loan approval process?\"\n}",
    "responses": {
        "200": {
            "description": "Streaming Chunked Response",
            "body": "The loan approval process involves... \n__METADATA__\n{\"sources\": [{\"chunk_id\": \"1\", \"relevance_score\": 5.4, \"source_file\": \"policy.pdf\"}]}"
        }
    }
}
```

---

# 🧩 Key Engineering Takeaways

```card
{
  "title": "Cross-Encoder Reranking",
  "content": "Standard vector search misses context. By fetching 6 chunks and applying a Cross-Encoder, we achieve near-perfect retrieval precision."
}
```

```card
{
  "title": "Native LLM-as-a-Judge",
  "content": "We bypassed expensive frameworks like Ragas by writing a strict zero-shot JSON prompt that forces the 120B LLM to grade its own Faithfulness mathematically."
}
```

```card
{
  "title": "Clean Architecture",
  "content": "Strictly separating stateless utils/ from stateful services/ ensures the codebase is highly testable, modular, and production-ready."
}
```

---

# 📝 Summary

This project demonstrates a production-aligned design for document-grounded question answering:

- Recursive chunking for high retrieval quality
- Isolated ChromaDB collections per session
- Cross-Encoder rescoring for maximum precision
- LLM reasoning strictly over provided context with live streaming
- Native, automated evaluation dashboards

All engineering decisions were made based on real-world trade-offs for reliability, transparency, and simplicity.

---

# 🎓 Further Reading

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Streamlit Chat Elements](https://docs.streamlit.io/library/api-reference/chat)

---

# 🚫 Failure Modes

**Expected Failure:**  
If a query is out-of-domain (e.g., "Explain quantum entanglement" when not present in any PDF), the system safely returns:

> "I could not find enough information in the document."

This confirms that the system is robust against hallucination and only answers from the provided knowledge base. The Native LLM Judge will automatically grade this refusal with a `1.0` for Faithfulness.
