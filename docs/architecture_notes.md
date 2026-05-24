# Architecture Notes

## Final Stack Goal
- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Retrieval**: LlamaIndex
- **Orchestration**: LangChain
- **Vector DB**: ChromaDB
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **LLM**: HuggingFace gpt-oss-120b

## Design Principles
- **Modularity**: Separation of concerns between UI, API, and internal services (retrieval, LLM processing).
- **Experiment-First**: Utilize Jupyter Notebooks for RAG experiments (chunking strategies, retrieval evaluation) before integrating into backend services.
- **Configuration-Driven**: Keep model names, keys, and paths in environment variables (`.env`) for easy staging and testing.
