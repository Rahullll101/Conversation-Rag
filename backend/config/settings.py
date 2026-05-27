from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_env: str = "development"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    
    huggingface_api_key: str = ""
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_app_url: str = ""
    openrouter_app_title: str = "Conversation RAG"
    llm_model_name: str = "openai/gpt-oss-120b:free"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_db_path: str = "./data/chroma_db/"
    rag_engine: str = "llamaindex"
    llm_orchestration: str = "langchain"
    legacy_rag_fallback_enabled: bool = True
    openai_sdk_fallback_enabled: bool = True
    
    # Upload Settings
    max_upload_size_mb: int = 100
    allowed_mime_types: list[str] = ["application/pdf", "text/plain"]
    upload_dir: str = "./data/uploads"

    # Processing Settings
    chunk_size: int = 800
    chunk_overlap: int = 120
    collection_prefix: str = "session_"

    # Retrieval Settings
    retrieval_top_k: int = 6
    retrieval_score_threshold: float = 0.0

    # Reranking & Memory Settings
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 3
    memory_max_turns: int = 5
    query_rewrite_enabled: bool = True

    # Generation Settings
    max_generation_tokens: int = 512
    temperature: float = 0.1
    prompt_debug_mode: bool = False
    source_chunk_limit: int = 3

    # Evaluation Settings
    evaluation_sample_size: int = 100
    evaluation_enabled: bool = False
    evaluation_debug_mode: bool = True
    hallucination_test_enabled: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
