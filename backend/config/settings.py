from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_env: str = "development"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    
    huggingface_api_key: str = ""
    openrouter_api_key: str = ""
    llm_model_name: str = "openai/gpt-oss-120b:free"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_db_path: str = "./data/chroma_db/"
    
    # Upload Settings
    max_upload_size_mb: int = 100
    allowed_mime_types: list[str] = ["application/pdf", "text/plain"]
    upload_dir: str = "./data/uploads"

    # Processing Settings
    chunk_size: int = 500
    chunk_overlap: int = 50
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
    ragas_enabled: bool = False
    evaluation_debug_mode: bool = True
    hallucination_test_enabled: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
