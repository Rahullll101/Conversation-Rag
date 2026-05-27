import sys
from importlib.util import find_spec

# Prefer pysqlite3 when it is installed because ChromaDB needs a recent
# SQLite build on some hosts. Fall back to stdlib sqlite3 for local dev.
try:
    __import__("pysqlite3")
except ImportError:
    pass
else:
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from fastapi import FastAPI, Response
from backend.config.settings import settings

import logging
from contextlib import asynccontextmanager

from backend.routes import upload, process, query, chat

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application...")

    try:
        logger.info("Application startup complete")
        yield

    finally:
        logger.info("Shutting down application...")


app = FastAPI(
    title="Document Q&A RAG System API",
    description="API for the conversational Document Q&A RAG system",
    version="0.1.0",
    lifespan=lifespan
)

# Include API routes
app.include_router(upload.router)
app.include_router(process.router)
app.include_router(query.router)
app.include_router(chat.router)


@app.get("/")
async def root():
    return {
        "message": "Document Q&A RAG System API is running successfully"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy"
    }


@app.get("/ready")
async def readiness_check(response: Response):
    dependencies = {
        "pypdf": _module_available("pypdf"),
        "sentence_transformers": _module_available("sentence_transformers"),
        "chromadb": _module_available("chromadb"),
        "llama_index.core": (
            True
            if settings.rag_engine.lower() != "llamaindex"
            else _module_available("llama_index.core")
        ),
        "langchain_core": (
            True
            if settings.llm_orchestration.lower() != "langchain"
            else _module_available("langchain_core")
        ),
        "langchain_openrouter": (
            True
            if settings.llm_orchestration.lower() != "langchain"
            else _module_available("langchain_openrouter")
        ),
    }
    missing = [name for name, available in dependencies.items() if not available]
    llm_configured = bool(settings.openrouter_api_key.strip())
    production = settings.app_env.lower() == "production"
    ready = not missing and (llm_configured or not production)

    if not ready:
        response.status_code = 503

    return {
        "status": "ready" if ready else "degraded",
        "rag_engine": settings.rag_engine,
        "llm_orchestration": settings.llm_orchestration,
        "dependencies": dependencies,
        "llm_configured": llm_configured,
        "chroma_db_path": settings.chroma_db_path,
    }


def _module_available(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False
