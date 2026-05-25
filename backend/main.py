__import__("pysqlite3")
import sys

# Override default sqlite3 with newer pysqlite3 version
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from fastapi import FastAPI
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