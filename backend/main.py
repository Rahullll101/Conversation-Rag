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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Placeholder for startup logic
    yield
    # Placeholder for shutdown logic

app = FastAPI(
    title="Document Q&A RAG System API",
    description="API for the conversational Document Q&A RAG system",
    version="0.1.0",
    lifespan=lifespan
)

app.include_router(upload.router)
app.include_router(process.router)
app.include_router(query.router)
app.include_router(chat.router)



@app.get("/health")
async def health_check():
    return {"status": "healthy"}
