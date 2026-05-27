FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ENV=production \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    RAG_ENGINE=llamaindex \
    LLM_ORCHESTRATION=langchain \
    CHROMA_DB_PATH=/app/data/chroma_db \
    UPLOAD_DIR=/app/data/uploads

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY backend ./backend
COPY evaluation ./evaluation

RUN mkdir -p /app/data/uploads /app/data/chroma_db /app/evaluation/results \
    && adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["sh", "-c", "uvicorn backend.main:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8000}"]
