import json
import time
from pathlib import Path
from fastapi import APIRouter, HTTPException
from backend.config.settings import settings
from backend.services.chunking_service import chunk_text
from backend.services.embedding_service import generate_embeddings
from backend.services.vector_store_service import store_chunks

router = APIRouter(tags=["Document Processing"])

@router.post(
    "/process/{session_id}",
    summary="Process document into chunks and store embeddings",
    description="Loads extracted text, chunks it, generates embeddings, and stores them in ChromaDB."
)
async def process_document(session_id: str):
    session_dir = Path(settings.upload_dir) / f"session_{session_id}"
    extracted_text_path = session_dir / "extracted_text.txt"
    metadata_path = session_dir / "metadata.json"

    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session directory not found.")
    
    if not extracted_text_path.exists():
        raise HTTPException(status_code=400, detail="extracted_text.txt not found. Did document extraction complete successfully?")
        
    if not metadata_path.exists():
        raise HTTPException(status_code=400, detail="metadata.json not found.")

    try:
        text = extracted_text_path.read_text(encoding="utf-8")
        with metadata_path.open('r', encoding='utf-8') as f:
            meta = json.load(f)
            
        source_filename = meta.get("original_filename", "unknown")
        original_file_type = meta.get("content_type", "unknown")
        upload_timestamp = str(int(time.time())) # Simplification for now

        # 1. Chunking
        chunks = chunk_text(
            text=text,
            session_id=session_id,
            source_filename=source_filename,
            upload_timestamp=upload_timestamp,
            original_file_type=original_file_type
        )
        
        if not chunks:
            raise ValueError("Chunking resulted in 0 chunks.")

        # 2. Embedding
        chunk_texts = [c["text"] for c in chunks]
        embeddings = generate_embeddings(chunk_texts)
        
        # 3. Vector Storage
        store_chunks(session_id=session_id, chunks=chunks, embeddings=embeddings)
        
        return {
            "status": "success",
            "session_id": session_id,
            "total_chunks": len(chunks),
            "embedding_model": settings.embedding_model_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.post("/evaluate/{session_id}", summary="Run evaluation pipeline")
async def evaluate_pipeline(session_id: str):
    try:
        from backend.services.evaluation_service import run_evaluation_pipeline
        report = run_evaluation_pipeline(session_id)
        
        # Serialize the report to JSON
        entries = []
        for e in report.entries:
            entries.append({
                "question": e.question,
                "refused": e.hallucination_refused,
                "faithfulness": e.metrics.faithfulness,
                "relevancy": e.metrics.answer_relevancy
            })
            
        return {
            "status": "success",
            "aggregate_faithfulness": report.aggregate_metrics.faithfulness,
            "aggregate_relevancy": report.aggregate_metrics.answer_relevancy,
            "entries": entries
        }
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
