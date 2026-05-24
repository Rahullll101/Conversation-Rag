from fastapi import APIRouter, HTTPException
import logging
from backend.schemas.query import QueryRequest, QueryResponse, RetrievedChunk
from backend.services.retrieval_service import retrieve_chunks

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Retrieval"])

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Semantic Retrieval",
    description="Retrieve most relevant document chunks based on semantic similarity to the query."
)
async def query_document(request: QueryRequest):
    try:
        raw_chunks = retrieve_chunks(
            session_id=request.session_id,
            query=request.query
        )
        
        # Build strict Pydantic response
        parsed_chunks = [
            RetrievedChunk(
                chunk_id=c["chunk_id"],
                text=c["text"],
                score=c["score"],
                metadata=c["metadata"]
            ) for c in raw_chunks
        ]
        
        return QueryResponse(
            status="success",
            query=request.query,
            session_id=request.session_id,
            retrieved_chunks=parsed_chunks
        )

    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during retrieval.")
