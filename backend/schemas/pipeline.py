from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from backend.schemas.query import RetrievedChunk

class ChatPipelineRequest(BaseModel):
    session_id: str = Field(..., description="The unique session ID to query.")
    query: str = Field(..., description="The follow-up or standalone question.")

class RerankedChunk(BaseModel):
    chunk_id: str
    text: str
    retrieval_score: float
    rerank_score: float
    metadata: Dict[str, Any]

class ChatPipelineResponse(BaseModel):
    status: str = "success"
    session_id: str
    original_query: str
    rewritten_query: str
    retrieved_chunks: List[RetrievedChunk]
    reranked_chunks: List[RerankedChunk]
