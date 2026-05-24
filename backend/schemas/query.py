from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    session_id: str = Field(..., description="The unique session ID to query.")
    query: str = Field(..., description="The semantic question to ask the document.")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "query": "What is the refund policy?"
            }
        }
    }

class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: Optional[float] = None
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    status: str
    query: str
    session_id: str
    retrieved_chunks: List[RetrievedChunk]
