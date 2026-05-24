from typing import List, Optional
from pydantic import BaseModel

class AnswerSource(BaseModel):
    chunk_id: str
    source_file: Optional[str] = None
    chunk_index: Optional[int] = None
    session_id: str
    relevance_score: float

class AnswerResponse(BaseModel):
    status: str = "success"
    session_id: str
    original_query: str
    rewritten_query: str
    answer: str
    sources: List[AnswerSource]
    memory_used: bool
