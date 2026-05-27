from typing import List, Dict, Optional, Any
from pydantic import BaseModel

class EvaluationMetrics(BaseModel):
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0

class EvaluationEntry(BaseModel):
    timestamp: str
    question: str
    query_type: str
    expected_answer: str
    generated_answer: str
    hallucination_refused: bool
    retrieval_failure_reason: Optional[str] = None
    top_k_before_rerank: List[Dict[str, Any]]
    top_k_after_rerank: List[Dict[str, Any]]
    metrics: EvaluationMetrics
    manual_observation: Optional[str] = None

class EvaluationReport(BaseModel):
    timestamp: str
    total_queries: int
    aggregate_metrics: EvaluationMetrics
    entries: List[EvaluationEntry]
