import json
import logging
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from backend.config.settings import settings
from backend.schemas.evaluation import EvaluationEntry, EvaluationReport, RagasMetrics
from backend.services.retrieval_service import retrieve_chunks
from backend.services.rerank_service import rerank_chunks
from backend.services.prompt_service import build_prompt
from backend.services.llm_service import generate_answer
from backend.schemas.query import RetrievedChunk
from backend.utils.evaluation import compute_ragas_metrics

logger = logging.getLogger(__name__)

EVALUATION_DIR = Path("evaluation")
RESULTS_DIR = EVALUATION_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_evaluation_dataset() -> List[Dict[str, Any]]:
    dataset_path = EVALUATION_DIR / "sample_queries.json"
    if not dataset_path.exists():
        logger.warning("Evaluation dataset not found.")
        return []
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_evaluation_pipeline(session_id: str) -> EvaluationReport:
    """
    Runs the chat pipeline for each query in the dataset and computes metrics.
    Strictly isolated from the main chat API route.
    """
    dataset = load_evaluation_dataset()
    limit = min(settings.evaluation_sample_size, len(dataset))
    dataset = dataset[:limit]
    
    entries = []
    
    for item in dataset:
        query = item.get("question", "")
        expected = item.get("expected_answer", "")
        qtype = item.get("type", "unknown")
        manual_obs = item.get("manual_observation", "")
        
        timestamp = datetime.utcnow().isoformat()
        
        # 1. Retrieval
        try:
            raw_retrieved = retrieve_chunks(session_id, query)
            retrieved_chunks = [
                RetrievedChunk(chunk_id=c["chunk_id"], text=c["text"], score=c["score"], metadata=c["metadata"]) 
                for c in raw_retrieved
            ]
        except ValueError as e:
            logger.warning(f"Evaluation retrieval failed: {e}")
            retrieved_chunks = []
        
        top_k_before = [{"chunk_id": c.chunk_id, "score": c.score} for c in retrieved_chunks]
        
        # 2. Reranking
        reranked_chunks = rerank_chunks(query, retrieved_chunks)
        limited_chunks = reranked_chunks[:settings.source_chunk_limit]
        
        top_k_after = [{"chunk_id": c.chunk_id, "score": c.rerank_score} for c in limited_chunks]
        
        # Failure analysis
        retrieval_failure_reason = None
        if not retrieved_chunks:
            retrieval_failure_reason = "no_chunks_retrieved"
        elif not limited_chunks:
            retrieval_failure_reason = "reranker_removed_relevant_chunk"
            
        # 3. Prompt & Generate
        prompt = build_prompt(query, limited_chunks, []) # Emulating no history for standalone queries
        answer = generate_answer(prompt)
        
        hallucination_refused = (answer == "I could not find enough information in the document.")
        
        # 4. Evaluate
        metrics = compute_ragas_metrics(query, answer, expected, limited_chunks)
        
        entry = EvaluationEntry(
            timestamp=timestamp,
            question=query,
            query_type=qtype,
            expected_answer=expected,
            generated_answer=answer,
            hallucination_refused=hallucination_refused,
            retrieval_failure_reason=retrieval_failure_reason,
            top_k_before_rerank=top_k_before,
            top_k_after_rerank=top_k_after,
            metrics=metrics,
            manual_observation=manual_obs
        )
        entries.append(entry)
        
    # Aggregate Metrics
    agg_metrics = RagasMetrics()
    if entries:
        agg_metrics.faithfulness = sum(e.metrics.faithfulness for e in entries) / len(entries)
        agg_metrics.answer_relevancy = sum(e.metrics.answer_relevancy for e in entries) / len(entries)
        agg_metrics.context_precision = sum(e.metrics.context_precision for e in entries) / len(entries)
        agg_metrics.context_recall = sum(e.metrics.context_recall for e in entries) / len(entries)
        
    report = EvaluationReport(
        timestamp=datetime.utcnow().isoformat(),
        total_queries=len(entries),
        aggregate_metrics=agg_metrics,
        entries=entries
    )
    
    _persist_reports(report)
    return report

def _persist_reports(report: EvaluationReport):
    """Save JSON and CSV reports."""
    # Save JSON
    json_path = RESULTS_DIR / "ragas_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))
        
    # Save CSV
    csv_path = RESULTS_DIR / "evaluation_summary.csv"
    with open(csv_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Question", "Type", "Refused", "RetrievalFailure", "Faithfulness", "Relevancy"])
        for e in report.entries:
            writer.writerow([
                e.timestamp,
                e.question,
                e.query_type,
                e.hallucination_refused,
                e.retrieval_failure_reason or "None",
                f"{e.metrics.faithfulness:.2f}",
                f"{e.metrics.answer_relevancy:.2f}"
            ])
    
    logger.info(f"Saved evaluation reports to {RESULTS_DIR}")
