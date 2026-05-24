import logging
from typing import List, Dict, Any
from backend.config.settings import settings
from backend.schemas.pipeline import RerankedChunk
from backend.schemas.query import RetrievedChunk

logger = logging.getLogger(__name__)

# Lazy initialization for the CrossEncoder to save memory on startup
_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker model: {settings.reranker_model_name}")
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(settings.reranker_model_name)
    return _reranker

def rerank_chunks(query: str, retrieved_chunks: List[RetrievedChunk], top_k: int = None) -> List[RerankedChunk]:
    """
    Rerank retrieved chunks against the query using a CrossEncoder.
    """
    if not retrieved_chunks:
        return []
        
    if top_k is None:
        top_k = settings.rerank_top_k
        
    logger.info(f"Reranking {len(retrieved_chunks)} chunks for query: '{query}'")
    
    model = get_reranker()
    
    # CrossEncoder expects pairs of (query, document)
    pairs = [[query, chunk.text] for chunk in retrieved_chunks]
    
    # Compute scores
    scores = model.predict(pairs)
    
    # Attach scores and map to RerankedChunk schema
    reranked_list = []
    for i, chunk in enumerate(retrieved_chunks):
        reranked_list.append(
            RerankedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                retrieval_score=chunk.score or 0.0,
                rerank_score=float(scores[i]),
                metadata=chunk.metadata
            )
        )
        
    # Sort descending by rerank_score (CrossEncoder higher is more relevant)
    reranked_list.sort(key=lambda x: x.rerank_score, reverse=True)
    
    # Trim to top_k
    final_chunks = reranked_list[:top_k]
    
    logger.info(f"Reranking complete. Returning top {len(final_chunks)} chunks.")
    return final_chunks
