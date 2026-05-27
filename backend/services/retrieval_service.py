import logging
import time
from typing import List, Dict, Any
from backend.config.settings import settings
from backend.services.llama_index_service import (
    LlamaIndexUnavailableError,
    retrieve_session_chunks,
)

logger = logging.getLogger(__name__)

def retrieve_chunks(session_id: str, query: str, top_k: int = None) -> List[Dict[str, Any]]:
    """Retrieve semantically relevant chunks from ChromaDB for a specific session."""
    if top_k is None:
        top_k = settings.retrieval_top_k

    if settings.rag_engine.lower() == "llamaindex":
        try:
            start_time = time.time()
            retrieved = retrieve_session_chunks(session_id=session_id, query=query, top_k=top_k)
            latency = time.time() - start_time
            logger.info(
                "LlamaIndex retrieved %s chunks in %.3f seconds.",
                len(retrieved),
                latency,
            )
            return retrieved
        except LlamaIndexUnavailableError as exc:
            if not settings.legacy_rag_fallback_enabled:
                raise
            logger.warning("Falling back to legacy retrieval: %s", exc)

    return _retrieve_chunks_legacy(session_id=session_id, query=query, top_k=top_k)


def _retrieve_chunks_legacy(session_id: str, query: str, top_k: int) -> List[Dict[str, Any]]:
    from backend.services.vector_store_service import client, get_collection_name
    from backend.services.embedding_service import generate_embeddings
    from backend.utils.retrieval import format_chroma_results

    collection_name = get_collection_name(session_id)
    start_time = time.time()

    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        logger.error(f"Failed to get collection {collection_name}: {e}")
        raise ValueError(f"Session {session_id} not found in database or not processed yet.")

    if collection.count() == 0:
        logger.warning(f"Collection {collection_name} is empty.")
        return []

    logger.info(f"Generating query embedding for session {session_id}")
    query_embeddings = generate_embeddings([query])
    
    logger.info(f"Querying ChromaDB for top {top_k} results")
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=top_k,
    )
    
    retrieved = format_chroma_results(results)

    latency = time.time() - start_time
    logger.info(f"Retrieved {len(retrieved)} chunks in {latency:.3f} seconds.")
    
    return retrieved
