import logging
import time
from typing import List, Dict, Any
from fastapi import HTTPException
from backend.services.vector_store_service import client, get_collection_name
from backend.services.embedding_service import generate_embeddings
from backend.config.settings import settings

logger = logging.getLogger(__name__)

def retrieve_chunks(session_id: str, query: str, top_k: int = None) -> List[Dict[str, Any]]:
    """Retrieve semantically relevant chunks from ChromaDB for a specific session."""
    if top_k is None:
        top_k = settings.retrieval_top_k
        
    collection_name = get_collection_name(session_id)
    
    start_time = time.time()
    
    try:
        # We explicitly fetch the collection here. If it doesn't exist, we fail gracefully.
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
    # ChromaDB query returns dictionaries of lists
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=top_k
    )
    
    from backend.utils.retrieval import format_chroma_results
    retrieved = format_chroma_results(results)

    latency = time.time() - start_time
    logger.info(f"Retrieved {len(retrieved)} chunks in {latency:.3f} seconds.")
    
    return retrieved
