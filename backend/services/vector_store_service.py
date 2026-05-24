import logging
import chromadb
from typing import List, Dict, Any
from backend.config.settings import settings

logger = logging.getLogger(__name__)

# Initialize persistent chroma client
client = chromadb.PersistentClient(path=settings.chroma_db_path)

def get_collection_name(session_id: str) -> str:
    """Generate collection name safely."""
    # Chroma collection names must be valid: alphanumeric and underscores
    safe_session = session_id.replace('-', '_')
    return f"{settings.collection_prefix}{safe_session}"

def store_chunks(
    session_id: str,
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]]
) -> None:
    """Stores text, metadata, and embeddings in ChromaDB, overwriting any existing session collection."""
    collection_name = get_collection_name(session_id)
    
    # Idempotent: delete existing collection if it exists
    try:
        client.delete_collection(name=collection_name)
        logger.info(f"Deleted existing collection for session {session_id}")
    except ValueError:
        pass # Collection didn't exist
    except Exception as e:
        logger.warning(f"Could not delete collection (might not exist): {e}")

    try:
        collection = client.create_collection(name=collection_name)
        logger.info(f"Created new collection: {collection_name}")
        
        ids = [chunk["metadata"]["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Batch insert
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        logger.info(f"Successfully stored {len(chunks)} vectors in {collection_name}")
        
    except Exception as e:
        logger.error(f"Failed to store chunks in ChromaDB: {e}")
        # Rollback: try to clean up the collection if it failed halfway
        try:
            client.delete_collection(name=collection_name)
            logger.info(f"Rolled back collection creation for {collection_name}")
        except:
            pass
        raise e
