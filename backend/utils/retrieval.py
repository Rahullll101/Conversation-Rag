from typing import List, Dict, Any

def format_chroma_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Format raw ChromaDB results into a clean list of dictionaries.
    Converts Chroma's default L2 distance into a pseudo similarity score where higher is better.
    similarity = 1 / (1 + distance)
    """
    retrieved = []
    
    if not results.get('ids') or not results['ids'][0]:
        return retrieved

    ids = results['ids'][0]
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0] if results.get('distances') else [None] * len(ids)
    
    for i in range(len(ids)):
        distance = distances[i]
        
        # Convert L2 distance to a bounded similarity score [0, 1]
        similarity_score = 1.0 / (1.0 + distance) if distance is not None else 0.0
        
        chunk = {
            "chunk_id": ids[i],
            "text": documents[i],
            "score": similarity_score, # Higher is better
            "metadata": metadatas[i]
        }
        retrieved.append(chunk)
        
    return retrieved
