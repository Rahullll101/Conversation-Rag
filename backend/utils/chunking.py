import hashlib
from typing import Optional, Dict, Any

def generate_chunk_id(session_id: str, chunk_index: int) -> str:
    """Generate a unique ID for a chunk based on session_id and its index."""
    unique_string = f"{session_id}_{chunk_index}"
    return hashlib.md5(unique_string.encode('utf-8')).hexdigest()

def create_chunk_metadata(
    session_id: str,
    chunk_index: int,
    text: str,
    source_filename: str,
    upload_timestamp: str,
    original_file_type: str,
    page_number: Optional[int] = None
) -> Dict[str, Any]:
    """Create rich metadata for a chunk."""
    metadata = {
        "session_id": session_id,
        "chunk_id": generate_chunk_id(session_id, chunk_index),
        "chunk_index": chunk_index,
        "source_filename": source_filename,
        "character_count": len(text),
        "word_count": len(text.split()),
        "upload_timestamp": upload_timestamp,
        "original_file_type": original_file_type,
        "chunk_preview": text[:100] + "..." if len(text) > 100 else text,
    }
    if page_number is not None:
        metadata["page_number"] = page_number
    return metadata
