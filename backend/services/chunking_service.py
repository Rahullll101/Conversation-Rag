import logging
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.utils.chunking import create_chunk_metadata
from backend.config.settings import settings

logger = logging.getLogger(__name__)

def chunk_text(
    text: str,
    session_id: str,
    source_filename: str,
    upload_timestamp: str,
    original_file_type: str
) -> List[Dict[str, Any]]:
    """Splits text into chunks and attaches metadata."""
    logger.info(f"Chunking text for session {session_id}")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    raw_chunks = splitter.split_text(text)
    
    processed_chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        metadata = create_chunk_metadata(
            session_id=session_id,
            chunk_index=i,
            text=chunk_text,
            source_filename=source_filename,
            upload_timestamp=upload_timestamp,
            original_file_type=original_file_type
        )
        
        processed_chunks.append({
            "text": chunk_text,
            "metadata": metadata
        })
        
    logger.info(f"Generated {len(processed_chunks)} chunks for session {session_id}")
    return processed_chunks
