import logging
import time
from fastapi import APIRouter, HTTPException
from backend.schemas.pipeline import ChatPipelineRequest
from backend.schemas.answer import AnswerResponse, AnswerSource
from backend.services.memory_service import get_session_memory, add_memory_turn, clear_session_memory
from backend.services.query_rewrite_service import rewrite_query
from backend.services.retrieval_service import retrieve_chunks
from backend.services.rerank_service import rerank_chunks
from backend.services.prompt_service import build_prompt
from backend.services.llm_service import generate_answer_stream
from backend.schemas.query import RetrievedChunk
from backend.config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat Pipeline"])

import json
from fastapi.responses import StreamingResponse

@router.post(
    "/chat",
    summary="Conversational Query Generation (Streaming)",
    description="Full conversational pipeline: Memory -> Rewrite -> Retrieve -> Rerank -> Prompt -> LLM (Streaming)."
)
async def process_chat_pipeline(request: ChatPipelineRequest):
    session_id = request.session_id
    original_query = request.query
    
    start_time = time.time()
    
    try:
        # 1. Load Memory
        chat_history = get_session_memory(session_id)
        memory_used = len(chat_history) > 0
        
        # 2. Query Rewriting
        rewritten_query = rewrite_query(original_query, chat_history)
        
        # 3. Semantic Retrieval
        raw_retrieved = retrieve_chunks(session_id, rewritten_query)
        retrieved_chunks = [
            RetrievedChunk(
                chunk_id=c["chunk_id"], text=c["text"], score=c["score"], metadata=c["metadata"]
            ) for c in raw_retrieved
        ]
        
        # 4. Reranking
        reranked_chunks = rerank_chunks(rewritten_query, retrieved_chunks)
        
        # Trim chunks strictly based on source_chunk_limit for the prompt
        limited_chunks = reranked_chunks[:settings.source_chunk_limit]
        
        # 5. Build Prompt
        prompt = build_prompt(rewritten_query, limited_chunks, chat_history)
        
        # 6. Generate Answer (Streaming)
        def generate():
            full_answer = ""
            for chunk in generate_answer_stream(prompt):
                full_answer += chunk
                yield chunk
            
            # 7. Update Memory
            add_memory_turn(session_id, original_query, full_answer)
            
            # 8. Attach Sources
            sources = []
            for i, chunk in enumerate(limited_chunks):
                sources.append({
                    "chunk_id": chunk.chunk_id,
                    "source_file": chunk.metadata.get("source_filename"),
                    "chunk_index": chunk.metadata.get("chunk_index"),
                    "session_id": session_id,
                    "relevance_score": chunk.rerank_score
                })
            
            latency = time.time() - start_time
            logger.info(f"Chat pipeline finished in {latency:.3f}s.")
            
            # 9. Yield Metadata Separator and JSON
            metadata = {
                "status": "success",
                "session_id": session_id,
                "original_query": original_query,
                "rewritten_query": rewritten_query,
                "sources": sources,
                "memory_used": memory_used
            }
            yield f"\n\n__METADATA__\n{json.dumps(metadata)}"

        return StreamingResponse(generate(), media_type="text/plain")

    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Chat pipeline error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during answer generation.")

@router.delete("/chat/memory/{session_id}", summary="Clear chat memory")
async def delete_chat_memory(session_id: str):
    try:
        clear_session_memory(session_id)
        return {"status": "success", "message": f"Memory cleared for session {session_id}"}
    except Exception as e:
        logger.error(f"Failed to clear memory: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear memory")
