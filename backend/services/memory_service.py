import logging
from typing import List, Dict, Any
from backend.config.settings import settings

logger = logging.getLogger(__name__)

# In-memory dictionary for session-scoped memory
# Format: { "session_id": [ {"role": "user", "content": "..."}, {"role": "assistant_context", "content": "..."} ] }
_MEMORY_STORE: Dict[str, List[Dict[str, str]]] = {}

def get_session_memory(session_id: str) -> List[Dict[str, str]]:
    """Retrieve the chat history for a given session."""
    return _MEMORY_STORE.get(session_id, [])

def add_memory_turn(session_id: str, user_query: str, assistant_context_summary: str) -> None:
    """
    Store a conversational turn in memory and trim to memory_max_turns.
    """
    if session_id not in _MEMORY_STORE:
        _MEMORY_STORE[session_id] = []
        
    _MEMORY_STORE[session_id].append({"role": "user", "content": user_query})
    _MEMORY_STORE[session_id].append({"role": "assistant_context", "content": assistant_context_summary})
    
    # Trim memory history. Each 'turn' has 2 messages (user + assistant_context).
    max_messages = settings.memory_max_turns * 2
    if len(_MEMORY_STORE[session_id]) > max_messages:
        logger.info(f"Trimming memory for session {session_id} to last {settings.memory_max_turns} turns.")
        _MEMORY_STORE[session_id] = _MEMORY_STORE[session_id][-max_messages:]

def clear_session_memory(session_id: str) -> None:
    """Clear memory for a specific session."""
    if session_id in _MEMORY_STORE:
        del _MEMORY_STORE[session_id]
