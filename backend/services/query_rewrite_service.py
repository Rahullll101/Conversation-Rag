import logging
import re
from typing import List, Dict
from backend.config.settings import settings

logger = logging.getLogger(__name__)

def is_ambiguous(query: str) -> bool:
    """
    Lightweight heuristic to detect if a query might need rewriting.
    We don't want to rewrite perfectly clear standalone queries.
    """
    query_lower = query.lower()
    
    # Check for short follow-ups
    if len(query.split()) <= 3:
        return True
        
    # Check for conversational pronouns or continuations
    ambiguous_keywords = [
        "what about", "how about", "and",
        "it", "this", "that", "those", "these",
        "he", "she", "they", "them", "their",
        "explain more", "tell me more"
    ]
    
    for kw in ambiguous_keywords:
        # Regex to match whole words only
        if re.search(r'\b' + re.escape(kw) + r'\b', query_lower):
            return True
            
    return False

def rewrite_query(query: str, chat_history: List[Dict[str, str]]) -> str:
    """
    Rewrites a query into a standalone question using chat history.
    """
    if not settings.query_rewrite_enabled or not chat_history:
        logger.info("Query rewriting skipped (disabled or no history).")
        return query
        
    if not is_ambiguous(query):
        logger.info("Query deemed unambiguous. Skipping rewrite.")
        return query
        
    logger.info("Ambiguity detected. Attempting to rewrite query.")
    
    # In a production environment, this would call an LLM (e.g., HuggingFace API).
    # Since the architecture forbids final generation here and we want deterministic fallback,
    # we implement a mock contextualizer for tests unless an API is strictly required.
    
    # Simple Mock Logic: Prepend the subject of the last user query.
    # E.g., if history is "What is the refund policy?" and query is "What about international?",
    # Rewrite: "[Context: refund policy] What about international?"
    
    last_user_query = ""
    for msg in reversed(chat_history):
        if msg["role"] == "user":
            last_user_query = msg["content"]
            break
            
    if last_user_query:
        # Extract a naive "subject" (last few words) for the mock
        subject = " ".join(last_user_query.split()[-3:]).strip("?")
        rewritten = f"Regarding {subject}, {query}"
        logger.info(f"Mock rewritten query: '{rewritten}'")
        return rewritten

    return query
