import logging
from typing import List
from sentence_transformers import SentenceTransformer
from backend.config.settings import settings

logger = logging.getLogger(__name__)

# Lazy initialization of the model
_model = None

def get_model():
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {settings.embedding_model_name}")
        _model = SentenceTransformer(settings.embedding_model_name)
    return _model

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates embeddings for a list of texts."""
    logger.info(f"Generating embeddings for {len(texts)} texts")
    model = get_model()
    # model.encode returns numpy arrays, we convert to lists of floats
    embeddings = model.encode(texts, convert_to_numpy=True).tolist()
    return embeddings
