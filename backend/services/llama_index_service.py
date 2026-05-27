import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from backend.config.settings import settings
from backend.utils.chunking import create_chunk_metadata

logger = logging.getLogger(__name__)

_EMBED_MODEL = None


class LlamaIndexUnavailableError(RuntimeError):
    """Raised when the configured LlamaIndex runtime packages are missing."""


def _require_llama_index():
    try:
        from llama_index.core import Document, SimpleDirectoryReader, StorageContext, VectorStoreIndex
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.schema import MetadataMode
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.vector_stores.chroma import ChromaVectorStore
    except ImportError as exc:
        raise LlamaIndexUnavailableError(
            "LlamaIndex dependencies are not installed. Install requirements.txt "
            "or set RAG_ENGINE=legacy."
        ) from exc

    return {
        "Document": Document,
        "SimpleDirectoryReader": SimpleDirectoryReader,
        "StorageContext": StorageContext,
        "VectorStoreIndex": VectorStoreIndex,
        "SentenceSplitter": SentenceSplitter,
        "MetadataMode": MetadataMode,
        "HuggingFaceEmbedding": HuggingFaceEmbedding,
        "ChromaVectorStore": ChromaVectorStore,
    }


def _get_embed_model():
    global _EMBED_MODEL
    modules = _require_llama_index()
    if _EMBED_MODEL is None:
        logger.info("Loading LlamaIndex embedding model: %s", settings.embedding_model_name)
        _EMBED_MODEL = modules["HuggingFaceEmbedding"](
            model_name=settings.embedding_model_name
        )
    return _EMBED_MODEL


def _session_paths(session_id: str) -> tuple[Path, Path, Path]:
    session_dir = Path(settings.upload_dir) / f"session_{session_id}"
    return session_dir, session_dir / "extracted_text.txt", session_dir / "metadata.json"


def _load_session_payload(session_id: str) -> tuple[Path, str, Dict[str, Any]]:
    session_dir, extracted_text_path, metadata_path = _session_paths(session_id)

    if not session_dir.exists():
        raise ValueError("Session directory not found.")
    if not extracted_text_path.exists():
        raise ValueError(
            "extracted_text.txt not found. Did document extraction complete successfully?"
        )
    if not metadata_path.exists():
        raise ValueError("metadata.json not found.")

    text = extracted_text_path.read_text(encoding="utf-8")
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    return session_dir, text, metadata


def _load_documents(session_dir: Path, text: str, metadata: Dict[str, Any]) -> list:
    modules = _require_llama_index()
    Document = modules["Document"]
    SimpleDirectoryReader = modules["SimpleDirectoryReader"]

    source_filename = Path(metadata.get("original_filename", "unknown")).name
    original_path = session_dir / source_filename
    base_metadata = {
        "session_id": metadata.get("session_id"),
        "source_filename": source_filename,
        "original_file_type": metadata.get("content_type", "unknown"),
    }

    if original_path.exists():
        try:
            documents = SimpleDirectoryReader(input_files=[str(original_path)]).load_data()
            if documents:
                for document in documents:
                    document.metadata = {**(document.metadata or {}), **base_metadata}
                return documents
        except Exception as exc:
            logger.warning(
                "LlamaIndex file loader failed for %s; falling back to extracted text: %s",
                original_path,
                exc,
            )

    return [Document(text=text, metadata=base_metadata)]


def _build_nodes(session_id: str, text: str, metadata: Dict[str, Any], documents: list) -> list:
    modules = _require_llama_index()
    SentenceSplitter = modules["SentenceSplitter"]
    MetadataMode = modules["MetadataMode"]

    source_filename = Path(metadata.get("original_filename", "unknown")).name
    original_file_type = metadata.get("content_type", "unknown")
    upload_timestamp = str(int(time.time()))

    parser = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    nodes = parser.get_nodes_from_documents(documents, show_progress=False)

    if not nodes and text:
        fallback_document = modules["Document"](
            text=text,
            metadata={
                "session_id": session_id,
                "source_filename": source_filename,
                "original_file_type": original_file_type,
            },
        )
        nodes = parser.get_nodes_from_documents([fallback_document], show_progress=False)

    processed_nodes = []
    for index, node in enumerate(nodes):
        node_text = node.get_content(metadata_mode=MetadataMode.NONE)
        chunk_metadata = create_chunk_metadata(
            session_id=session_id,
            chunk_index=index,
            text=node_text,
            source_filename=source_filename,
            upload_timestamp=upload_timestamp,
            original_file_type=original_file_type,
        )

        source_metadata = getattr(node, "metadata", {}) or {}
        page_label = source_metadata.get("page_label") or source_metadata.get("page")
        if isinstance(page_label, int):
            chunk_metadata["page_number"] = page_label
        elif isinstance(page_label, str) and page_label.isdigit():
            chunk_metadata["page_number"] = int(page_label)

        node.id_ = chunk_metadata["chunk_id"]
        node.metadata = chunk_metadata
        node.excluded_embed_metadata_keys = list(chunk_metadata.keys())
        node.excluded_llm_metadata_keys = list(chunk_metadata.keys())
        processed_nodes.append(node)

    return processed_nodes


def _get_chroma_collection(session_id: str, reset: bool = False, create: bool = True):
    from backend.services.vector_store_service import client, get_collection_name

    collection_name = get_collection_name(session_id)
    if reset:
        try:
            client.delete_collection(name=collection_name)
            logger.info("Deleted existing LlamaIndex collection for session %s", session_id)
        except ValueError:
            pass
        except Exception as exc:
            logger.warning("Could not delete existing collection %s: %s", collection_name, exc)

    if create:
        return client.get_or_create_collection(name=collection_name)
    return client.get_collection(name=collection_name)


def index_session_document(session_id: str) -> Dict[str, Any]:
    """
    Build the per-session Chroma index through LlamaIndex.

    This keeps the external API unchanged while making LlamaIndex responsible for
    loading, node parsing, indexing, vector storage, and later retrieval.
    """
    modules = _require_llama_index()
    StorageContext = modules["StorageContext"]
    VectorStoreIndex = modules["VectorStoreIndex"]
    ChromaVectorStore = modules["ChromaVectorStore"]

    session_dir, text, metadata = _load_session_payload(session_id)
    documents = _load_documents(session_dir=session_dir, text=text, metadata=metadata)
    nodes = _build_nodes(
        session_id=session_id,
        text=text,
        metadata=metadata,
        documents=documents,
    )

    if not nodes:
        raise ValueError("Chunking resulted in 0 chunks.")

    chroma_collection = _get_chroma_collection(session_id=session_id, reset=True, create=True)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=_get_embed_model(),
    )

    logger.info("Indexed %s nodes through LlamaIndex for session %s", len(nodes), session_id)
    return {
        "status": "success",
        "session_id": session_id,
        "total_chunks": len(nodes),
        "embedding_model": settings.embedding_model_name,
        "rag_engine": "llamaindex",
    }


def retrieve_session_chunks(session_id: str, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
    modules = _require_llama_index()
    VectorStoreIndex = modules["VectorStoreIndex"]
    ChromaVectorStore = modules["ChromaVectorStore"]
    MetadataMode = modules["MetadataMode"]

    if top_k is None:
        top_k = settings.retrieval_top_k

    try:
        chroma_collection = _get_chroma_collection(
            session_id=session_id,
            reset=False,
            create=False,
        )
    except Exception as exc:
        logger.error("Failed to get LlamaIndex collection for session %s: %s", session_id, exc)
        raise ValueError(f"Session {session_id} not found in database or not processed yet.")

    if chroma_collection.count() == 0:
        return []

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=_get_embed_model(),
    )
    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(query)

    retrieved = []
    for result in results:
        node = result.node
        metadata = dict(getattr(node, "metadata", {}) or {})
        node_id = (
            metadata.get("chunk_id")
            or getattr(node, "node_id", None)
            or getattr(node, "id_", None)
        )
        retrieved.append(
            {
                "chunk_id": node_id,
                "text": node.get_content(metadata_mode=MetadataMode.NONE),
                "score": float(result.score or 0.0),
                "metadata": metadata,
            }
        )

    return retrieved
