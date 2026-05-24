import uuid
import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

def generate_session_id() -> str:
    """Generates a new unique session ID."""
    return str(uuid.uuid4())

def create_session_directory(upload_dir: str, session_id: str) -> Path:
    """Creates a directory for the session."""
    session_dir = Path(upload_dir) / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir

def save_file(path: Path, content: bytes) -> None:
    """Saves raw bytes to a file."""
    path.write_bytes(content)

def save_metadata(session_dir: Path, metadata: dict) -> None:
    """Saves metadata.json in the session directory."""
    metadata_path = session_dir / "metadata.json"
    with metadata_path.open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

def cleanup_session_directory(session_dir: Path) -> None:
    """Removes a session directory and its contents."""
    if session_dir.exists() and session_dir.is_dir():
        try:
            shutil.rmtree(session_dir)
            logger.info(f"Cleaned up session directory: {session_dir}")
        except Exception as e:
            logger.error(f"Failed to clean up session directory {session_dir}: {e}")
