import logging
import io

logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts text from PDF bytes using pypdf gracefully."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(file_bytes))
        text_pages = []
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text_pages.append(extracted)
        return "\n\n".join(text_pages).strip()
    except Exception as e:
        logger.error(f"Failed to extract PDF text: {e}")
        raise ValueError("Corrupted or unreadable PDF file.") from e

def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extracts text from TXT bytes gracefully."""
    try:
        return file_bytes.decode('utf-8').strip()
    except UnicodeDecodeError as e:
        logger.error(f"Failed to decode TXT text: {e}")
        raise ValueError("Invalid UTF-8 encoding in TXT file.") from e
