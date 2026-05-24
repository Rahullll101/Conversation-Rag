import logging
from pathlib import Path
from fastapi import UploadFile, HTTPException
from backend.config.settings import settings
from backend.utils.file_handling import (
    generate_session_id, create_session_directory, save_file, save_metadata, cleanup_session_directory
)
from backend.utils.extraction import extract_text_from_pdf, extract_text_from_txt
from backend.schemas.upload import UploadResponse

logger = logging.getLogger(__name__)

async def process_upload(file: UploadFile) -> UploadResponse:
    # 1. Validation
    if file.content_type not in settings.allowed_mime_types:
        logger.warning(f"Rejected unsupported MIME type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF and TXT are allowed.")
    
    file_bytes = await file.read()
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > settings.max_upload_size_mb:
        logger.warning(f"Rejected file exceeding size limit: {file_size_mb:.2f} MB")
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {settings.max_upload_size_mb} MB.")

    # 2. Session creation
    session_id = generate_session_id()
    session_dir = None
    try:
        session_dir = create_session_directory(settings.upload_dir, session_id)
        
        # 3. Save original file
        original_filename = file.filename or "unknown_file"
        original_file_path = session_dir / original_filename
        save_file(original_file_path, file_bytes)

        # 4. Extract Text
        if file.content_type == "application/pdf":
            extracted_text = extract_text_from_pdf(file_bytes)
        else:
            extracted_text = extract_text_from_txt(file_bytes)
            
        if not extracted_text:
            raise ValueError("No text could be extracted from the file.")

        # 5. Save extracted text
        extracted_text_path = session_dir / "extracted_text.txt"
        extracted_text_path.write_text(extracted_text, encoding="utf-8")

        # 6. Save metadata
        text_length = len(extracted_text)
        text_preview = extracted_text[:200] + ("..." if text_length > 200 else "")
        metadata = {
            "session_id": session_id,
            "original_filename": original_filename,
            "content_type": file.content_type,
            "text_length": text_length
        }
        save_metadata(session_dir, metadata)

        logger.info(f"Successfully processed upload for session {session_id}")
        
        return UploadResponse(
            status="success",
            session_id=session_id,
            filename=original_filename,
            file_type=file.content_type,
            text_length=text_length,
            text_preview=text_preview
        )

    except ValueError as e:
        logger.error(f"Extraction error for session {session_id}: {e}")
        if session_dir:
            cleanup_session_directory(session_dir)
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing upload for session {session_id}: {e}")
        if session_dir:
            cleanup_session_directory(session_dir)
        raise HTTPException(status_code=500, detail="Internal server error during upload processing.")
