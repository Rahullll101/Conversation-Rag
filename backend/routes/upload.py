from fastapi import APIRouter, UploadFile, File
from backend.schemas.upload import UploadResponse
from backend.services.upload_service import process_upload

router = APIRouter(tags=["Document Upload"])

@router.post(
    "/upload", 
    response_model=UploadResponse,
    summary="Upload a document for processing",
    description="Accepts a PDF or TXT file, validates it, creates a new session, extracts text, and stores the results."
)
async def upload_document(file: UploadFile = File(...)):
    return await process_upload(file)
