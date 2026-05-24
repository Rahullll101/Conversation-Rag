from pydantic import BaseModel

class UploadResponse(BaseModel):
    status: str
    session_id: str
    filename: str
    file_type: str
    text_length: int
    text_preview: str
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "success",
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "filename": "document.pdf",
                "file_type": "application/pdf",
                "text_length": 1024,
                "text_preview": "This is the start of the document..."
            }
        }
    }
