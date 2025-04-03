from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class FileMetadata(BaseModel):
    """Schema for file metadata."""
    filename: str
    file_type: str
    size: int
    created_at: str
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    additional_metadata: Dict[str, Any] = Field(default_factory=dict)

class FileResponse(BaseModel):
    """Schema for file response."""
    message: str
    filename: str
    metadata: FileMetadata

class FileListResponse(BaseModel):
    """Schema for file list response."""
    files: List[str]

class ErrorResponse(BaseModel):
    """Schema for error responses."""
    detail: str
    status_code: int 