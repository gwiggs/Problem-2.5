from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class MediaMetadata(BaseModel):
    """Schema for media metadata."""
    filename: str
    file_type: str
    size: int
    created_at: datetime = Field(default_factory=datetime.now)
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    additional_metadata: Dict[str, Any] = Field(default_factory=dict)

class FileUploadResponse(BaseModel):
    """Schema for file upload response."""
    message: str
    filename: str
    metadata: MediaMetadata

class FileListResponse(BaseModel):
    """Schema for file list response."""
    files: list[str]

class ErrorResponse(BaseModel):
    """Schema for error responses."""
    detail: str
    status_code: int 

class LLMPrompt(BaseModel):
    """Schema for LLM prompts."""
    name: str
    description: str
    template: str

class LLMRequest(BaseModel):
    """Schema for LLM analysis request."""
    filenames: List[str]
    prompt_name: str

class LLMResponse(BaseModel):
    """Schema for LLM analysis response."""
    filenames: List[str]
    prompt_name: str
    analysis: str