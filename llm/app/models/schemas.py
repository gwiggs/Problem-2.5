from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class ProcessRequest(BaseModel):
    """Schema for process request"""
    prompt: str
    files: Optional[List[str]] = None

class ProcessResponse(BaseModel):
    """Schema for process response"""
    prompt: str
    result: str
    processed_files: List[str]
    device: str

class HealthResponse(BaseModel):
    """Schema for health check response"""
    status: str
    model_loaded: bool
    processor_loaded: bool
    tokenizer_loaded: bool
    device: str
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str
    cuda_memory_allocated: str
    cuda_memory_cached: str
    model_path: str
    model_path_exists: bool
    model_path_contents: List[str]
    current_directory: str
    current_directory_contents: List[str]
    environment: Dict[str, Any]
    model_device: Optional[str] = None
    model_dtype: Optional[str] = None 