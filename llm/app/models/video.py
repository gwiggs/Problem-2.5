from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class VideoChunkResponse(BaseModel):
    """Response model for a single video chunk processing result."""
    chunk_index: int
    total_chunks: int
    num_frames: int
    start_time: float
    end_time: float
    processed_data: Dict[str, Any]

class VideoProcessingResponse(BaseModel):
    """Response model for video processing results."""
    success: bool
    message: str
    data: Dict[str, Any] 