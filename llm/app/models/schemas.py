from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ProcessRequest(BaseModel):
    """Schema for process request"""
    prompt: str
    files: Optional[List[str]] = None

class ContentAnalysis(BaseModel):
    """Schema for content analysis"""
    scene_description: str
    identified_objects: List[str]
    activities_detected: List[str]

class SecurityAnalysis(BaseModel):
    """Schema for security analysis"""
    risk_level: str
    concerns: List[str]
    recommendations: List[str]

class SentimentAnalysis(BaseModel):
    """Schema for sentiment analysis"""
    overall_sentiment: str
    emotional_indicators: Dict[str, float]

class ProcessResponse(BaseModel):
    """Schema for process response"""
    filenames: List[str]
    prompt_name: str
    analysis: str
    content_analysis: Optional[ContentAnalysis] = None
    security_analysis: Optional[SecurityAnalysis] = None
    sentiment_analysis: Optional[SentimentAnalysis] = None
    model_version: str = "Qwen2.5-VL"
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now)

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