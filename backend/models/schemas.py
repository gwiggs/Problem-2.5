from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

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

# Import shared enums
class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    THREATENING = "threatening"

# Detailed component schemas
class DetectedFace(BaseModel):
    timestamp: Optional[float] = None
    frame_position: Optional[Dict[str, float]] = None
    is_obscured: bool
    expression: str
    confidence: float
    screenshot_url: Optional[str] = None

class DetectedDocument(BaseModel):
    """Schema for detected documents in media."""
    document_type: str
    timestamp: Optional[float] = None  # For videos
    frame_position: Optional[Dict[str, float]] = None
    confidence: float
    extracted_text: Optional[str] = None
    is_sensitive: bool = False
    screenshot_url: Optional[str] = None

class SecurityConcern(BaseModel):
    category: str
    description: str
    timestamp: Optional[float] = None
    confidence: float
    screenshot_url: Optional[str] = None
    risk_level: RiskLevel

class PoliticalContent(BaseModel):
    """Schema for detected political content."""
    content_type: str  # e.g., "symbol", "text", "gesture"
    description: str
    timestamp: Optional[float] = None
    affiliation: Optional[str] = None
    confidence: float
    screenshot_url: Optional[str] = None

class TimelineEvent(BaseModel):
    """Schema for events in video timeline."""
    timestamp: float
    event_type: str
    description: str
    importance_level: int  # 1-5
    screenshot_url: Optional[str] = None

# API Response schemas
class SecurityAnalysisResponse(BaseModel):
    overall_risk_level: RiskLevel
    security_concerns: List[SecurityConcern]
    recommendations: List[str]

class AnalysisResponse(BaseModel):
    file_id: str
    file_metadata: MediaMetadata
    security: SecurityAnalysisResponse
    faces: List[DetectedFace]
    documents: List[Dict[str, Any]]
    sentiment: Dict[str, Any]
    political_content: List[Dict[str, Any]]
    
# API Request schemas
class AnalysisRequest(BaseModel):
    file_ids: List[str]
    analysis_types: List[str]
    priority: Optional[str] = "normal"

class DocumentAnalysisResponse(BaseModel):
    """Schema for document analysis results."""
    detected_documents: List[DetectedDocument]
    sensitive_information_found: bool
    redaction_recommended: bool
    document_types_summary: Dict[str, int]  # Count of each document type

class FacialAnalysisResponse(BaseModel):
    """Schema for facial analysis results."""
    total_faces_detected: int
    unique_faces: int
    detected_faces: List[DetectedFace]
    crowd_estimate: Optional[int] = None
    face_clustering: Optional[Dict[str, List[str]]] = None  # Groups similar faces

class ContentAnalysisResponse(BaseModel):
    """Schema for general content analysis."""
    scene_description: str
    identified_objects: List[str]
    activities_detected: List[str]
    environment_details: Dict[str, Any]
    notable_events: List[TimelineEvent]
    content_warnings: List[str]

class SentimentAnalysisResponse(BaseModel):
    """Schema for sentiment analysis results."""
    overall_sentiment: SentimentType
    sentiment_timeline: Optional[List[Dict[str, Any]]] = None  # For videos
    emotional_indicators: Dict[str, float]
    behavioral_concerns: List[str]
    group_dynamics: Optional[Dict[str, Any]] = None
    confidence_score: float

# Main response schema that combines all analyses
class ComprehensiveAnalysisResponse(BaseModel):
    """Schema for comprehensive media analysis results."""
    file_info: MediaMetadata
    security_analysis: Optional[SecurityAnalysisResponse] = None
    document_analysis: Optional[DocumentAnalysisResponse] = None
    facial_analysis: Optional[FacialAnalysisResponse] = None
    content_analysis: Optional[ContentAnalysisResponse] = None
    sentiment_analysis: Optional[SentimentAnalysisResponse] = None
    political_content: Optional[List[PoliticalContent]] = None
    timeline: Optional[List[TimelineEvent]] = None
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    processing_time: float
    model_version: str = "Qwen2.5-VL"

class LLMResponse(BaseModel):
    """Schema for LLM analysis response."""
    filenames: List[str]
    prompt_name: str
    analysis: ComprehensiveAnalysisResponse