from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

# Core enums and base types
class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    THREATENING = "threatening"

# Base response types
class AnalysisBase(BaseModel):
    """Base class for all analysis responses"""
    model_version: str = "Qwen2.5-VL"
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.now)

class LLMAnalysisResponse(AnalysisBase):
    """Main response schema for LLM container"""
    security_analysis: Dict[str, Any]
    document_analysis: Dict[str, Any]
    facial_analysis: Dict[str, Any]
    content_analysis: Dict[str, Any]
    sentiment_analysis: Dict[str, Any]
    political_content: List[Dict[str, Any]]
    timeline: Optional[List[Dict[str, Any]]] = None
    raw_model_output: Optional[str] = None

# Interfaces for analysis components
class DetectedFace(BaseModel):
    timestamp?: float
    framePosition?: Dict[str, float]
    isObscured: bool
    expression: str
    confidence: float
    screenshotUrl?: str

class SecurityConcern(BaseModel):
    category: str
    description: str
    timestamp?: float
    confidence: float
    screenshotUrl?: str
    riskLevel: RiskLevel

# Response interfaces
class SecurityAnalysisResponse(BaseModel):
    overallRiskLevel: RiskLevel
    securityConcerns: List[SecurityConcern]
    recommendations: List[str]

class AnalysisResponse(BaseModel):
    fileId: str
    fileMetadata: Dict[str, Any]
    security: SecurityAnalysisResponse
    faces: List[DetectedFace]
    documents: List[Any]
    sentiment: Any
    politicalContent: List[Any]

# API Request schemas
class AnalysisRequest(BaseModel):
    file_ids: List[str]
    analysis_types: List[str]
    priority: Optional[str] = "normal" 