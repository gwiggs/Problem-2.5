from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

class EmotionalIndicators(BaseModel):
    """Schema for emotional indicators in sentiment analysis."""
    happiness: float = Field(ge=0.0, le=1.0, description="Happiness score between 0 and 1")
    sadness: float = Field(ge=0.0, le=1.0, description="Sadness score between 0 and 1")
    anger: float = Field(ge=0.0, le=1.0, description="Anger score between 0 and 1")
    fear: float = Field(ge=0.0, le=1.0, description="Fear score between 0 and 1")

class ContentAnalysis(BaseModel):
    """Schema for content analysis section."""
    scene_description: str = Field(description="Detailed description of the scene")
    identified_objects: List[str] = Field(description="List of objects identified in the scene")
    activities_detected: List[str] = Field(description="List of activities detected in the scene")
    key_events: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="List of key events with timestamps if applicable"
    )

class SecurityAnalysis(BaseModel):
    """Schema for security analysis section."""
    risk_level: str = Field(description="Overall risk level assessment")
    concerns: List[str] = Field(description="List of security concerns identified")
    recommendations: List[str] = Field(description="List of security recommendations")
    threat_indicators: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="List of specific threat indicators identified"
    )

class SentimentAnalysis(BaseModel):
    """Schema for sentiment analysis section."""
    overall_sentiment: str = Field(description="Overall sentiment description")
    emotional_indicators: EmotionalIndicators = Field(description="Quantitative emotional indicators")
    behavioral_indicators: Optional[List[str]] = Field(
        default=None,
        description="List of behavioral indicators observed"
    )

class AnalysisResponse(BaseModel):
    """Schema for the complete analysis response."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique request identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the analysis")
    model_version: str = Field(description="Version of the model used for analysis")
    processing_time: float = Field(description="Time taken to process the request in seconds")
    content_analysis: ContentAnalysis = Field(description="Content analysis results")
    security_analysis: SecurityAnalysis = Field(description="Security analysis results")
    sentiment_analysis: SentimentAnalysis = Field(description="Sentiment analysis results")
    metadata: Optional[Dict[str, str]] = Field(
        default=None,
        description="Additional metadata about the analysis"
    )

# Default schema for LLM system message
DEFAULT_SCHEMA = {
    "content_analysis": {
        "scene_description": "Detailed description of the scene",
        "identified_objects": ["List of objects identified"],
        "activities_detected": ["List of activities detected"],
        "key_events": [{"timestamp": "Time of event", "description": "Event description"}]
    },
    "security_analysis": {
        "risk_level": "LOW|MEDIUM|HIGH",
        "concerns": ["List of security concerns"],
        "recommendations": ["List of security recommendations"],
        "threat_indicators": [{"type": "Threat type", "description": "Threat description"}]
    },
    "sentiment_analysis": {
        "overall_sentiment": "Overall sentiment description",
        "emotional_indicators": {
            "happiness": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0
        },
        "behavioral_indicators": ["List of behavioral indicators"]
    }
} 