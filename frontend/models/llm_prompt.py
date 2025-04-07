from pydantic import BaseModel, Field
from typing import Optional

class LLMPrompt(BaseModel):
    """Model for LLM prompts."""
    name: str = Field(..., description="Unique name identifier for the prompt")
    description: str = Field(..., description="Description of what the prompt does")
    template: str = Field(..., description="The actual prompt template text")
    is_default: bool = Field(default=False, description="Whether this is the default prompt")
    version: Optional[str] = Field(default=None, description="Version of the prompt")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "security_analysis",
                "description": "Analyzes video content for security concerns",
                "template": "Analyze the following video content for security concerns...",
                "is_default": True,
                "version": "1.0"
            }
        } 