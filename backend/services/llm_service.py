import httpx
from typing import List, Dict
from pathlib import Path
from fastapi import HTTPException
from config.settings import LLM_SERVICE_URL
from models.schemas import LLMPrompt, LLMResponse
from pydantic import BaseModel

#Define default prompts
DEFAULT_PROMPTS = {
    "security_analysis": LLMPrompt(
        name="Security Analysis",
        description="Comprehensive security analysis of media content",
        template="""Analyze this media for security concerns. Provide a structured analysis covering:
1. Violence/Aggression: Identify any violent acts, aggressive behavior, or threatening gestures
2. Weapons/Military: Note any firearms, military equipment, tactical gear, or weapons
3. Face Coverage: Identify if individuals have concealed or obscured their faces
4. Location Anomalies: Note if individuals appear to be in unexpected locations based on apparent ethnicity/dress
5. Group Affiliations: Identify any visible symbols, flags, or emblems associated with known extremist groups
6. Security Rating: Provide a risk assessment (Low/Medium/High) with explanation

Format the response as a structured report with clear sections."""
    ),
    
    "document_extraction": LLMPrompt(
        name="Document Detection",
        description="Detect and analyze visible documents in media",
        template="""Identify and analyze any visible documents in this media. Focus on:
1. Document Type: Identify the type of documents visible (ID cards, passports, letters, etc.)
2. Document Details: Note any visible text, numbers, or identifying information
3. Document Location: Describe where in the frame documents appear
4. Authenticity Indicators: Note any visible security features or potential signs of alteration
5. Privacy Concerns: Flag any sensitive personal information visible

Provide timestamps for videos where documents are visible."""
    ),
    
    "political_content": LLMPrompt(
        name="Political Content Analysis",
        description="Analyze political content and messaging",
        template="""Analyze this media for political content and messaging. Consider:
1. Political Symbols: Identify flags, emblems, or symbols of political movements
2. Political Messages: Note any visible text, slogans, or messages
3. Political Context: Describe any political events or gatherings shown
4. Political Figures: Identify any known political figures present
5. Political Sentiment: Assess the overall political tone (neutral/supportive/opposing)

Provide an objective analysis without political bias."""
    ),
    
    "facial_analysis": LLMPrompt(
        name="Facial Analysis",
        description="Detailed analysis of visible faces",
        template="""Analyze visible faces in this media. For each distinct person:
1. Face Visibility: Note if face is clearly visible, partially visible, or obscured
2. Facial Features: Describe notable features while maintaining ethical considerations
3. Facial Expression: Describe emotional expression
4. Face Location: Note position in frame
5. Face Timestamps: For videos, note when faces appear/disappear
6. Identity Protection: Flag if faces appear to be intentionally concealed

Maintain privacy and ethical considerations in descriptions."""
    ),
    
    "comprehensive_description": LLMPrompt(
        name="Comprehensive Description",
        description="Detailed description of media content",
        template="""Provide a comprehensive description of this media. Include:
1. Scene Overview: Describe the overall setting and context
2. People Present: Number of people, general appearance, activities
3. Notable Objects: Identify significant items or equipment
4. Environmental Details: Describe location, time of day, weather conditions
5. Activities: Describe main actions or events occurring
6. Audio Content: For videos, describe any significant sounds or speech
7. Sequence of Events: For videos, provide a timeline of major events

Maintain objective, factual descriptions."""
    ),
    
    "sentiment_analysis": LLMPrompt(
        name="Sentiment Analysis",
        description="Analyze emotional and behavioral content",
        template="""Analyze the emotional and behavioral content in this media:
1. Overall Mood: Describe the predominant emotional tone
2. Individual Emotions: Note emotions displayed by specific individuals
3. Group Dynamics: Analyze interactions and collective behavior
4. Behavioral Indicators: Note any concerning behavioral patterns
5. Environmental Mood: Describe how setting/context affects sentiment
6. Temporal Changes: For videos, note how emotions/behavior change over time

Provide specific examples supporting your analysis."""
    ),
    
    "threat_assessment": LLMPrompt(
        name="Threat Assessment",
        description="Comprehensive threat and risk analysis",
        template="""Conduct a detailed threat assessment of this media content:
1. Immediate Threats: Identify any clear and present dangers
2. Behavioral Indicators: Note concerning patterns or behaviors
3. Environmental Risks: Assess location-based security concerns
4. Group Dynamics: Analyze crowd behavior and potential risks
5. Weapons/Tools: Identify any items that could pose security risks
6. Risk Level: Provide an overall threat assessment (Low/Medium/High)
7. Recommendations: Suggest appropriate security responses

Maintain professional, objective analysis."""
    ),
    
    "temporal_analysis": LLMPrompt(
        name="Temporal Analysis",
        description="Time-based analysis of video content",
        template="""Provide a detailed timeline analysis of this video:
1. Key Events: List significant events with timestamps
2. People Tracking: Note when individuals enter/exit frame
3. Behavioral Changes: Track changes in behavior/mood over time
4. Environmental Changes: Note changes in setting/context
5. Critical Moments: Identify timestamps requiring closer review
6. Activity Patterns: Identify repeated behaviors or patterns

Format as a chronological timeline with clear timestamps."""
    )
}

class SecurityAnalysisResponse(BaseModel):
    risk_level: str
    violence_detected: bool
    weapons_detected: bool
    face_concealment: bool
    location_anomalies: bool
    group_affiliations: List[str]
    details: Dict[str, str]

class LLMService:
    """Service for interacting with the LLM."""

    @staticmethod
    async def get_available_prompts() -> List[LLMPrompt]:
        """Get all available prompts."""
        return list(DEFAULT_PROMPTS.values())
    
    @staticmethod
    async def add_prompt(prompt: LLMPrompt) -> LLMPrompt:
        """Add a new prompt."""
        if prompt.name in DEFAULT_PROMPTS:
            raise HTTPException(status_code=400, detail=f"Prompt with name '{prompt.name}' already exists")
        DEFAULT_PROMPTS[prompt.name] = prompt
        return prompt
    
    @staticmethod
    async def update_prompt(prompt_name: str, prompt: LLMPrompt) -> LLMPrompt:
        """Update an existing prompt."""
        if prompt_name not in DEFAULT_PROMPTS:
            raise HTTPException(status_code=404, detail=f"Prompt '{prompt_name}' not found")
        if prompt_name != prompt.name:
            # If name is being changed, check if new name already exists
            if prompt.name in DEFAULT_PROMPTS:
                raise HTTPException(status_code=400, detail=f"Prompt with name '{prompt.name}' already exists")
            # Remove old prompt and add new one
            del DEFAULT_PROMPTS[prompt_name]
        DEFAULT_PROMPTS[prompt.name] = prompt
        return prompt
    
    @staticmethod
    async def delete_prompt(prompt_name: str) -> None:
        """Delete a prompt."""
        if prompt_name not in DEFAULT_PROMPTS:
            raise HTTPException(status_code=404, detail=f"Prompt '{prompt_name}' not found")
        del DEFAULT_PROMPTS[prompt_name]
    
    @staticmethod
    async def analyze_videos(filenames: List[str], prompt_name: str) -> LLMResponse:
        """Send videos to LLM for analysis."""
        if prompt_name not in DEFAULT_PROMPTS:
            raise HTTPException(status_code=400, detail=f"Invalid prompt name: {prompt_name}")
            
        prompt = DEFAULT_PROMPTS[prompt_name]
        
        try:
            # Prepare the request to the LLM service
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{LLM_SERVICE_URL}/process/",
                    files=[('files', open(filename, 'rb')) for filename in filenames],
                    data={'prompt': prompt.template}
                )
                
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"LLM service error: {response.text}"
                    )
                    
                data = response.json()
                return LLMResponse(
                    filenames=filenames,
                    prompt_name=prompt_name,
                    analysis=data["result"]  # The result now contains raw_response and metadata
                )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to analyze videos: {str(e)}"
            )



