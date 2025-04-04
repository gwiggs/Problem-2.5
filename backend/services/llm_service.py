import httpx
from typing import List
#from pathlib import Path
from fastapi import HTTPException
from config.settings import LLM_SERVICE_URL
from models.schemas import LLMPrompt, LLMResponse

#Define default prompts
DEFAULT_PROMPTS = {
    "transcription": LLMPrompt(
    name="Transcription",
    description="Transcribe the audio in the video file.",
    prompt="You are a transcription service. You will be given a video file and you need to transcribe it into text.",
    ),
    "summary": LLMPrompt(
        name="Summary",
        description="Summarize the content of the video file.",
        prompt="You are a summary service. You will be given a video file and you need to summarize it into a few sentences.",
    ),
    "describe_video": LLMPrompt(
        name="Describe Video",
        description="Describe the content of the video file.",
        prompt="You are a description service. You will be given a video file and you need to describe it in detail.",
    ),
    "analyze_sentiment": LLMPrompt(
        name="Analyze Sentiment",
        description="Analyze the sentiment of the video file.",
        prompt="You are a sentiment analysis service. You will be given a video file and you need to analyze the sentiment of the video file.",
    )
}

class LLMService:
    """Service for interacting with the LLM."""

    @staticmethod
    async def get_available_prompts() -> List[LLMPrompt]:
        """Get all available prompts."""
        return list(DEFAULT_PROMPTS.values())
    
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
                    f"{LLM_SERVICE_URL}/analyze",
                    json={
                        "files": filenames,
                        "prompt": prompt.template
                    }
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
                    analysis=data["analysis"]
                )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to analyze videos: {str(e)}"
            )



