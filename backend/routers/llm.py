from fastapi import APIRouter
from typing import List
from models.schemas import LLMPrompt, LLMRequest, LLMResponse
from services.llm_service import LLMService

router = APIRouter(prefix="/llm", tags=["llm"])

@router.get("/prompts", response_model=List[LLMPrompt])
async def get_prompts():
    """Get available LLM prompts."""
    return await LLMService.get_available_prompts()

@router.post("/analyze", response_model=LLMResponse)
async def analyze_videos(request: LLMRequest):
    """Analyze videos using LLM."""
    return await LLMService.analyze_videos(request.filenames, request.prompt_name)