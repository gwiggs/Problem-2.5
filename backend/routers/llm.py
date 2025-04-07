from fastapi import APIRouter, HTTPException
from typing import List
from models.schemas import LLMPrompt, LLMRequest, LLMResponse
from services.llm_service import LLMService

router = APIRouter(prefix="/llm", tags=["llm"])

@router.get("/prompts", response_model=List[LLMPrompt])
async def get_prompts():
    """Get available LLM prompts."""
    return await LLMService.get_available_prompts()

@router.post("/prompts", response_model=LLMPrompt)
async def add_prompt(prompt: LLMPrompt):
    """Add a new LLM prompt."""
    return await LLMService.add_prompt(prompt)

@router.put("/prompts/{prompt_name}", response_model=LLMPrompt)
async def update_prompt(prompt_name: str, prompt: LLMPrompt):
    """Update an existing LLM prompt."""
    return await LLMService.update_prompt(prompt_name, prompt)

@router.delete("/prompts/{prompt_name}")
async def delete_prompt(prompt_name: str):
    """Delete an LLM prompt."""
    await LLMService.delete_prompt(prompt_name)
    return {"message": f"Prompt '{prompt_name}' deleted successfully"}

@router.post("/analyze", response_model=LLMResponse)
async def analyze_videos(request: LLMRequest):
    """Analyze videos using LLM."""
    return await LLMService.analyze_videos(request.filenames, request.prompt_name)