import json
from pathlib import Path
from typing import Dict, List
from models.schemas import LLMPrompt
from fastapi import HTTPException

class PromptStorage:
    """Service for persisting prompts to disk."""
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
    def load_prompts(self) -> Dict[str, LLMPrompt]:
        """Load prompts from disk."""
        try:
            if not self.storage_path.exists():
                return {}
                
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                return {name: LLMPrompt(**prompt_data) for name, prompt_data in data.items()}
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load prompts: {str(e)}"
            )
            
    def save_prompts(self, prompts: Dict[str, LLMPrompt]) -> None:
        """Save prompts to disk."""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(
                    {name: prompt.model_dump() for name, prompt in prompts.items()},
                    f,
                    indent=2
                )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save prompts: {str(e)}"
            )
            
    def add_prompt(self, prompt: LLMPrompt, prompts: Dict[str, LLMPrompt]) -> None:
        """Add a new prompt and save to disk."""
        if prompt.name in prompts:
            raise HTTPException(
                status_code=400,
                detail=f"Prompt with name '{prompt.name}' already exists"
            )
        prompts[prompt.name] = prompt
        self.save_prompts(prompts)
        
    def update_prompt(self, prompt_name: str, prompt: LLMPrompt, prompts: Dict[str, LLMPrompt]) -> None:
        """Update an existing prompt and save to disk."""
        if prompt_name not in prompts:
            raise HTTPException(
                status_code=404,
                detail=f"Prompt '{prompt_name}' not found"
            )
        if prompt_name != prompt.name and prompt.name in prompts:
            raise HTTPException(
                status_code=400,
                detail=f"Prompt with name '{prompt.name}' already exists"
            )
        if prompt_name != prompt.name:
            del prompts[prompt_name]
        prompts[prompt.name] = prompt
        self.save_prompts(prompts)
        
    def delete_prompt(self, prompt_name: str, prompts: Dict[str, LLMPrompt]) -> None:
        """Delete a prompt and save to disk."""
        if prompt_name not in prompts:
            raise HTTPException(
                status_code=404,
                detail=f"Prompt '{prompt_name}' not found"
            )
        del prompts[prompt_name]
        self.save_prompts(prompts) 