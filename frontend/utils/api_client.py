from typing import List, Optional, Dict, Any
import requests
from models.schemas import (FileResponse, FileListResponse, FileMetadata, ErrorResponse, LLMPrompt, LLMResponse, LLMRequest)

class APIClient:
    """Client for handling API communication with the backend."""
    
    def __init__(self, base_url: str, llm_api_url: str):
        self.base_url = base_url.rstrip('/')
        self.llm_api_url = llm_api_url.rstrip('/')
    
    def _extract_metadata(self, data: Dict[str, Any], filename: str) -> FileMetadata:
        """Helper method to extract metadata from response data."""

        tracks = data.get("tracks", [])
        general_track = next((track for track in tracks if track.get("track_type") == "General"), {})
        video_track = next((track for track in tracks if track.get("track_type") == "Video"), {})
        
        return FileMetadata(
            filename=filename,
            file_type=general_track.get("file_extension", ""),
            size=general_track.get("file_size", 0),
            created_at=general_track.get("file_last_modification_date", ""),
            duration=general_track.get("duration", None),
            width=video_track.get("width", None),
            height=video_track.get("height", None),
            format=general_track.get("format", None),
            additional_metadata=data
        )
    
    def upload_file(self, file_data: bytes, filename: str, content_type: str) -> FileResponse:
        """Upload a file to the backend."""
        files = {"file": (filename, file_data, content_type)}
        response = requests.post(f"{self.base_url}/upload/", files=files)
        
        if response.status_code != 200:
            raise ValueError(f"Upload failed: {response.text}")
            
        data = response.json()
        return FileResponse(
            message=data.get("message", "File uploaded successfully"),
            filename=data.get("filename", filename),
            metadata=self._extract_metadata(data, data.get("filename", filename))
        )
    
    def get_files(self) -> List[str]:
        """Get list of all uploaded files."""
        response = requests.get(f"{self.base_url}/files/")
        
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch files: {response.text}")
            
        data = response.json()
        return data.get("files", [])
    
    def get_file(self, filename: str) -> bytes:
        """Get file content by filename."""
        response = requests.get(f"{self.base_url}/file/{filename}")
        
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch file: {response.text}")
            
        return response.content
    
    def get_metadata(self, filename: str) -> FileMetadata:
        """Get file metadata by filename."""
        response = requests.get(f"{self.base_url}/metadata/{filename}")
        
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch metadata: {response.text}")
            
        data = response.json()
        return self._extract_metadata(data, filename)
    
    def delete_file(self, filename: str) -> None:
        """Delete a file by filename."""
        response = requests.delete(f"{self.base_url}/file/{filename}")
        
        if response.status_code != 200:
            raise ValueError(f"Failed to delete file: {response.text}")
            
        data = response.json()
        if not data.get("message", "").startswith("File deleted successfully"):
            raise ValueError(f"Unexpected response: {data.get('message', 'Unknown error')}") 

    def get_llm_prompts(self) -> List[LLMPrompt]:
        """Get available LLM prompts."""
        response = requests.get(f"{self.base_url}/llm/prompts")
        
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch prompts: {response.text}")
            
        return [LLMPrompt(**p) for p in response.json()]

    def add_prompt(self, prompt: LLMPrompt) -> None:
        """Add a new prompt."""
        response = requests.post(
            f"{self.base_url}/llm/prompts",
            json=prompt.model_dump()
        )
        
        if response.status_code != 200:
            raise ValueError(f"Failed to add prompt: {response.text}")

    def update_prompt(self, prompt: LLMPrompt) -> None:
        """Update an existing prompt."""
        response = requests.put(
            f"{self.base_url}/llm/prompts/{prompt.name}",
            json=prompt.model_dump()
        )
        
        if response.status_code != 200:
            raise ValueError(f"Failed to update prompt: {response.text}")

    def delete_prompt(self, prompt_name: str) -> None:
        """Delete a prompt."""
        response = requests.delete(
            f"{self.base_url}/llm/prompts/{prompt_name}"
        )
        
        if response.status_code != 200:
            raise ValueError(f"Failed to delete prompt: {response.text}")

    def analyze_videos(self, filenames: List[str], prompt_name: str) -> LLMResponse:
        """Send videos for LLM analysis."""
        # First get the prompt template
        prompts = self.get_llm_prompts()
        prompt = next((p for p in prompts if p.name == prompt_name), None)
        if not prompt:
            raise ValueError(f"Prompt '{prompt_name}' not found")
            
        # Prepare files for upload
        files = []
        for filename in filenames:
            file_content = self.get_file(filename)
            files.append(('files', (filename, file_content, 'application/octet-stream')))
            
        # Send request with form data
        response = requests.post(
            f"{self.llm_api_url}/process/",
            files=files,
            data={'prompt': prompt.template}
        )
        
        if response.status_code != 200:
            raise ValueError(f"Analysis failed: {response.text}")
            
        data = response.json()
        
        # Extract the result which contains raw_response and metadata
        result = data.get('result', {})
        if not isinstance(result, dict):
            raise ValueError(f"Unexpected response format: {data}")
            
        return LLMResponse(
            filenames=filenames,
            prompt_name=prompt_name,
            analysis=result  # This is now a dict with raw_response and metadata
        )