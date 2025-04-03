from typing import List, Optional
import requests
from models.schemas import FileResponse, FileListResponse, FileMetadata, ErrorResponse

class APIClient:
    """Client for handling API communication with the backend."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
    
    def upload_file(self, file_data: bytes, filename: str, content_type: str) -> FileResponse:
        """Upload a file to the backend."""
        files = {"file": (filename, file_data, content_type)}
        response = requests.post(f"{self.base_url}/upload/", files=files)
        
        if response.status_code != 200:
            raise ValueError(f"Upload failed: {response.text}")
            
        return FileResponse(**response.json())
    
    def get_files(self) -> List[str]:
        """Get list of all uploaded files."""
        response = requests.get(f"{self.base_url}/files/")
        
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch files: {response.text}")
            
        return FileListResponse(**response.json()).files
    
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
            
        return FileMetadata(**response.json())
    
    def delete_file(self, filename: str) -> None:
        """Delete a file by filename."""
        response = requests.delete(f"{self.base_url}/file/{filename}")
        
        if response.status_code != 200:
            raise ValueError(f"Failed to delete file: {response.text}") 