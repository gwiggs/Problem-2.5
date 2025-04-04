from typing import List, Dict, Any#, Optional
import requests
from models.schemas import FileResponse, FileMetadata #FileListResponse, FileMetadata, ErrorResponse

class APIClient:
    """Client for handling API communication with the backend."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
    
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