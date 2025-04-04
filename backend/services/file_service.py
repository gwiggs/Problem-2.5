# import os
import json
from pathlib import Path
from typing import List, Optional
from fastapi import UploadFile, HTTPException
from pymediainfo import MediaInfo
from config.settings import UPLOAD_DIR, METADATA_DIR, MAX_FILE_SIZE
from models.schemas import MediaMetadata#, FileUploadResponse

class FileService:
    """Service for handling file operations."""
    
    @staticmethod
    async def save_file(file: UploadFile) -> Path:
        """Save uploaded file and return its path."""
        if file.size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE/1024/1024}MB"
            )
            
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        return file_path
    
    @staticmethod
    def extract_metadata(file_path: Path) -> MediaMetadata:
        """Extract metadata from file using MediaInfo."""
        try:
            media_info = MediaInfo.parse(str(file_path))
            data = media_info.to_data()
            
            # Extract relevant metadata
            tracks = data.get("tracks", [])
            general_track = next((track for track in tracks if track.get("track_type") == "General"), {})
            video_track = next((track for track in tracks if track.get("track_type") == "Video"), {})
            
            return MediaMetadata(
                filename=file_path.name,
                file_type=general_track.get("file_extension", ""),
                size=general_track.get("file_size", 0),
                duration=general_track.get("duration", None),
                width=video_track.get("width", None),
                height=video_track.get("height", None),
                format=video_track.get("format", None),
                additional_metadata=data
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to extract metadata: {str(e)}"
            )
    
    @staticmethod
    def save_metadata(metadata: MediaMetadata) -> Path:
        """Save metadata to JSON file."""
        metadata_path = METADATA_DIR / f"{metadata.filename}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.dict(), f, indent=4, default=str)
        return metadata_path
    
    @staticmethod
    def get_file_list() -> List[str]:
        """Get list of all uploaded files."""
        return [f.name for f in UPLOAD_DIR.iterdir() if f.is_file()]
    
    @staticmethod
    def get_file_path(filename: str) -> Optional[Path]:
        """Get file path if it exists."""
        file_path = UPLOAD_DIR / filename
        return file_path if file_path.exists() else None
    
    @staticmethod
    def delete_file(filename: str) -> None:
        """Delete file and its metadata."""
        file_path = UPLOAD_DIR / filename
        metadata_path = METADATA_DIR / f"{filename}.json"
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
            
        try:
            file_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete file: {str(e)}"
            )
    
    @staticmethod
    def get_metadata(filename: str) -> MediaMetadata:
        """Get metadata for a file."""
        metadata_path = METADATA_DIR / f"{filename}.json"
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail="Metadata not found")
            
        try:
            with open(metadata_path, "r") as f:
                data = json.load(f)
            return MediaMetadata(**data)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read metadata: {str(e)}"
            ) 