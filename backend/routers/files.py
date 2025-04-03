from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
# from typing import List
from mimetypes import guess_type

from services.file_service import FileService
from models.schemas import FileListResponse, FileUploadResponse

router = APIRouter(prefix="/files", tags=["files"])

@router.post("/upload/", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a file and extract its metadata."""
    file_path = await FileService.save_file(file)
    metadata = FileService.extract_metadata(file_path)
    FileService.save_metadata(metadata)
    
    return FileUploadResponse(
        message=f"File '{file.filename}' uploaded successfully!",
        filename=file.filename,
        metadata=metadata
    )

@router.get("/{filename}")
async def get_file(filename: str):
    """Serve a file from the upload directory."""
    file_path = FileService.get_file_path(filename)
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found")
        
    mime_type, _ = guess_type(str(file_path))
    return FileResponse(str(file_path), media_type=mime_type, filename=filename)

@router.get("/", response_model=FileListResponse)
async def list_files():
    """List all uploaded files."""
    files = FileService.get_file_list()
    return FileListResponse(files=files)

@router.delete("/{filename}")
async def delete_file(filename: str):
    """Delete a file and its metadata."""
    FileService.delete_file(filename)
    return {"message": f"File '{filename}' deleted successfully!"}

@router.get("/metadata/{filename}")
async def get_metadata(filename: str):
    """Get metadata for a specific file."""
    return FileService.get_metadata(filename) 