# fastapi_app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from classes.backend_class_example import HelloWorld
import os
from mimetypes import guess_type
from pymediainfo import MediaInfo
import json


app = FastAPI()

UPLOAD_DIR = "uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

METADATA_DIR = "metadata"
os.makedirs(METADATA_DIR, exist_ok=True)

@app.get("/")
def hello_world():
    exampleClass = HelloWorld()
    return {"Message": exampleClass.get()}

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    metadata_path = os.path.join(METADATA_DIR, f"{file.filename}.json")  # Define metadata_path here

    # Save the uploaded video
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        # Extract metadata
        media_info = MediaInfo.parse(file_path)
        metadata = media_info.to_data()

        # Save metadata as JSON
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metadata extraction failed: {str(e)}")

    return {
        "message": f"File '{file.filename}' uploaded successfully!",
        "path": file_path,
        "metadata_path": metadata_path,
    }

@app.get("/video/{filename}")
async def get_video(filename: str):
    """
    Serve a video file from the uploaded_videos directory.
    """
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        mime_type, _ = guess_type(file_path)
        # Default to "video/mp4" if MIME type is not detected
        mime_type = mime_type or "video/mp4"
        return FileResponse(file_path, media_type=mime_type, filename=filename)
    return {"error": "File not found"}

@app.get("/videos/")
async def list_videos():
    """
    List all video files in the uploaded_videos directory.
    """
    if not os.path.exists(UPLOAD_DIR):
        return {"videos": []}
    
    files = os.listdir(UPLOAD_DIR)
    video_files = [file for file in files if os.path.isfile(os.path.join(UPLOAD_DIR, file))]
    return {"videos": video_files}

@app.delete("/video/{filename}")
async def delete_video(filename: str):
    """
    Delete a video file from the uploaded_videos directory.
    """
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return {"message": f"Video '{filename}' deleted successfully!"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")
    else:
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/metadata/{video}")
async def get_metadata(video: str):
    """
    Retrieve metadata for a specific video.
    """
    metadata_path = os.path.join(METADATA_DIR, f"{video}.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading metadata: {str(e)}")
    else:
        raise HTTPException(status_code=404, detail="Metadata not found")