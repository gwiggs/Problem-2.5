# fastapi_app/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from classes.backend_class_example import HelloWorld
import os
from mimetypes import guess_type

app = FastAPI()

UPLOAD_DIR = "uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def hello_world():
    exampleClass = HelloWorld()
    return {"Message": exampleClass.get()}

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return {"message": f"File '{file.filename}' uploaded successfully!", "path": file_path}

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