# fastapi_app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from classes.backend_class_example import HelloWorld
import os
from mimetypes import guess_type
from pymediainfo import MediaInfo
import json


app = FastAPI()

UPLOAD_DIR = "uploaded_files"  # Renamed folder
os.makedirs(UPLOAD_DIR, exist_ok=True)

METADATA_DIR = "metadata"
os.makedirs(METADATA_DIR, exist_ok=True)

@app.get("/")
def hello_world():
    exampleClass = HelloWorld()
    return {"Message": exampleClass.get()}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    metadata_path = os.path.join(METADATA_DIR, f"{file.filename}.json")

    # Save the uploaded file
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

@app.get("/file/{filename}")
async def get_file(filename: str):
    """
    Serve a file (video or image) from the uploaded_files directory.
    """
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        mime_type, _ = guess_type(file_path)
        return FileResponse(file_path, media_type=mime_type, filename=filename)
    return {"error": "File not found"}

@app.get("/files/")
async def list_files():
    """
    List all files (videos and images) in the uploaded_files directory.
    """
    if not os.path.exists(UPLOAD_DIR):
        return {"files": []}
    
    files = os.listdir(UPLOAD_DIR)
    uploaded_files = [file for file in files if os.path.isfile(os.path.join(UPLOAD_DIR, file))]
    return {"files": uploaded_files}

@app.delete("/file/{filename}")
async def delete_file(filename: str):
    """
    Delete a file (video or image) from the uploaded_files directory.
    """
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return {"message": f"File '{filename}' deleted successfully!"}
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