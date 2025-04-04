from fastapi import FastAPI, File, UploadFile, Form
from typing import List
import shutil

app = FastAPI()

@app.post("/process/")
async def process_request(
    prompt: str = Form(...),
    files: List[UploadFile] = File(None)
):
    # Save uploaded files
    saved_files = []
    for file in files:
        file_location = f"/tmp/{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file_location)

    # Simulate LLM Processing (Replace this with actual model inference)
    response = {
        "prompt": prompt,
        "processed_files": saved_files,
        "result": f"Processed '{prompt}' with {len(saved_files)} files."
    }
    return response