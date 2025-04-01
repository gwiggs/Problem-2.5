from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import json
import re

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for JSON data
json_data = []

@app.post("/upload-json/")
async def upload_json(file: UploadFile = File(...)):
    global json_data
    try:
        contents = await file.read()
        json_data = json.loads(contents)
        return {"message": "File uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# backend/app/main.py
@app.get("/get-data/")
async def get_data() -> List[Dict[str, Any]]:
    return json_data

@app.get("/search/")
async def search(query: str = "") -> List[Dict[str, Any]]:
    if not json_data:
        return []
    
    results = []
    or_groups = [g.strip() for g in query.split(" OR ")] if query else []

    for item in json_data:
        if not isinstance(item, dict):
            continue
        
        # Always show all data when no search query
        if not query:
            results.append(item)
            continue
            
        # Process OR groups
        matched = False
        for group in or_groups:
            and_terms = [t.strip() for t in group.split(" AND ")] if group else []
            
            # Process AND terms
            group_match = True
            for term in and_terms:
                term_match = False
                for value in item.values():
                    try:
                        if re.search(term, str(value), re.IGNORECASE):
                            term_match = True
                            break
                    except re.error:
                        continue
                if not term_match:
                    group_match = False
                    break
            if group_match:
                matched = True
                break
                
        if matched:
            results.append(item)
    
    return results
