# import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploaded_files"
METADATA_DIR = BASE_DIR / "metadata"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
METADATA_DIR.mkdir(exist_ok=True)

# Allowed file types
ALLOWED_EXTENSIONS = {
    "video": [".mp4", ".avi", ".mov"],
    "image": [".jpg", ".jpeg", ".png"]
}

# Maximum file size (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024 

# LLM service URL
LLM_SERVICE_URL = "http://localhost:8100"