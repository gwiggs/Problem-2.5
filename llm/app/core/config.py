import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / ".cache"

# Model paths
MODEL_PATH = "/app/Qwen2.5-VL-3B-Instruct"

# API settings
API_PREFIX = "/api"
HEALTH_ENDPOINT = "/health"
PROCESS_ENDPOINT = "/process"

# File settings
TEMP_DIR = "/tmp"
UPLOADED_FILES_DIR = "/app/uploaded_files"

# Video processing settings
DEFAULT_NUM_FRAMES = 128

# Model generation settings
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# File extensions
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

# Model settings
DEFAULT_MODEL_PATH = str(MODELS_DIR / "Qwen2.5-VL-3B-Instruct")
DEFAULT_DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

# API settings
API_V1_STR = "/api/v1"
PROJECT_NAME = "LLM Service"

# Create necessary directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True) 