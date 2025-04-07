import os
from pathlib import Path

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
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'} 