import logging
import os
import shutil
import time
import json
import torch
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from PIL import Image
from pydantic import BaseModel

from qwen_vl_utils import process_vision_info

from app.core.config import TEMP_DIR, MODEL_PATH
from app.core.model_manager import ModelManager
from app.models.analysis_schema import (
    AnalysisResponse,
    ContentAnalysis,
    SecurityAnalysis,
    SentimentAnalysis,
    DEFAULT_SCHEMA
)

from app.utils.media_processor import (
    is_video_file,
    is_image_file,
    process_video_frames,
    process_image
)

from torch.cuda.amp import autocast, GradScaler  # Import for mixed precision

logger = logging.getLogger(__name__)
router = APIRouter()

# Optimize CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.6"

def log_memory_usage():
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"GPU Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

# Dependency to get the model manager
def get_model_manager():
    from app.core.lifespan import model_manager
    return model_manager

# Log the value of the environment variable
logger.info(f"PYTORCH_CUDA_ALLOC_CONF: {os.getenv('PYTORCH_CUDA_ALLOC_CONF')}")

async def generate_response(prompt: str, media_files: List[str], media_types: List[str], model_manager: ModelManager) -> Dict[str, Any]:
    """Generate response using Qwen2.5-VL model with proper video frame handling"""
    model = model_manager.get_model()
    processor = model_manager.get_processor()
    device = model_manager.get_device()
    
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        start_time = time.time()
        logger.info(f"Generating response on device: {device}")
        
        # Clear GPU memory before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log_memory_usage()
        
        # Parameters for video processing
        total_pixels = 10240 * 28 * 28  # Same as reference implementation
        min_pixels = 16 * 28 * 28       # Same as reference implementation
        
        # Prepare messages in chat format
        messages = []
        image_inputs = []
        video_inputs = []
        fps_list = []
        
        # Process media files in smaller batches
        batch_size = 2  # Reduced batch size
        for i in range(0, len(media_files), batch_size):
            batch_files = media_files[i:i + batch_size]
            batch_types = media_types[i:i + batch_size]
            
            for file_path, media_type in zip(batch_files, batch_types):
                if media_type == "video":
                    # Process video frames with reduced frame count and resolution
                    frames, timestamps, total_frames = process_video_frames(
                        file_path, 
                        max_frames=16,  # Further limit to 16 frames
                        target_size=(224, 224)  # Further reduce resolution
                    )
                    
                    # Calculate FPS using the last timestamp
                    if len(timestamps) > 0:
                        last_timestamp = timestamps[-1]
                        fps = len(frames) / last_timestamp if last_timestamp > 0 else 30.0
                    else:
                        fps = 30.0
                    
                    message = {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"video": file_path, "total_pixels": total_pixels, "min_pixels": min_pixels}
                        ]
                    }
                    messages.append(message)
                    
                    # Convert frames to tensor format and move to GPU immediately
                    with torch.cuda.amp.autocast():
                        video_array = np.array(frames)
                        video_tensor = torch.from_numpy(video_array).permute(0, 3, 1, 2).to(device)  # [T, C, H, W]
                        video_inputs.append(video_tensor)
                        fps_list.append(fps)
                    
                    # Log video frame info
                    num_frames, _, height, width = video_tensor.shape
                    logger.info(f"Video input shape: {video_tensor.shape}")
                    logger.info(f"Number of video tokens: {int(num_frames / 2 * height / 28 * width / 28)}")
                    
                else:
                    # Process image and move to GPU immediately
                    with torch.cuda.amp.autocast():
                        image = process_image(file_path)
                        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(device)
                        image_inputs.append(image_tensor)
                        message = {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"image": image}
                            ]
                        }
                        messages.append(message)
            
            # Clear GPU memory after processing each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                log_memory_usage()
        
        # Add system message with minimal guidance
        system_message = {
            "role": "system",
            "content": "You are a helpful assistant that analyzes media content. Provide your analysis based on the user's prompt."
        }
        messages.insert(0, system_message)
        
        # Apply chat template with GPU acceleration
        with torch.cuda.amp.autocast():
            text = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        
        # Initialize GradScaler for mixed precision
        scaler = GradScaler()
        
        # Process all inputs with mixed precision and batch processing
        with torch.inference_mode(), torch.cuda.amp.autocast():
            for i in range(0, len(video_inputs), batch_size):
                batch_videos = video_inputs[i:i + batch_size]
                batch_fps = fps_list[i:i + batch_size]
                
                # Process inputs directly on GPU
                processor_output = processor(
                    text=[text],
                    images=image_inputs if image_inputs else None,
                    videos=batch_videos,
                    fps=batch_fps,
                    padding=True,
                    return_tensors="pt"
                )
                
                # Move all tensors to device
                inputs = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in processor_output.items()
                }
                
                # Generate response for batch
                with torch.cuda.amp.autocast():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        use_cache=True  # Enable KV cache for faster generation
                    )
                
                # Extract only the generated part
                input_length = inputs.get('input_ids', inputs.get('pixel_values', torch.tensor([]))).shape[1]
                generated_ids = output_ids[:, input_length:]
                response = processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=True
                )[0]
                
                # Clear GPU memory after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    log_memory_usage()
        
        if torch.cuda.is_available():
            log_memory_usage()
            torch.cuda.empty_cache()
        
        # Return the raw response
        return {
            "result": {
                "raw_response": response,
                "metadata": {
                    "input_files": [os.path.basename(f) for f in media_files],
                    "prompt": prompt,
                    "device": str(device),
                    "processing_time": time.time() - start_time
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        logger.error(f"Full error details: {str(e.__class__.__name__)}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

# Define the simplified response model
class SimpleResponse(BaseModel):
    raw_response: str
    metadata: Dict[str, Any]

class ProcessResponse(BaseModel):
    result: SimpleResponse

@router.post("/process/", response_model=ProcessResponse)
async def process_request(
    prompt: str = Form(...),
    files: List[UploadFile] = File(None),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Process a request with text and media files"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Save uploaded files
    saved_files = []
    media_types = []
    
    for file in files:
        file_location = f"{TEMP_DIR}/{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file_location)
        
        # Determine media type
        if is_video_file(file.filename):
            media_types.append("video")
        elif is_image_file(file.filename):
            media_types.append("image")
        else:
            # Clean up the file
            os.remove(file_location)
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.filename}"
            )
    
    try:
        # Process with Qwen model
        response = await generate_response(prompt, saved_files, media_types, model_manager)
        
        # Clean up temporary files
        for file_path in saved_files:
            os.remove(file_path)
            
        return response
        
    except Exception as e:
        # Clean up temporary files in case of error
        for file_path in saved_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@router.get("/health")
async def health_check(model_manager: ModelManager = Depends(get_model_manager)):
    """Health check endpoint with detailed status information."""
    model = model_manager.get_model()
    processor = model_manager.get_processor()
    tokenizer = model_manager.get_tokenizer()
    device = model_manager.get_device()
    
    return {
        "status": "healthy" if model is not None and processor is not None else "unhealthy",
        "model_loaded": model is not None,
        "processor_loaded": processor is not None,
        "tokenizer_loaded": tokenizer is not None,
        "device": str(device) if device is not None else "Not set",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "cuda_memory_allocated": f"{torch.cuda.memory_allocated() / 1e9:.2f} GB" if torch.cuda.is_available() else "N/A",
        "cuda_memory_cached": f"{torch.cuda.memory_reserved() / 1e9:.2f} GB" if torch.cuda.is_available() else "N/A",
        "model_path": os.path.abspath(MODEL_PATH),
        "model_path_exists": os.path.exists(MODEL_PATH),
        "model_path_contents": os.listdir(MODEL_PATH) if os.path.exists(MODEL_PATH) else [],
        "current_directory": os.getcwd(),
        "current_directory_contents": os.listdir(os.getcwd()),
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"),
            "CUDA_HOME": os.environ.get("CUDA_HOME", "Not set"),
            "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES", "Not set"),
            "NVIDIA_DRIVER_CAPABILITIES": os.environ.get("NVIDIA_DRIVER_CAPABILITIES", "Not set"),
            "PYTORCH_VERSION": torch.__version__,
            "CUDA_VERSION": torch.version.cuda if hasattr(torch.version, "cuda") else "Not available"
        }
    } 