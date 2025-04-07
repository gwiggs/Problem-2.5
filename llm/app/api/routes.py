import logging
import os
import shutil
import torch
from typing import List, Dict, Any
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from PIL import Image

from app.core.config import TEMP_DIR, MODEL_PATH
from app.core.model_manager import ModelManager
from app.models.schemas import ProcessResponse, HealthResponse
from app.utils.media_processor import (
    is_video_file,
    is_image_file,
    process_video_frames,
    process_image
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Dependency to get the model manager
def get_model_manager():
    from app.core.lifespan import model_manager
    return model_manager

async def generate_response(prompt: str, media_files: List[str], media_types: List[str], model_manager: ModelManager) -> Dict[str, Any]:
    """Generate response using Qwen2.5-VL model with proper video frame handling"""
    model = model_manager.get_model()
    processor = model_manager.get_processor()
    device = model_manager.get_device()
    
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        logger.info(f"Generating response on device: {device}")
        if torch.cuda.is_available():
            logger.info(f"GPU memory before inference: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # Parameters for video processing
        total_pixels = 20480 * 28 * 28  # Same as reference implementation
        min_pixels = 16 * 28 * 28       # Same as reference implementation
        
        # Prepare messages in chat format
        messages = []
        image_inputs = []
        video_inputs = []
        fps_list = []
        
        for file_path, media_type in zip(media_files, media_types):
            if media_type == "video":
                # Process video frames
                frames, timestamps, total_frames = await process_video_frames(file_path)
                fps = len(frames) / timestamps[-1] if timestamps[-1] > 0 else 30.0
                
                message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"video": file_path, "total_pixels": total_pixels, "min_pixels": min_pixels}
                    ]
                }
                messages.append(message)
                
                # Convert frames to tensor format
                video_tensor = torch.tensor(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
                video_inputs.append(video_tensor)
                fps_list.append(fps)
                
                # Log video frame info
                num_frames, _, height, width = video_tensor.shape
                logger.info(f"Video input shape: {video_tensor.shape}")
                logger.info(f"Number of video tokens: {int(num_frames / 2 * height / 28 * width / 28)}")
                
            else:
                # Process image
                image = await process_image(file_path)
                image_inputs.append(image)
                message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"image": image}
                    ]
                }
                messages.append(message)
        
        # Add system message
        messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})
        
        # Apply chat template
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process all inputs
        inputs = processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            fps=fps_list if fps_list else None,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            # Extract only the generated part
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            response = processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )[0]
        
        if torch.cuda.is_available():
            logger.info(f"GPU memory after inference: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        return {
            "prompt": prompt,
            "result": response,
            "processed_files": [os.path.basename(f) for f in media_files],
            "device": str(device)
        }
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

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

@router.get("/health", response_model=HealthResponse)
async def health_check(model_manager: ModelManager = Depends(get_model_manager)):
    """Health check endpoint with detailed status information."""
    model = model_manager.get_model()
    processor = model_manager.get_processor()
    tokenizer = model_manager.get_tokenizer()
    device = model_manager.get_device()
    
    status = {
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
    
    # If model is loaded, add model-specific information
    if model is not None:
        status["model_device"] = str(model.device) if hasattr(model, "device") else "Unknown"
        status["model_dtype"] = str(next(model.parameters()).dtype) if hasattr(model, "parameters") else "Unknown"
    
    return status 