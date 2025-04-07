from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import List, Dict, Any, Optional
import shutil
import os
import torch
import numpy as np
from PIL import Image
import cv2
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import subprocess
import traceback
import sys
import signal
import atexit
from safetensors.torch import load_file
import safetensors
# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global variables for model and processor
model = None
processor = None
device = None
is_shutting_down = False
model_path = "/app/Qwen2.5-VL-3B-Instruct"

# Create the quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # or "fp4"
    bnb_4bit_compute_dtype="float16"  # or "float16" if needed
)

def cleanup_resources():
    """Clean up resources before shutdown"""
    global model, processor, device, is_shutting_down
    
    if is_shutting_down:
        return
        
    is_shutting_down = True
    logger.info("Cleaning up resources...")
    
    try:
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            logger.info("Clearing CUDA cache...")
            torch.cuda.empty_cache()
            logger.info(f"CUDA memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # Set model to None to help garbage collection
        if model is not None:
            logger.info("Releasing model...")
            model = None
            
        if processor is not None:
            logger.info("Releasing processor...")
            processor = None
            
        logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        logger.error(traceback.format_exc())

# Register cleanup function to run on exit
atexit.register(cleanup_resources)

def signal_handler(sig, frame):
    """Handle termination signals"""
    logger.info(f"Received signal {sig}, initiating graceful shutdown...")
    cleanup_resources()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def check_gpu_availability():
    """Check GPU availability and log detailed information"""
    logger.info("Checking GPU availability...")
    
    # Check CUDA availability in PyTorch
    logger.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    
    # Check NVIDIA-SMI
    try:
        nvidia_smi = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        logger.info(f"NVIDIA-SMI output:\n{nvidia_smi.decode()}")
    except Exception as e:
        logger.warning(f"Failed to run nvidia-smi: {str(e)}")
    
    # Check environment variables
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    logger.info(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
    logger.info(f"NVIDIA_VISIBLE_DEVICES: {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'Not set')}")
    logger.info(f"NVIDIA_DRIVER_CAPABILITIES: {os.environ.get('NVIDIA_DRIVER_CAPABILITIES', 'Not set')}")
    
    # Check PyTorch version and CUDA version
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version (from PyTorch): {torch.version.cuda if hasattr(torch.version, 'cuda') else 'Not available'}")
    
    return torch.cuda.is_available()

def log_gpu_info():
    """Log GPU information"""
    if torch.cuda.is_available():
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"GPU memory before inference: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    else:
        logger.info("CUDA not available, using CPU")

def check_flash_attention():
    """Check if flash attention is available"""
    try:
        import flash_attn
        logger.info(f"Flash Attention 2 is available: {flash_attn.__version__}")
        return True
    except ImportError:
        logger.warning("Flash Attention 2 is not available, will use standard attention")
        return False

def load_model_primary(model_path, gpu_available, use_flash_attention):
    """Primary model loading method"""
    logger.info("Attempting to load model with primary method")
    model_kwargs = {
        "device_map": "auto" if gpu_available else "cpu",
        "low_cpu_mem_usage": True,
        "quantization_config": bnb_config,
    }
    
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Using Flash Attention 2")
    else:
        logger.info("Using standard attention")
    
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        **model_kwargs
    )

def load_model_alternative(model_path, gpu_available, use_flash_attention):
    """Alternative model loading method with explicit device mapping"""
    logger.info("Attempting to load model with alternative method")
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "quantization_config": bnb_config
    }
    
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Using Flash Attention 2")
    else:
        logger.info("Using standard attention")
    
    if gpu_available:
        device_map = {
            "model": 0,  # Main model
            "vision_model": 0,  # Vision encoder
            "language_model": 0,  # Language model
            "vision_projection": 0,  # Vision projection
            "language_projection": 0,  # Language projection
            "lm_head": 0,  # Language model head
            "vision_model.embeddings": 0,  # Vision embeddings
            "vision_model.encoder": 0,  # Vision encoder
            "language_model.embeddings": 0,  # Language embeddings
            "language_model.encoder": 0,  # Language encoder
            "language_model.decoder": 0  # Language decoder
        }
        model_kwargs["device_map"] = device_map
        logger.info(f"Using explicit device map: {device_map}")
    else:
        model_kwargs["device_map"] = "cpu"
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        **model_kwargs
    )
    
    if gpu_available and not hasattr(model, "device"):
        logger.info("Moving model to GPU explicitly")
        model = model.to(device)
        
    return model

def load_model_minimal(model_path, gpu_available):
    """Minimal model loading method with basic settings"""
    logger.info("Attempting to load model with minimal settings")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        device_map=None  # Don't use device_map at all
    )
    
    if gpu_available:
        logger.info("Moving model to GPU explicitly")
        model = model.to(device)
        
    return model
def load_tokenizer(model_path):
    """Load the tokenizer for the model"""
    logger.info("Loading tokenizer")
    return AutoTokenizer.from_pretrained(model_path)

def load_processor(model_path):
    """Load the processor for the model"""
    logger.info("Loading processor")
    return AutoProcessor.from_pretrained(model_path)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, processor, device, is_shutting_down
    logger.info("Initializing Qwen2.5-VL model...")
    try:
        # Check GPU availability
        logger.info("Step 1: Checking GPU availability")
        gpu_available = check_gpu_availability()
        
        if gpu_available:
            # Force CUDA device selection
            logger.info("Step 2: Setting up CUDA device")
            device = torch.device("cuda:0")
            log_gpu_info()
            
            # Set CUDA device explicitly
            torch.cuda.set_device(0)
        else:
            device = torch.device("cpu")
            logger.info("CUDA not available, using CPU")
        
        # Log PyTorch version
        logger.info(f"PyTorch version: {torch.__version__}")
        
        # Check if flash-attn is installed
        use_flash_attention = check_flash_attention()
        
        # Initialize model with explicit device placement
        logger.info("Step 3: Loading model from pretrained")
        logger.info(f"Model path: {os.path.abspath(model_path)}")
        logger.info(f"Model path exists: {os.path.exists(model_path)}")
        
        if os.path.exists(model_path):
            logger.info(f"Model path contents: {os.listdir(model_path)}")
        else:
            logger.error(f"Model path does not exist: {model_path}")
            logger.info("Current directory contents:")
            logger.info(os.listdir("/app"))
            logger.info("Attempting to create model directory...")
            os.makedirs(model_path, exist_ok=True)
            logger.info(f"Model directory created: {os.path.exists(model_path)}")
        
        try:
            # Try primary loading method
            model = load_model_primary(model_path, gpu_available, use_flash_attention)
            logger.info("Model loaded successfully with primary method")
        except Exception as model_error:
            logger.error(f"Failed to load model with primary method: {str(model_error)}")
            logger.error(traceback.format_exc())
            
            try:
                # Try alternative loading method
                model = load_model_alternative(model_path, gpu_available, use_flash_attention)
                logger.info("Model loaded successfully with alternative method")
            except Exception as alt_error:
                logger.error(f"Failed to load model with alternative method: {str(alt_error)}")
                logger.error(traceback.format_exc())
                
                try:
                    # Try minimal loading method
                    model = load_model_minimal(model_path, gpu_available)
                    logger.info("Model loaded successfully with minimal settings")
                except Exception as final_error:
                    logger.error(f"Failed to load model with minimal settings: {str(final_error)}")
                    logger.error(traceback.format_exc())
                    raise
        
        # Move model to device explicitly if needed
        if gpu_available and not hasattr(model, "device"):
            logger.info("Moving model to device explicitly")
            model = model.to(device)
        # Load tokenizer
        tokenizer = load_tokenizer(model_path)
        logger.info("Tokenizer loaded successfully")
        
        # Load processor
        processor = load_processor(model_path)
        logger.info("Processor loaded successfully")
        
        # Log model device
        if hasattr(model, "device"):
            logger.info(f"Model device: {model.device}")
        else:
            logger.info("Model device information not available")
            
        logger.info("Model initialization completed successfully")
            
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    yield
    # Shutdown
    logger.info("Shutting down...")
    cleanup_resources()

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def is_video_file(filename: str) -> bool:
    """Check if the file is a video based on extension"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    return os.path.splitext(filename.lower())[1] in video_extensions

def is_image_file(filename: str) -> bool:
    """Check if the file is an image based on extension"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    return os.path.splitext(filename.lower())[1] in image_extensions

async def process_video_frames(video_path: str, num_frames: int = 128) -> tuple:
    """Extract frames and timestamps from video using OpenCV"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        # Sample frames uniformly
        indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
        frames = []
        timestamps = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                timestamps.append(idx / fps)
            else:
                logger.warning(f"Could not read frame at index {idx}")
        
        cap.release()
        
        if not frames:
            raise ValueError("No frames could be extracted from the video")
        
        return np.array(frames), np.array(timestamps), total_frames
    except Exception as e:
        logger.error(f"Error processing video frames: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

async def process_image(image_path: str) -> Image.Image:
    """Process a single image file"""
    try:
        return Image.open(image_path)
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

async def generate_response(prompt: str, media_files: List[str], media_types: List[str]) -> Dict[str, Any]:
    """Generate response using Qwen2.5-VL model"""
    global model, processor, device
    
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # Log device information before inference
        logger.info(f"Generating response on device: {device}")
        if torch.cuda.is_available():
            logger.info(f"GPU memory before inference: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # Prepare media content for the model
        media_content = []
        
        for file_path, media_type in zip(media_files, media_types):
            if media_type == "video":
                # Process video frames
                frames, timestamps, total_frames = await process_video_frames(file_path)
                
                # Convert frames to PIL Images
                pil_frames = [Image.fromarray(frame) for frame in frames]
                
                # Add frames to media content
                for frame in pil_frames:
                    media_content.append({"image": frame})
                
                # Add video metadata to prompt
                prompt = f"{prompt} [Video: {os.path.basename(file_path)}, {total_frames} frames, duration: {timestamps[-1]:.2f}s]"
            else:
                # Process image
                image = await process_image(file_path)
                media_content.append({"image": image})
        
        # Process inputs with the processor
        inputs = processor(
            text=prompt,
            images=media_content,
            return_tensors="pt"
        ).to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        # Decode the generated text
        response = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Log GPU memory after inference
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

@app.post("/process/")
async def process_request(
    prompt: str = Form(...),
    files: List[UploadFile] = File(None)
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Save uploaded files
    saved_files = []
    media_types = []
    
    for file in files:
        file_location = f"/tmp/{file.filename}"
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
        response = await generate_response(prompt, saved_files, media_types)
        
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

@app.get("/health")
async def health_check():
    """Health check endpoint with detailed status information."""
    global model, processor, device
    
    status = {
        "status": "healthy" if model is not None and processor is not None else "unhealthy",
        "model_loaded": model is not None,
        "processor_loaded": processor is not None,
        "device": str(device) if device is not None else "Not set",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "cuda_memory_allocated": f"{torch.cuda.memory_allocated() / 1e9:.2f} GB" if torch.cuda.is_available() else "N/A",
        "cuda_memory_cached": f"{torch.cuda.memory_reserved() / 1e9:.2f} GB" if torch.cuda.is_available() else "N/A",
        "model_path": os.path.abspath(model_path),
        "model_path_exists": os.path.exists(model_path),
        "model_path_contents": os.listdir(model_path) if os.path.exists(model_path) else [],
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