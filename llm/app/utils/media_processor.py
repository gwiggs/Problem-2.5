import logging
import os
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image
import decord
from decord import VideoReader, cpu
from fastapi import HTTPException
from app.core.config import (
    VIDEO_EXTENSIONS,
    IMAGE_EXTENSIONS,
    DEFAULT_NUM_FRAMES,
    CACHE_DIR
)
import cv2
import torch
import av

logger = logging.getLogger(__name__)

def is_video_file(file_path: str) -> bool:
    """Check if a file is a video based on its extension."""
    return any(file_path.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)

def is_image_file(file_path: str) -> bool:
    """Check if a file is an image based on its extension."""
    return any(file_path.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)

def create_image_grid(images: List[np.ndarray], num_columns: int = 8) -> Image.Image:
    """Create a grid of images from a list of numpy arrays."""
    pil_images = [Image.fromarray(image) for image in images]
    num_rows = (len(images) + num_columns - 1) // num_columns

    img_width, img_height = pil_images[0].size
    grid_width = num_columns * img_width
    grid_height = num_rows * img_height
    grid_image = Image.new('RGB', (grid_width, grid_height))

    for idx, image in enumerate(pil_images):
        row_idx = idx // num_columns
        col_idx = idx % num_columns
        position = (col_idx * img_width, row_idx * img_height)
        grid_image.paste(image, position)

    return grid_image

def process_video_frames(
    file_path: str, 
    max_frames: int = 16,
    target_size: Optional[Tuple[int, int]] = (224, 224)
) -> Tuple[List[np.ndarray], List[float], int]:
    """
    Process a video file and extract frames with caching.
    
    Args:
        file_path: Path to the video file
        max_frames: Maximum number of frames to extract (default: 16)
        target_size: Target size for resizing frames (width, height) or None for original size
        
    Returns:
        Tuple containing:
        - List of frames as numpy arrays
        - List of timestamps
        - Total number of frames in the video
    """
    # Create cache directory if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Generate cache key based on file path and parameters
    cache_key = f"{os.path.basename(file_path)}_{max_frames}_{target_size}"
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.npz")
    
    # Check if cached frames exist
    if os.path.exists(cache_file):
        try:
            logger.info(f"Loading cached frames for {file_path}")
            cached_data = np.load(cache_file, allow_pickle=True)
            frames = cached_data['frames']
            timestamps = cached_data['timestamps'].tolist()
            total_frames = int(cached_data['total_frames'])
            logger.info(f"Frames cached for future use")
            return frames, timestamps, total_frames
        except Exception as e:
            logger.warning(f"Error loading cached frames: {str(e)}")
            # Continue with processing if cache loading fails
    
    try:
        # Use decord for efficient video loading
        video_reader = decord.VideoReader(file_path)
        total_frames = len(video_reader)
        
        # Calculate frame indices to sample evenly
        if total_frames > max_frames:
            # Sample frames evenly across the video
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        else:
            # Use all frames if video is shorter than max_frames
            frame_indices = np.arange(total_frames)
        
        # Extract frames in smaller chunks
        frames = []
        timestamps = []
        chunk_size = 4  # Process 4 frames at a time
        
        for i in range(0, len(frame_indices), chunk_size):
            chunk_indices = frame_indices[i:i + chunk_size]
            
            for idx in chunk_indices:
                # Get frame and timestamp
                frame = video_reader[idx].asnumpy()
                timestamp = float(idx) / video_reader.get_avg_fps()
                
                # Resize if target size is provided
                if target_size is not None:
                    frame = cv2.resize(frame, target_size)
                
                frames.append(frame)
                timestamps.append(timestamp)
            
            # Clear CUDA cache after each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Cache the frames for future use
        try:
            np.savez(
                cache_file,
                frames=frames,
                timestamps=np.array(timestamps),
                total_frames=total_frames
            )
            logger.info(f"Frames cached for future use")
        except Exception as e:
            logger.warning(f"Error caching frames: {str(e)}")
        
        return frames, timestamps, total_frames
        
    except Exception as e:
        logger.error(f"Error processing video {file_path}: {str(e)}")
        raise

def process_image(image_path: str) -> Image.Image:
    """Process a single image file."""
    try:
        if not os.path.exists(image_path):
            raise HTTPException(status_code=400, detail=f"Image file not found: {image_path}")
        return Image.open(image_path)
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}") 