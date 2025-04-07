import logging
import os
from typing import Tuple, List
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
    video_path: str,
    num_frames: int = DEFAULT_NUM_FRAMES,
    use_cache: bool = True
) -> Tuple[List[np.ndarray], List[float], int]:
    """
    Extract frames from a local video file using decord for efficient processing.
    
    Args:
        video_path: Path to the local video file
        num_frames: Number of frames to extract
        use_cache: Whether to use cached frames if available
        
    Returns:
        Tuple containing:
        - List of frames as numpy arrays
        - List of timestamps
        - Total number of frames in video
    """
    try:
        if not os.path.exists(video_path):
            raise HTTPException(status_code=400, detail=f"Video file not found: {video_path}")

        # Create cache directory if it doesn't exist
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Generate cache key from file path and modification time
        file_stat = os.stat(video_path)
        cache_key = f"{os.path.basename(video_path)}_{file_stat.st_mtime}_{num_frames}"
        
        # Check cache
        if use_cache:
            frames_cache_file = os.path.join(CACHE_DIR, f'{cache_key}_frames.npy')
            timestamps_cache_file = os.path.join(CACHE_DIR, f'{cache_key}_timestamps.npy')
            
            if os.path.exists(frames_cache_file) and os.path.exists(timestamps_cache_file):
                logger.info("Loading frames from cache")
                frames = np.load(frames_cache_file)
                timestamps = np.load(timestamps_cache_file)
                return frames, timestamps, len(frames)

        # Load video using decord
        logger.info(f"Loading video: {video_path}")
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        
        if total_frames == 0:
            raise HTTPException(status_code=400, detail="Video file is empty or corrupted")

        # Sample frames uniformly
        indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        timestamps = np.array([vr.get_frame_timestamp(idx) for idx in indices])

        # Cache the results
        if use_cache:
            np.save(frames_cache_file, frames)
            np.save(timestamps_cache_file, timestamps)
            logger.info("Frames cached for future use")

        return frames, timestamps, total_frames

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")

def process_image(image_path: str) -> Image.Image:
    """Process a single image file."""
    try:
        if not os.path.exists(image_path):
            raise HTTPException(status_code=400, detail=f"Image file not found: {image_path}")
        return Image.open(image_path)
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}") 