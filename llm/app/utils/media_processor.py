import logging
import os
import cv2
import numpy as np
from PIL import Image
from fastapi import HTTPException

from app.core.config import VIDEO_EXTENSIONS, IMAGE_EXTENSIONS, DEFAULT_NUM_FRAMES

logger = logging.getLogger(__name__)

def is_video_file(filename: str) -> bool:
    """Check if the file is a video based on extension"""
    return os.path.splitext(filename.lower())[1] in VIDEO_EXTENSIONS

def is_image_file(filename: str) -> bool:
    """Check if the file is an image based on extension"""
    return os.path.splitext(filename.lower())[1] in IMAGE_EXTENSIONS

async def process_video_frames(video_path: str, num_frames: int = DEFAULT_NUM_FRAMES) -> tuple:
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