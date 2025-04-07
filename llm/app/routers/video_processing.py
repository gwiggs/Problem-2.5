from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
import os
import logging
import numpy as np
from app.utils.media_processor import (
    process_video_frames,
    chunk_video_frames,
    calculate_token_size,
    validate_video_stream
)
from app.core.config import (
    VIDEO_EXTENSIONS,
    TEMP_DIR,
    MAX_NEW_TOKENS
)
from app.models.video import VideoProcessingResponse, VideoChunkResponse
import asyncio
from concurrent.futures import ThreadPoolExecutor

router = APIRouter()
logger = logging.getLogger(__name__)

async def process_video_chunk(
    frames: List[np.ndarray],
    timestamps: List[float],
    chunk_index: int,
    total_chunks: int
) -> Dict[str, Any]:
    """
    Process a single chunk of video frames.
    Returns the processing results for this chunk.
    """
    try:
        # Process the chunk (implement your processing logic here)
        # This is a placeholder - replace with your actual processing logic
        results = {
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "num_frames": len(frames),
            "start_time": timestamps[0],
            "end_time": timestamps[-1],
            "processed_data": {}  # Add your processed data here
        }
        return results
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_index}: {str(e)}")
        raise

@router.post("/process", response_model=VideoProcessingResponse)
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    max_frames: Optional[int] = 16,
    target_size: Optional[tuple] = (224, 224)
):
    """
    Process a video file, automatically handling large files by chunking.
    Supports multiple video codecs including H.264, MPEG-4, and others.
    """
    try:
        # Validate file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported formats: {', '.join(VIDEO_EXTENSIONS)}"
            )
        
        # Save uploaded file
        temp_file_path = os.path.join(TEMP_DIR, file.filename)
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        try:
            # Validate video stream
            if not validate_video_stream(temp_file_path):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid or unsupported video format"
                )
            
            # Process video frames
            frames, timestamps, total_frames = process_video_frames(
                temp_file_path,
                max_frames=max_frames,
                target_size=target_size
            )
            
            # Calculate token size and determine if chunking is needed
            token_size = calculate_token_size(frames, target_size[1], target_size[0])
            
            if token_size > MAX_NEW_TOKENS:
                logger.info(f"Token size {token_size} exceeds max tokens {MAX_NEW_TOKENS}, processing in chunks")
                
                # Split frames into chunks
                chunks = chunk_video_frames(frames, timestamps, MAX_NEW_TOKENS, target_size)
                
                # Process chunks in parallel
                chunk_results = []
                with ThreadPoolExecutor() as executor:
                    futures = []
                    for i, (chunk_frames, chunk_timestamps) in enumerate(chunks):
                        future = executor.submit(
                            process_video_chunk,
                            chunk_frames,
                            chunk_timestamps,
                            i,
                            len(chunks)
                        )
                        futures.append(future)
                    
                    # Collect results as they complete
                    for future in futures:
                        try:
                            result = future.result()
                            chunk_results.append(result)
                        except Exception as e:
                            logger.error(f"Error processing chunk: {str(e)}")
                            raise HTTPException(
                                status_code=500,
                                detail=f"Error processing video chunk: {str(e)}"
                            )
                
                # Combine chunk results
                combined_results = {
                    "total_chunks": len(chunks),
                    "chunk_results": chunk_results,
                    "total_frames": total_frames,
                    "processed_frames": len(frames)
                }
                
                return VideoProcessingResponse(
                    success=True,
                    message="Video processed successfully in chunks",
                    data=combined_results
                )
            else:
                # Process entire video at once
                results = await process_video_chunk(frames, timestamps, 0, 1)
                
                return VideoProcessingResponse(
                    success=True,
                    message="Video processed successfully",
                    data=results
                )
                
        finally:
            # Clean up temporary file
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logger.warning(f"Error removing temporary file: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}"
        ) 