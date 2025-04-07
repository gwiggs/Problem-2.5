import logging
import torch
import traceback

logger = logging.getLogger(__name__)

def cleanup_resources(model_manager):
    """Clean up resources before shutdown"""
    if model_manager.is_shutting_down:
        return
        
    model_manager.is_shutting_down = True
    logger.info("Cleaning up resources...")
    
    try:
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            logger.info("Clearing CUDA cache...")
            torch.cuda.empty_cache()
            logger.info(f"CUDA memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # Release model resources
        model_manager.release_resources()
            
        logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        logger.error(traceback.format_exc()) 