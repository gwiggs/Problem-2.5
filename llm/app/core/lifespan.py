from contextlib import asynccontextmanager
import logging
import sys
import signal
import atexit
import traceback

from app.core.model_manager import ModelManager
from app.core.cleanup import cleanup_resources

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global model manager
model_manager = ModelManager()

def signal_handler(sig, frame):
    """Handle termination signals"""
    logger.info(f"Received signal {sig}, initiating graceful shutdown...")
    cleanup_resources(model_manager)
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Register cleanup function to run on exit
atexit.register(lambda: cleanup_resources(model_manager))

@asynccontextmanager
async def lifespan(app):
    """Application lifespan manager for FastAPI"""
    # Startup
    logger.info("Initializing Qwen2.5-VL model...")
    try:
        # Initialize model manager
        await model_manager.initialize()
        logger.info("Model initialization completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    cleanup_resources(model_manager) 