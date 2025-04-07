import logging
import os
import torch
import traceback
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer, BitsAndBytesConfig

from app.core.config import MODEL_PATH
from app.utils.gpu import check_gpu_availability, log_gpu_info
from app.utils.model_loader import (
    load_model_primary,
    load_model_alternative,
    load_model_minimal,
    load_processor,
    load_tokenizer
)

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages the LLM model, processor, and tokenizer"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = None
        self.is_shutting_down = False
        
        # Create the quantization config
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",  # or "fp4"
            bnb_4bit_compute_dtype="float16"  # or "float16" if needed
        )
    
    async def initialize(self):
        """Initialize the model, processor, and tokenizer"""
        # Check GPU availability
        logger.info("Step 1: Checking GPU availability")
        gpu_available = check_gpu_availability()
        
        if gpu_available:
            # Force CUDA device selection
            logger.info("Step 2: Setting up CUDA device")
            self.device = torch.device("cuda:0")
            log_gpu_info()
            
            # Set CUDA device explicitly
            torch.cuda.set_device(0)
        else:
            self.device = torch.device("cpu")
            logger.info("CUDA not available, using CPU")
        
        # Log PyTorch version
        logger.info(f"PyTorch version: {torch.__version__}")
        
        # Check if flash-attn is installed
        use_flash_attention = self._check_flash_attention()
        
        # Initialize model with explicit device placement
        logger.info("Step 3: Loading model from pretrained")
        logger.info(f"Model path: {os.path.abspath(MODEL_PATH)}")
        logger.info(f"Model path exists: {os.path.exists(MODEL_PATH)}")
        
        if os.path.exists(MODEL_PATH):
            logger.info(f"Model path contents: {os.listdir(MODEL_PATH)}")
        else:
            logger.error(f"Model path does not exist: {MODEL_PATH}")
            logger.info("Current directory contents:")
            logger.info(os.listdir("/app"))
            logger.info("Attempting to create model directory...")
            os.makedirs(MODEL_PATH, exist_ok=True)
            logger.info(f"Model directory created: {os.path.exists(MODEL_PATH)}")
        
        try:
            # Try primary loading method
            self.model = load_model_primary(MODEL_PATH, gpu_available, use_flash_attention, self.bnb_config)
            logger.info("Model loaded successfully with primary method")
        except Exception as model_error:
            logger.error(f"Failed to load model with primary method: {str(model_error)}")
            logger.error(traceback.format_exc())
            
            try:
                # Try alternative loading method
                self.model = load_model_alternative(MODEL_PATH, gpu_available, use_flash_attention, self.bnb_config, self.device)
                logger.info("Model loaded successfully with alternative method")
            except Exception as alt_error:
                logger.error(f"Failed to load model with alternative method: {str(alt_error)}")
                logger.error(traceback.format_exc())
                
                try:
                    # Try minimal loading method
                    self.model = load_model_minimal(MODEL_PATH, gpu_available, self.bnb_config, self.device)
                    logger.info("Model loaded successfully with minimal settings")
                except Exception as final_error:
                    logger.error(f"Failed to load model with minimal settings: {str(final_error)}")
                    logger.error(traceback.format_exc())
                    raise
        
        # Move model to device explicitly if needed
        if gpu_available and not hasattr(self.model, "device"):
            logger.info("Moving model to device explicitly")
            self.model = self.model.to(self.device)
            
        # Load tokenizer
        self.tokenizer = load_tokenizer(MODEL_PATH)
        logger.info("Tokenizer loaded successfully")
        
        # Load processor
        self.processor = load_processor(MODEL_PATH)
        logger.info("Processor loaded successfully")
        
        # Log model device
        if hasattr(self.model, "device"):
            logger.info(f"Model device: {self.model.device}")
        else:
            logger.info("Model device information not available")
    
    def _check_flash_attention(self):
        """Check if flash attention is available"""
        try:
            import flash_attn
            logger.info(f"Flash Attention 2 is available: {flash_attn.__version__}")
            return True
        except ImportError:
            logger.warning("Flash Attention 2 is not available, will use standard attention")
            return False
    
    def release_resources(self):
        """Release model resources"""
        if self.model is not None:
            logger.info("Releasing model...")
            self.model = None
            
        if self.processor is not None:
            logger.info("Releasing processor...")
            self.processor = None
            
        if self.tokenizer is not None:
            logger.info("Releasing tokenizer...")
            self.tokenizer = None
    
    def get_model(self):
        """Get the model"""
        return self.model
    
    def get_processor(self):
        """Get the processor"""
        return self.processor
    
    def get_tokenizer(self):
        """Get the tokenizer"""
        return self.tokenizer
    
    def get_device(self):
        """Get the device"""
        return self.device 