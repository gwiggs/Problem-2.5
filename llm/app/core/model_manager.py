import logging
import os
import torch
import traceback
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM

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
        self.load_model()
        
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
    
    def load_model(self) -> None:
        """Load the model with optimized memory settings."""
        try:
            logger.info("Step 3: Loading model from pretrained")
            logger.info(f"Model path: {MODEL_PATH}")
            logger.info(f"Model path exists: {os.path.exists(MODEL_PATH)}")
            logger.info(f"Model path contents: {os.listdir(MODEL_PATH)}")
            
            # Load model with memory optimizations
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                device_map="auto",
                torch_dtype=torch.float16,  # Use float16 for reduced memory usage
                low_cpu_mem_usage=True,
                use_cache=True,  # Enable KV cache for faster inference
                quantization_config=self.bnb_config,  # Use the quantization config
                max_memory={0: "12GB"},  # Limit GPU memory usage
                offload_folder="offload",  # Enable disk offloading if needed
                offload_state_dict=True  # Enable state dict offloading
            )
            
            # Enable gradient checkpointing
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            
            # Move model to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Enable CUDA graph optimization if available
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            logger.info("Model loaded successfully with primary method")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
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