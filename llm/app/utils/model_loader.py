import logging
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    # bnb_4bit_block_size=128
    bnb_4bit_compute_dtype=torch.float16
)

def load_model_primary(model_path, gpu_available, use_flash_attention, bnb_config):
    """Primary model loading method"""
    logger.info("Attempting to load model with primary method")
    model_kwargs = {
        "device_map": "auto" if gpu_available else "cpu",
        # "low_cpu_mem_usage": True,
        "quantization_config": bnb_config,
        "max_memory": {0: "8GiB"},
        "trust_remote_code": True
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

def load_model_alternative(model_path, gpu_available, use_flash_attention, bnb_config, device):
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

def load_model_minimal(model_path, gpu_available, bnb_config, device):
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