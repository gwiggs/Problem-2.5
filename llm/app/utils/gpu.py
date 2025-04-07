import logging
import subprocess
import os
import torch

logger = logging.getLogger(__name__)

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