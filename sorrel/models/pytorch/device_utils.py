"""Utility for automatic device detection in PyTorch models.

Automatically selects the best available device:
1. MPS (Apple Silicon GPU) if available
2. CUDA (NVIDIA GPU) if available  
3. CPU as fallback
"""

import torch


def get_optimal_device() -> str:
    """Automatically detect and return the best available PyTorch device.
    
    Priority order:
    1. MPS (Metal Performance Shaders) - Apple Silicon GPU
    2. CUDA - NVIDIA GPU
    3. CPU - Fallback
    
    Returns:
        str: Device string ("mps", "cuda", or "cpu")
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def resolve_device(device: str | torch.device | None = None) -> str:
    """Resolve device specification with automatic detection fallback.
    
    Args:
        device: User-specified device ("cpu", "cuda", "mps", torch.device, or None)
               If None or "auto", will auto-detect optimal device.
    
    Returns:
        str: Resolved device string
    """
    if device is None or device == "auto":
        return get_optimal_device()
    
    if isinstance(device, torch.device):
        return str(device)
    
    return device
