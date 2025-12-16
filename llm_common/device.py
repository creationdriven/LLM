"""
Device management utilities.
"""

import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_device(device: Optional[str] = None) -> str:
    """
    Get the best available device.
    
    Args:
        device: Optional device string ('cuda', 'cpu', 'mps', etc.)
                If None, auto-detects best available device
        
    Returns:
        Device string
        
    Example:
        >>> device = get_device()
        >>> device = get_device("cuda")
    """
    if device is not None:
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, using CPU")
            return "cpu"
        if device == "mps" and not hasattr(torch.backends, "mps"):
            logger.warning("MPS requested but not available, using CPU")
            return "cpu"
        return device
    
    # Auto-detect best device
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

