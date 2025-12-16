"""
Common utilities shared across BERT Kit modules.

Contains device management, model utilities, and other shared functionality.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the best available device.
    
    Args:
        device: Optional device string ("cpu", "cuda", "mps")
        
    Returns:
        torch.device instance
    """
    if device is not None:
        return torch.device(device)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, only count trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_model_summary(model: nn.Module, input_size: tuple = (1, 512)) -> None:
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input size tuple (batch_size, seq_len)
    """
    try:
        from torchsummary import summary
        summary(model, input_size)
    except ImportError:
        logger.warning("torchsummary not installed. Install with: pip install torchsummary")
        print(f"\n{'='*60}")
        print(f"Model: {model.__class__.__name__}")
        print(f"{'='*60}")
        print(f"Total parameters: {count_parameters(model):,}")
        print(f"Trainable parameters: {count_parameters(model, trainable_only=True):,}")
        print(f"{'='*60}\n")
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                num_params = sum(p.numel() for p in module.parameters())
                if num_params > 0:
                    print(f"{name:40s} {str(type(module).__name__):30s} {num_params:>12,} params")

