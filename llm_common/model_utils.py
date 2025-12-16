"""
Model utility functions.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, only count trainable parameters
        
    Returns:
        Number of parameters
        
    Example:
        >>> num_params = count_parameters(model)
        >>> print(f"Model has {num_params:,} parameters")
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_model_summary(
    model: nn.Module,
    input_size: Optional[tuple] = None,
    device: Optional[str] = None
) -> None:
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Optional input size tuple for summary
        device: Optional device string
        
    Example:
        >>> print_model_summary(model, input_size=(1, 512))
    """
    print(f"\n{'='*60}")
    print(f"Model: {model.__class__.__name__}")
    print(f"{'='*60}")
    
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    if input_size is not None:
        try:
            # Try to use torchsummary if available
            try:
                from torchsummary import summary
                summary(model, input_size, device=device)
                return
            except ImportError:
                pass
            
            # Fallback: manual forward pass
            model.eval()
            with torch.no_grad():
                if device:
                    model = model.to(device)
                    dummy_input = torch.randn(input_size).to(device)
                else:
                    dummy_input = torch.randn(input_size)
                
                try:
                    output = model(dummy_input)
                    if isinstance(output, dict):
                        output = output.get("logits", list(output.values())[0])
                    print(f"\nInput shape: {input_size}")
                    print(f"Output shape: {output.shape}")
                except Exception as e:
                    logger.warning(f"Could not compute output shape: {e}")
        except Exception as e:
            logger.warning(f"Could not print detailed summary: {e}")
    
    print(f"\n{'='*60}\n")

