"""
Common utilities shared across GPT Kit modules.

Contains device management, model utilities, and other shared functionality.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the best available device.
    
    Args:
        device: Optional device string ("cpu", "cuda", "mps")
        
    Returns:
        torch.device instance
        
    Example:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device("cuda")  # Force CUDA
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
        
    Example:
        >>> num_params = count_parameters(model)
        >>> print(f"Model has {num_params:,} parameters")
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
        
    Example:
        >>> print_model_summary(model, input_size=(1, 1024))
    """
    try:
        from torchsummary import summary
        summary(model, input_size)
    except ImportError:
        # Fallback to manual summary
        logger.warning("torchsummary not installed. Install with: pip install torchsummary")
        print(f"\n{'='*60}")
        print(f"Model: {model.__class__.__name__}")
        print(f"{'='*60}")
        print(f"Total parameters: {count_parameters(model):,}")
        print(f"Trainable parameters: {count_parameters(model, trainable_only=True):,}")
        print(f"{'='*60}\n")
        
        # Print layer information
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf node
                num_params = sum(p.numel() for p in module.parameters())
                if num_params > 0:
                    print(f"{name:40s} {str(type(module).__name__):30s} {num_params:>12,} params")


def compute_perplexity(
    model: nn.Module,
    data_loader,
    device: str,
    ignore_index: int = -100
) -> float:
    """
    Compute perplexity for a language model.
    
    Args:
        model: Language model instance
        data_loader: DataLoader with input and target batches
        device: Device to run computation on
        ignore_index: Index to ignore in loss calculation
        
    Returns:
        Perplexity value
        
    Example:
        >>> perplexity = compute_perplexity(model, val_loader, device="cuda")
        >>> print(f"Perplexity: {perplexity:.2f}")
    """
    import torch.nn.functional as F
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                input_ids, target_ids = batch[0].to(device), batch[1].to(device)
            elif isinstance(batch, dict):
                input_ids = batch.get("input_ids", batch.get("input")).to(device)
                target_ids = batch.get("labels", batch.get("target")).to(device)
            else:
                raise ValueError("Unsupported batch format")
            
            logits = model(input_ids)
            
            # Flatten for cross entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=ignore_index,
                reduction='sum'
            )
            
            # Count non-ignored tokens
            num_tokens = (target_ids != ignore_index).sum().item()
            total_loss += loss.item()
            total_tokens += num_tokens
    
    if total_tokens == 0:
        logger.warning("No valid tokens found for perplexity calculation")
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity

