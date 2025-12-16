"""
Learning rate schedulers for GPT training.

Contains various learning rate scheduling strategies.
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional


def get_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1
) -> _LRScheduler:
    """
    Get cosine annealing learning rate scheduler with warmup.
    
    Args:
        optimizer: Optimizer instance
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as ratio of initial LR
        
    Returns:
        Learning rate scheduler
        
    Example:
        >>> scheduler = get_cosine_scheduler(optimizer, 1000, 10000)
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine annealing
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            min_lr_ratio,
            0.5 * (1.0 + math.cos(math.pi * progress))
        )
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_linear_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int
) -> _LRScheduler:
    """
    Get linear learning rate scheduler with warmup.
    
    Args:
        optimizer: Optimizer instance
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Linear decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 1.0 - progress)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_constant_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int
) -> _LRScheduler:
    """
    Get constant learning rate scheduler with warmup.
    
    Args:
        optimizer: Optimizer instance
        num_warmup_steps: Number of warmup steps
        
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

