"""
Training utilities for LLM fine-tuning.

Contains functions for loss computation and training evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional
from torch.utils.data import DataLoader

from .config import IGNORE_INDEX

logger = logging.getLogger(__name__)


def compute_batch_loss(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    device: str,
    ignore_index: int = IGNORE_INDEX
) -> torch.Tensor:
    """
    Calculate cross-entropy loss for a single batch in GPT training.
    
    This function implements the standard next-token prediction loss used in
    GPT training. The model predicts the next token at each position, and we
    compute the cross-entropy loss between predictions and actual next tokens.
    
    Key points:
    - Input and target are shifted by 1 position (teacher forcing)
    - Loss is computed for all positions simultaneously
    - Padding tokens are ignored using ignore_index
    
    Example:
        Input:  [The, cat, sat]
        Target: [cat, sat, on]  (shifted by 1)
        Model predicts "cat" given "The", "sat" given "cat", etc.
    
    Args:
        input_batch: Input token IDs of shape (batch_size, seq_len)
                    These are the tokens the model sees
        target_batch: Target token IDs of shape (batch_size, seq_len)
                     These are the tokens the model should predict
                     (typically input_batch shifted by 1 position)
        model: GPT model instance
        device: Device to run computation on ("cuda" or "cpu")
        ignore_index: Index to ignore in loss calculation (for padding tokens)
                    Tokens with this index won't contribute to loss
        
    Returns:
        Scalar loss tensor (average loss over all non-ignored tokens)
    """
    # Move tensors to the specified device (GPU or CPU)
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    # Forward pass: model predicts logits for next token at each position
    # Shape: (batch_size, seq_len, vocab_size)
    # Each position has a probability distribution over the vocabulary
    logits = model(input_batch)
    
    # Reshape for cross-entropy loss:
    # - Flatten batch and sequence dimensions: (batch*seq_len, vocab_size)
    # - Flatten targets: (batch*seq_len,)
    # This allows computing loss for all positions at once
    loss = F.cross_entropy(
        logits.flatten(0, 1),  # (batch*seq_len, vocab_size)
        target_batch.flatten(),  # (batch*seq_len,)
        ignore_index=ignore_index  # Ignore padding tokens
    )
    
    return loss


def train_step_with_gradient_clipping(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    max_grad_norm: float = 1.0,
    ignore_index: int = IGNORE_INDEX,
    accumulation_steps: int = 1
) -> float:
    """
    Perform a single training step with gradient clipping and accumulation.
    
    Args:
        input_batch: Input token IDs of shape (batch_size, seq_len)
        target_batch: Target token IDs of shape (batch_size, seq_len)
        model: GPT model instance
        optimizer: Optimizer instance
        device: Device to run computation on
        max_grad_norm: Maximum gradient norm for clipping
        ignore_index: Index to ignore in loss calculation
        accumulation_steps: Number of steps to accumulate gradients before updating
        
    Returns:
        Loss value as float
    """
    model.train()
    
    loss = compute_batch_loss(input_batch, target_batch, model, device, ignore_index)
    # Scale loss by accumulation steps
    loss = loss / accumulation_steps
    loss.backward()
    
    # Only update weights after accumulation_steps
    if (getattr(train_step_with_gradient_clipping, 'step_count', 0) + 1) % accumulation_steps == 0:
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        train_step_with_gradient_clipping.step_count = 0
    else:
        train_step_with_gradient_clipping.step_count = getattr(
            train_step_with_gradient_clipping, 'step_count', 0
        ) + 1
    
    return loss.item() * accumulation_steps  # Return unscaled loss


def train_step_with_accumulation(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    accumulation_steps: int = 1,
    max_grad_norm: Optional[float] = None,
    ignore_index: int = IGNORE_INDEX,
    step_count: int = 0
) -> tuple[float, int]:
    """
    Perform a training step with gradient accumulation.
    
    Args:
        input_batch: Input token IDs
        target_batch: Target token IDs
        model: Model instance
        optimizer: Optimizer instance
        device: Device to run on
        accumulation_steps: Number of steps to accumulate
        max_grad_norm: Optional gradient clipping norm
        ignore_index: Index to ignore in loss
        step_count: Current step count (for accumulation tracking)
        
    Returns:
        Tuple of (loss_value, new_step_count)
    """
    model.train()
    
    loss = compute_batch_loss(input_batch, target_batch, model, device, ignore_index)
    loss = loss / accumulation_steps
    loss.backward()
    
    new_step_count = step_count + 1
    
    if new_step_count % accumulation_steps == 0:
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        new_step_count = 0
    
    return loss.item() * accumulation_steps, new_step_count


def compute_loader_loss(
    data_loader: DataLoader,
    model: nn.Module,
    device: str,
    num_batches: Optional[int] = None,
    ignore_index: int = IGNORE_INDEX
) -> float:
    """
    Calculate average loss over a data loader.
    
    Args:
        data_loader: DataLoader instance
        model: GPT model instance
        device: Device to run computation on
        num_batches: Optional number of batches to evaluate (None = all batches)
        ignore_index: Index to ignore in loss calculation
        
    Returns:
        Average loss as float (NaN if data_loader is empty)
    """
    if len(data_loader) == 0:
        logger.warning("DataLoader is empty")
        return float("nan")
    
    total_loss = 0.0
    num_batches_to_eval = (
        min(num_batches, len(data_loader)) 
        if num_batches is not None 
        else len(data_loader)
    )
    
    model.eval()
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches_to_eval:
                break
            loss = compute_batch_loss(
                input_batch, target_batch, model, device, ignore_index
            )
            total_loss += loss.item()
    
    return total_loss / num_batches_to_eval

