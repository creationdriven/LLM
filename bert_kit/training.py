"""
Training utilities for BERT fine-tuning.

Contains functions for loss computation for different BERT tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional
from torch.utils.data import DataLoader

from .config import IGNORE_INDEX

logger = logging.getLogger(__name__)


def compute_mlm_loss(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    model: nn.Module,
    device: str,
    ignore_index: int = IGNORE_INDEX
) -> torch.Tensor:
    """
    Calculate Masked Language Modeling (MLM) loss for BERT pretraining.
    
    MLM loss is computed only for masked token positions. Non-masked tokens
    have label -100 (ignore_index) and don't contribute to the loss.
    
    Key differences from GPT loss:
    - GPT: Predicts next token at every position
    - BERT: Only predicts masked tokens (15% of positions)
    - GPT: Uses causal (unidirectional) context
    - BERT: Uses bidirectional context (can see both sides)
    
    Example:
        Input:  [The, [MASK], sat, on, the, mat]
        Labels: [-100, cat, -100, -100, -100, -100]
        Loss:   Only computed for position 1 (the masked "cat")
    
    Args:
        input_ids: Input token IDs of shape (batch_size, seq_len)
                  Contains [MASK] tokens at masked positions
        labels: Label token IDs of shape (batch_size, seq_len)
                - Actual token ID at masked positions
                - ignore_index (-100) at non-masked positions
                This tells the loss function which positions to compute loss for
        model: BERTForMaskedLM model instance
              Returns logits of shape (batch_size, seq_len, vocab_size)
        device: Device to run computation on ("cuda" or "cpu")
        ignore_index: Index to ignore in loss calculation (default: -100)
                     Positions with this value don't contribute to loss
        
    Returns:
        Scalar loss tensor (average loss over all masked tokens)
    """
    # Move tensors to the specified device
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    
    # Forward pass: model predicts logits for each position
    # Shape: (batch_size, seq_len, vocab_size)
    # Even though we only care about masked positions, model computes all positions
    logits = model(input_ids)
    
    # Reshape for cross-entropy loss:
    # - Flatten batch and sequence dimensions: (batch*seq_len, vocab_size)
    # - Flatten labels: (batch*seq_len,)
    # Cross-entropy will automatically ignore positions with ignore_index
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),  # (batch*seq_len, vocab_size)
        labels.view(-1),  # (batch*seq_len,)
        ignore_index=ignore_index  # Ignore non-masked positions (-100)
    )
    
    return loss


def compute_classification_loss(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    model: nn.Module,
    device: str
) -> torch.Tensor:
    """
    Calculate classification loss.
    
    Args:
        input_ids: Input token IDs of shape (batch_size, seq_len)
        attention_mask: Attention mask of shape (batch_size, seq_len)
        labels: Label integers of shape (batch_size,)
        model: BERTForSequenceClassification model instance
        device: Device to run computation on
        
    Returns:
        Scalar loss tensor
    """
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    
    logits = model(input_ids, attention_mask)
    loss = F.cross_entropy(logits, labels)
    return loss


def compute_mlm_loader_loss(
    data_loader: DataLoader,
    model: nn.Module,
    device: str,
    num_batches: Optional[int] = None
) -> float:
    """
    Calculate average MLM loss over a data loader.
    
    Args:
        data_loader: DataLoader instance with MLM data
        model: BERTForMaskedLM model instance
        device: Device to run computation on
        num_batches: Optional number of batches to evaluate (None = all batches)
        
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
        for i, batch in enumerate(data_loader):
            if i >= num_batches_to_eval:
                break
            
            loss = compute_mlm_loss(
                batch["input_ids"],
                batch["labels"],
                model,
                device
            )
            total_loss += loss.item()
    
    return total_loss / num_batches_to_eval


def compute_classification_loader_loss(
    data_loader: DataLoader,
    model: nn.Module,
    device: str,
    num_batches: Optional[int] = None
) -> float:
    """
    Calculate average classification loss over a data loader.
    
    Args:
        data_loader: DataLoader instance with classification data
        model: BERTForSequenceClassification model instance
        device: Device to run computation on
        num_batches: Optional number of batches to evaluate (None = all batches)
        
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
        for i, batch in enumerate(data_loader):
            if i >= num_batches_to_eval:
                break
            
            loss = compute_classification_loss(
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
                model,
                device
            )
            total_loss += loss.item()
    
    return total_loss / num_batches_to_eval

