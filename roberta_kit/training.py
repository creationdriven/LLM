"""
Training utilities for RoBERTa fine-tuning.
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
    """Calculate Masked Language Modeling (MLM) loss."""
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    
    logits = model(input_ids)
    
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=ignore_index
    )
    return loss


def compute_classification_loss(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    model: nn.Module,
    device: str
) -> torch.Tensor:
    """Calculate classification loss."""
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
    """Calculate average MLM loss over a data loader."""
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
    """Calculate average classification loss over a data loader."""
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

