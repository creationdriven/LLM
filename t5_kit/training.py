"""
Training utilities for T5 fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def compute_t5_loss(
    input_ids: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    labels: torch.Tensor,
    model: nn.Module,
    device: str,
    attention_mask: Optional[torch.Tensor] = None,
    decoder_attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Calculate T5 loss."""
    input_ids = input_ids.to(device)
    decoder_input_ids = decoder_input_ids.to(device)
    labels = labels.to(device)
    
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    if decoder_attention_mask is not None:
        decoder_attention_mask = decoder_attention_mask.to(device)
    
    outputs = model(
        input_ids=input_ids,
        decoder_input_ids=decoder_input_ids,
        attention_mask=attention_mask,
        decoder_attention_mask=decoder_attention_mask,
        labels=labels
    )
    
    return outputs["loss"]


def compute_t5_loader_loss(
    data_loader: DataLoader,
    model: nn.Module,
    device: str,
    num_batches: Optional[int] = None
) -> float:
    """Calculate average T5 loss over a data loader."""
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
            
            loss = compute_t5_loss(
                batch["input_ids"],
                batch["decoder_input_ids"],
                batch["labels"],
                model,
                device,
                attention_mask=batch.get("attention_mask"),
                decoder_attention_mask=batch.get("decoder_attention_mask")
            )
            total_loss += loss.item()
    
    return total_loss / num_batches_to_eval

