"""
Evaluation module for BERT models.

Contains functions for evaluating BERT model performance on various tasks.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)


def evaluate_classification(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str,
    num_batches: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate BERT model on classification task.
    
    Args:
        model: BERTForSequenceClassification model instance
        data_loader: DataLoader with classification data
        device: Device to run evaluation on
        num_batches: Optional number of batches to evaluate
        
    Returns:
        Dictionary with 'accuracy', 'loss', and 'num_samples'
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    
    num_batches_to_eval = (
        min(num_batches, len(data_loader))
        if num_batches is not None
        else len(data_loader)
    )
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_batches_to_eval:
                break
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)
            
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
    
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / num_batches_to_eval if num_batches_to_eval > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "num_samples": total
    }


def evaluate_mlm_perplexity(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str,
    num_batches: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate BERT model on MLM task and compute perplexity.
    
    Args:
        model: BERTForMaskedLM model instance
        data_loader: DataLoader with MLM data
        device: Device to run evaluation on
        num_batches: Optional number of batches to evaluate
        
    Returns:
        Dictionary with 'perplexity', 'loss', and 'num_tokens'
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    num_batches_to_eval = (
        min(num_batches, len(data_loader))
        if num_batches is not None
        else len(data_loader)
    )
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_batches_to_eval:
                break
            
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids)
            
            # Compute loss only for masked tokens (labels != -100)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            
            # Count non-ignored tokens
            num_masked = (labels != -100).sum().item()
            total_tokens += num_masked
            total_loss += loss.item() * num_masked
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {
        "perplexity": perplexity,
        "loss": avg_loss,
        "num_tokens": total_tokens
    }

