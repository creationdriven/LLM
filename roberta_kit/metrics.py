"""
Evaluation metrics for BERT models.

Contains functions for computing classification metrics.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_precision_recall_f1(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: Optional[int] = None,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 score.
    
    Args:
        predictions: Predicted class labels of shape (batch_size,)
        labels: True class labels of shape (batch_size,)
        num_classes: Number of classes (None = auto-detect)
        average: Averaging strategy ('micro', 'macro', 'weighted', 'none')
        
    Returns:
        Dictionary with 'precision', 'recall', and 'f1' scores
        
    Example:
        >>> metrics = compute_precision_recall_f1(preds, labels, num_classes=2)
        >>> print(f"F1: {metrics['f1']:.4f}")
    """
    if num_classes is None:
        num_classes = max(predictions.max().item(), labels.max().item()) + 1
    
    # Convert to numpy for easier computation
    pred_np = predictions.cpu().numpy()
    label_np = labels.cpu().numpy()
    
    # Compute confusion matrix
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for i in range(len(predictions)):
        confusion_matrix[label_np[i], pred_np[i]] += 1
    
    # Compute metrics per class
    tp = torch.diag(confusion_matrix)
    fp = confusion_matrix.sum(dim=0) - tp
    fn = confusion_matrix.sum(dim=1) - tp
    
    # Avoid division by zero
    precision_per_class = tp.float() / (tp + fp + 1e-10)
    recall_per_class = tp.float() / (tp + fn + 1e-10)
    f1_per_class = 2 * (precision_per_class * recall_per_class) / (
        precision_per_class + recall_per_class + 1e-10
    )
    
    if average == 'micro':
        # Micro-averaged: overall precision/recall/F1
        total_tp = tp.sum().float()
        total_fp = fp.sum().float()
        total_fn = fn.sum().float()
        precision = total_tp / (total_tp + total_fp + 1e-10)
        recall = total_tp / (total_tp + total_fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    elif average == 'macro':
        # Macro-averaged: average of per-class metrics
        precision = precision_per_class.mean().item()
        recall = recall_per_class.mean().item()
        f1 = f1_per_class.mean().item()
    elif average == 'weighted':
        # Weighted average by support (number of true instances per class)
        support = confusion_matrix.sum(dim=1).float()
        weights = support / support.sum()
        precision = (precision_per_class * weights).sum().item()
        recall = (recall_per_class * weights).sum().item()
        f1 = (f1_per_class * weights).sum().item()
    else:  # 'none'
        return {
            'precision': precision_per_class.tolist(),
            'recall': recall_per_class.tolist(),
            'f1': f1_per_class.tolist()
        }
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: Optional[int] = None
) -> torch.Tensor:
    """
    Compute confusion matrix.
    
    Args:
        predictions: Predicted class labels of shape (batch_size,)
        labels: True class labels of shape (batch_size,)
        num_classes: Number of classes (None = auto-detect)
        
    Returns:
        Confusion matrix of shape (num_classes, num_classes)
        where matrix[i, j] = number of instances with true label i predicted as j
    """
    if num_classes is None:
        num_classes = max(predictions.max().item(), labels.max().item()) + 1
    
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    
    pred_np = predictions.cpu().numpy()
    label_np = labels.cpu().numpy()
    
    for i in range(len(predictions)):
        confusion_matrix[label_np[i], pred_np[i]] += 1
    
    return confusion_matrix


def evaluate_classification_metrics(
    model: torch.nn.Module,
    data_loader,
    device: str,
    num_classes: int,
    num_batches: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate classification model and compute comprehensive metrics.
    
    Args:
        model: BERTForSequenceClassification model instance
        data_loader: DataLoader with classification data
        device: Device to run evaluation on
        num_classes: Number of classes
        num_batches: Optional number of batches to evaluate
        
    Returns:
        Dictionary with 'accuracy', 'precision', 'recall', 'f1', 'loss'
    """
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0
    
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
            
            all_predictions.append(predictions)
            all_labels.append(labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
    
    # Concatenate all predictions and labels
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    # Compute metrics
    accuracy = (all_predictions == all_labels).float().mean().item()
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    
    metrics = compute_precision_recall_f1(
        all_predictions, all_labels, num_classes, average='weighted'
    )
    
    metrics.update({
        'accuracy': accuracy,
        'loss': avg_loss
    })
    
    return metrics

