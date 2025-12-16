"""
Automatic Mixed Precision (AMP) training utilities for BERT Kit.

Provides utilities for training with FP16/BF16 precision to reduce memory usage
and speed up training on compatible GPUs.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Tuple
from torch.cuda.amp import autocast, GradScaler

from .training import compute_mlm_loss, compute_classification_loss
from .config import IGNORE_INDEX

logger = logging.getLogger(__name__)


class AMPTrainer:
    """
    Automatic Mixed Precision trainer wrapper for BERT models.
    
    Handles gradient scaling and mixed precision training automatically.
    
    Args:
        dtype: Precision dtype - 'float16' (FP16) or 'bfloat16' (BF16)
        enabled: Whether AMP is enabled (default: True if CUDA available)
        init_scale: Initial gradient scaling factor
        growth_factor: Factor to increase scale on successful steps
        backoff_factor: Factor to decrease scale on overflow
        growth_interval: Steps between scale increases
        
    Example:
        >>> trainer = AMPTrainer(dtype='float16')
        >>> for batch in dataloader:
        ...     loss = trainer.train_mlm_step(
        ...         batch["input_ids"], batch["labels"], model, optimizer, device
        ...     )
    """
    
    def __init__(
        self,
        dtype: str = 'float16',
        enabled: Optional[bool] = None,
        init_scale: float = 2.0 ** 16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000
    ):
        if dtype not in ['float16', 'bfloat16']:
            raise ValueError(f"dtype must be 'float16' or 'bfloat16', got {dtype}")
        
        self.dtype = dtype
        self.enabled = enabled if enabled is not None else torch.cuda.is_available()
        
        if self.enabled and not torch.cuda.is_available():
            logger.warning("AMP requested but CUDA not available. Disabling AMP.")
            self.enabled = False
        
        if self.enabled:
            self.scaler = GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval
            )
            self.amp_dtype = torch.float16 if dtype == 'float16' else torch.bfloat16
        else:
            self.scaler = None
            self.amp_dtype = None
    
    def train_mlm_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        max_grad_norm: Optional[float] = None,
        ignore_index: int = IGNORE_INDEX
    ) -> float:
        """
        Perform an MLM training step with mixed precision.
        
        Args:
            input_ids: Input token IDs
            labels: Target labels
            model: BERTForMaskedLM model instance
            optimizer: Optimizer instance
            device: Device to run on
            max_grad_norm: Optional gradient clipping norm
            ignore_index: Index to ignore in loss calculation
            
        Returns:
            Loss value as float
        """
        model.train()
        optimizer.zero_grad()
        
        if self.enabled:
            with autocast(dtype=self.amp_dtype):
                loss = compute_mlm_loss(
                    input_ids, labels, model, device, ignore_index
                )
            
            self.scaler.scale(loss).backward()
            
            if max_grad_norm is not None:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss = compute_mlm_loss(
                input_ids, labels, model, device, ignore_index
            )
            loss.backward()
            
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
        
        return loss.item()
    
    def train_classification_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        max_grad_norm: Optional[float] = None
    ) -> float:
        """
        Perform a classification training step with mixed precision.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels
            model: BERTForSequenceClassification model instance
            optimizer: Optimizer instance
            device: Device to run on
            max_grad_norm: Optional gradient clipping norm
            
        Returns:
            Loss value as float
        """
        model.train()
        optimizer.zero_grad()
        
        if self.enabled:
            with autocast(dtype=self.amp_dtype):
                loss = compute_classification_loss(
                    input_ids, attention_mask, labels, model, device
                )
            
            self.scaler.scale(loss).backward()
            
            if max_grad_norm is not None:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss = compute_classification_loss(
                input_ids, attention_mask, labels, model, device
            )
            loss.backward()
            
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
        
        return loss.item()


def train_mlm_step_with_amp(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    dtype: str = 'float16',
    max_grad_norm: Optional[float] = None,
    ignore_index: int = IGNORE_INDEX,
    scaler: Optional[GradScaler] = None
) -> Tuple[float, Optional[GradScaler]]:
    """
    Perform an MLM training step with automatic mixed precision.
    
    Args:
        input_ids: Input token IDs
        labels: Target labels
        model: BERTForMaskedLM model instance
        optimizer: Optimizer instance
        device: Device to run computation on
        dtype: Precision dtype - 'float16' (FP16) or 'bfloat16' (BF16)
        max_grad_norm: Optional gradient clipping norm
        ignore_index: Index to ignore in loss calculation
        scaler: Optional GradScaler instance (creates new one if None)
        
    Returns:
        Tuple of (loss_value, scaler_instance)
    """
    if dtype not in ['float16', 'bfloat16']:
        raise ValueError(f"dtype must be 'float16' or 'bfloat16', got {dtype}")
    
    enabled = torch.cuda.is_available() and device.startswith('cuda')
    
    if enabled:
        if scaler is None:
            scaler = GradScaler()
        
        amp_dtype = torch.float16 if dtype == 'float16' else torch.bfloat16
        
        model.train()
        optimizer.zero_grad()
        
        with autocast(dtype=amp_dtype):
            loss = compute_mlm_loss(
                input_ids, labels, model, device, ignore_index
            )
        
        scaler.scale(loss).backward()
        
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        scaler.step(optimizer)
        scaler.update()
    else:
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using standard precision")
        
        model.train()
        optimizer.zero_grad()
        
        loss = compute_mlm_loss(
            input_ids, labels, model, device, ignore_index
        )
        loss.backward()
        
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        scaler = None
    
    return loss.item(), scaler


def train_classification_step_with_amp(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    dtype: str = 'float16',
    max_grad_norm: Optional[float] = None,
    scaler: Optional[GradScaler] = None
) -> Tuple[float, Optional[GradScaler]]:
    """
    Perform a classification training step with automatic mixed precision.
    
    Args:
        input_ids: Input token IDs
        attention_mask: Attention mask
        labels: Target labels
        model: BERTForSequenceClassification model instance
        optimizer: Optimizer instance
        device: Device to run computation on
        dtype: Precision dtype - 'float16' (FP16) or 'bfloat16' (BF16)
        max_grad_norm: Optional gradient clipping norm
        scaler: Optional GradScaler instance (creates new one if None)
        
    Returns:
        Tuple of (loss_value, scaler_instance)
    """
    if dtype not in ['float16', 'bfloat16']:
        raise ValueError(f"dtype must be 'float16' or 'bfloat16', got {dtype}")
    
    enabled = torch.cuda.is_available() and device.startswith('cuda')
    
    if enabled:
        if scaler is None:
            scaler = GradScaler()
        
        amp_dtype = torch.float16 if dtype == 'float16' else torch.bfloat16
        
        model.train()
        optimizer.zero_grad()
        
        with autocast(dtype=amp_dtype):
            loss = compute_classification_loss(
                input_ids, attention_mask, labels, model, device
            )
        
        scaler.scale(loss).backward()
        
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        scaler.step(optimizer)
        scaler.update()
    else:
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using standard precision")
        
        model.train()
        optimizer.zero_grad()
        
        loss = compute_classification_loss(
            input_ids, attention_mask, labels, model, device
        )
        loss.backward()
        
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        scaler = None
    
    return loss.item(), scaler

