"""
Automatic Mixed Precision (AMP) training utilities for T5 Kit.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Tuple
from torch.cuda.amp import autocast, GradScaler

from .training import compute_t5_loss

logger = logging.getLogger(__name__)


class AMPTrainer:
    """Automatic Mixed Precision trainer wrapper for T5 models."""
    
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
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        labels: torch.Tensor,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        max_grad_norm: Optional[float] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None
    ) -> float:
        """Perform a T5 training step with mixed precision."""
        model.train()
        optimizer.zero_grad()
        
        if self.enabled:
            with autocast(dtype=self.amp_dtype):
                loss = compute_t5_loss(
                    input_ids,
                    decoder_input_ids,
                    labels,
                    model,
                    device,
                    attention_mask=attention_mask,
                    decoder_attention_mask=decoder_attention_mask
                )
            
            self.scaler.scale(loss).backward()
            
            if max_grad_norm is not None:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss = compute_t5_loss(
                input_ids,
                decoder_input_ids,
                labels,
                model,
                device,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask
            )
            loss.backward()
            
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
        
        return loss.item()


def train_t5_step_with_amp(
    input_ids: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    labels: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    dtype: str = 'float16',
    max_grad_norm: Optional[float] = None,
    attention_mask: Optional[torch.Tensor] = None,
    decoder_attention_mask: Optional[torch.Tensor] = None,
    scaler: Optional[GradScaler] = None
) -> Tuple[float, Optional[GradScaler]]:
    """Perform a T5 training step with automatic mixed precision."""
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
            loss = compute_t5_loss(
                input_ids,
                decoder_input_ids,
                labels,
                model,
                device,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask
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
        
        loss = compute_t5_loss(
            input_ids,
            decoder_input_ids,
            labels,
            model,
            device,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask
        )
        loss.backward()
        
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        scaler = None
    
    return loss.item(), scaler

