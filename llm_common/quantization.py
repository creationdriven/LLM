"""
Model quantization utilities.

Provides utilities for quantizing models to reduce memory usage and speed up inference.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def quantize_model_dynamic(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8
) -> nn.Module:
    """
    Apply dynamic quantization to a model.
    
    Dynamic quantization quantizes weights and activations on-the-fly during inference.
    Good for LSTM, GRU, and Linear layers.
    
    Args:
        model: PyTorch model to quantize
        dtype: Quantization dtype (torch.qint8 or torch.float16)
        
    Returns:
        Quantized model
        
    Example:
        >>> quantized_model = quantize_model_dynamic(model)
    """
    try:
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=dtype
        )
        logger.info(f"Model quantized with dynamic quantization (dtype: {dtype})")
        return quantized_model
    except Exception as e:
        logger.warning(f"Dynamic quantization failed: {e}")
        return model


def quantize_model_static(
    model: nn.Module,
    calibration_data,
    dtype: torch.dtype = torch.qint8
) -> nn.Module:
    """
    Apply static quantization to a model.
    
    Static quantization requires calibration data and provides better performance
    but requires more setup.
    
    Args:
        model: PyTorch model to quantize
        calibration_data: DataLoader or list of inputs for calibration
        dtype: Quantization dtype
        
    Returns:
        Quantized model
        
    Example:
        >>> quantized_model = quantize_model_static(model, calibration_loader)
    """
    try:
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        
        # Calibrate
        with torch.no_grad():
            for data in calibration_data:
                if isinstance(data, (list, tuple)):
                    model(*data)
                elif isinstance(data, dict):
                    model(**data)
                else:
                    model(data)
        
        torch.quantization.convert(model, inplace=True)
        logger.info(f"Model quantized with static quantization (dtype: {dtype})")
        return model
    except Exception as e:
        logger.warning(f"Static quantization failed: {e}")
        return model


def get_model_size_mb(model: nn.Module) -> float:
    """
    Get model size in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


def compare_model_sizes(original_model: nn.Module, quantized_model: nn.Module) -> Dict[str, float]:
    """
    Compare sizes of original and quantized models.
    
    Args:
        original_model: Original model
        quantized_model: Quantized model
        
    Returns:
        Dictionary with size information
    """
    original_size = get_model_size_mb(original_model)
    quantized_size = get_model_size_mb(quantized_model)
    compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0
    
    return {
        'original_size_mb': original_size,
        'quantized_size_mb': quantized_size,
        'compression_ratio': compression_ratio,
        'size_reduction_percent': (1 - quantized_size / original_size) * 100 if original_size > 0 else 0
    }

