"""
Distributed training utilities.

Provides utilities for multi-GPU and distributed training.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def setup_distributed(
    rank: int = 0,
    world_size: int = 1,
    backend: str = 'nccl'
) -> None:
    """
    Setup distributed training environment.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU)
    """
    if world_size <= 1:
        logger.warning("world_size <= 1, distributed training not needed")
        return
    
    try:
        torch.distributed.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size
        )
        logger.info(f"Distributed training initialized: rank={rank}, world_size={world_size}")
    except Exception as e:
        logger.error(f"Failed to initialize distributed training: {e}")
        raise


def wrap_model_for_distributed(
    model: nn.Module,
    device_ids: Optional[list] = None,
    find_unused_parameters: bool = False
) -> nn.Module:
    """
    Wrap model for distributed or multi-GPU training.
    
    Args:
        model: PyTorch model
        device_ids: List of device IDs for DataParallel
        find_unused_parameters: Whether to find unused parameters (for DDP)
        
    Returns:
        Wrapped model
        
    Example:
        >>> model = wrap_model_for_distributed(model, device_ids=[0, 1])
    """
    if torch.cuda.device_count() > 1:
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        
        if torch.distributed.is_initialized():
            # Use DistributedDataParallel
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[device_ids[0]] if device_ids else None,
                find_unused_parameters=find_unused_parameters
            )
            logger.info("Model wrapped with DistributedDataParallel")
        else:
            # Use DataParallel
            model = nn.DataParallel(model, device_ids=device_ids)
            logger.info(f"Model wrapped with DataParallel (devices: {device_ids})")
    else:
        logger.info("Single GPU/CPU detected, no parallelization needed")
    
    return model


def get_distributed_rank() -> int:
    """Get current process rank."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    return get_distributed_rank() == 0


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """
    Reduce tensor across all processes.
    
    Args:
        tensor: Tensor to reduce
        average: Whether to average (True) or sum (False)
        
    Returns:
        Reduced tensor
    """
    if not torch.distributed.is_initialized():
        return tensor
    
    world_size = get_world_size()
    if world_size == 1:
        return tensor
    
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    
    if average:
        rt /= world_size
    
    return rt

