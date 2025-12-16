"""
Shared utilities for LLM kits.

This package contains common utilities that can be shared across all LLM kits
to reduce code duplication.
"""

from .device import get_device
from .model_utils import count_parameters, print_model_summary
from .checkpoint import (
    save_model,
    load_model,
    CHECKPOINT_VERSION,
    SUPPORTED_VERSIONS,
    migrate_checkpoint,
    get_checkpoint_version,
    validate_checkpoint,
)
from .quantization import (
    quantize_model_dynamic,
    quantize_model_static,
    get_model_size_mb,
    compare_model_sizes,
)
from .distributed import (
    setup_distributed,
    wrap_model_for_distributed,
    get_distributed_rank,
    get_world_size,
    is_main_process,
    reduce_tensor,
)

__all__ = [
    # Device
    "get_device",
    # Model utilities
    "count_parameters",
    "print_model_summary",
    # Checkpointing
    "save_model",
    "load_model",
    "CHECKPOINT_VERSION",
    "SUPPORTED_VERSIONS",
    "migrate_checkpoint",
    "get_checkpoint_version",
    "validate_checkpoint",
    # Quantization
    "quantize_model_dynamic",
    "quantize_model_static",
    "get_model_size_mb",
    "compare_model_sizes",
    # Distributed training
    "setup_distributed",
    "wrap_model_for_distributed",
    "get_distributed_rank",
    "get_world_size",
    "is_main_process",
    "reduce_tensor",
]

