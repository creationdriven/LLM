"""
Model checkpoint utilities with versioning support.
"""

import torch
import torch.nn as nn
import os
import logging
from typing import Dict, Any, Optional, Tuple, Type

logger = logging.getLogger(__name__)

# Checkpoint format version
CHECKPOINT_VERSION = "1.0"
SUPPORTED_VERSIONS = ["1.0", "0.9"]  # Support older versions for migration


def save_model(
    model: nn.Module,
    path: str,
    config: Dict[str, Any],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    loss: Optional[float] = None,
    version: str = CHECKPOINT_VERSION,
    **kwargs
) -> None:
    """
    Save a model checkpoint with versioning.
    
    Args:
        model: Model to save
        path: Path to save checkpoint
        config: Model configuration dictionary
        optimizer: Optional optimizer state
        epoch: Optional epoch number
        loss: Optional loss value
        version: Checkpoint format version (default: current version)
        **kwargs: Additional metadata to save
        
    Example:
        >>> save_model(model, "checkpoint.pt", config, optimizer=optimizer, epoch=10)
    """
    checkpoint = {
        'version': version,
        'model_state_dict': model.state_dict(),
        'config': config,
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if loss is not None:
        checkpoint['loss'] = loss
    checkpoint.update(kwargs)
    
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    torch.save(checkpoint, path)
    logger.info(f"Model saved to {path} (version {version})")


def load_model(
    path: str,
    model_class: Type[nn.Module],
    device: Optional[str] = None,
    return_optimizer: bool = False,
    optimizer_class: Optional[Type[torch.optim.Optimizer]] = None,
    auto_migrate: bool = True,
    **optimizer_kwargs
) -> Tuple[nn.Module, Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Load a model checkpoint with version support and migration.
    
    Args:
        path: Path to checkpoint file
        model_class: Model class to instantiate
        device: Optional device to load model on
        return_optimizer: Whether to return optimizer state
        optimizer_class: Optional optimizer class
        auto_migrate: Whether to automatically migrate old checkpoints
        **optimizer_kwargs: Optimizer initialization kwargs
        
    Returns:
        Tuple of (model, config, optimizer_state_or_dict)
        
    Example:
        >>> model, config, optimizer = load_model(
        ...     "checkpoint.pt", GPTModel, device="cuda",
        ...     return_optimizer=True, optimizer_class=torch.optim.AdamW
        ... )
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    
    # Check version and migrate if needed
    checkpoint_version = checkpoint.get('version', '0.9')  # Default to old version if missing
    if checkpoint_version not in SUPPORTED_VERSIONS:
        if auto_migrate:
            checkpoint = migrate_checkpoint(checkpoint, checkpoint_version, CHECKPOINT_VERSION)
            logger.info(f"Migrated checkpoint from version {checkpoint_version} to {CHECKPOINT_VERSION}")
        else:
            raise ValueError(
                f"Unsupported checkpoint version {checkpoint_version}. "
                f"Supported versions: {SUPPORTED_VERSIONS}"
            )
    
    config = checkpoint['config']
    
    model = model_class(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if device is not None:
        model = model.to(device)
    
    logger.info(f"Model loaded from {path} (version {checkpoint.get('version', 'unknown')})")
    
    optimizer_state = None
    if return_optimizer and 'optimizer_state_dict' in checkpoint:
        if optimizer_class is None:
            logger.warning("Optimizer class not provided, returning state dict only")
            optimizer_state = checkpoint['optimizer_state_dict']
        else:
            optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            optimizer_state = optimizer
    
    return model, config, optimizer_state


def migrate_checkpoint(
    checkpoint: Dict[str, Any],
    from_version: str,
    to_version: str
) -> Dict[str, Any]:
    """
    Migrate checkpoint from one version to another.
    
    Args:
        checkpoint: Checkpoint dictionary
        from_version: Source version
        to_version: Target version
        
    Returns:
        Migrated checkpoint dictionary
    """
    migrated = checkpoint.copy()
    
    # Migration from 0.9 to 1.0
    if from_version == "0.9" and to_version == "1.0":
        # Add version field if missing
        if 'version' not in migrated:
            migrated['version'] = to_version
        
        # Ensure all required fields exist
        if 'model_state_dict' not in migrated:
            raise ValueError("Checkpoint missing 'model_state_dict'")
        if 'config' not in migrated:
            raise ValueError("Checkpoint missing 'config'")
        
        logger.info("Migrated checkpoint from 0.9 to 1.0")
    
    # Add more migration logic here for future versions
    elif from_version == to_version:
        # No migration needed
        pass
    else:
        logger.warning(f"Unknown migration path: {from_version} -> {to_version}")
    
    return migrated


def get_checkpoint_version(path: str) -> Optional[str]:
    """
    Get checkpoint version without loading the full checkpoint.
    
    Args:
        path: Path to checkpoint file
        
    Returns:
        Checkpoint version string or None if not found
    """
    if not os.path.exists(path):
        return None
    
    try:
        checkpoint = torch.load(path, map_location='cpu')
        return checkpoint.get('version', '0.9')  # Default to old version
    except Exception as e:
        logger.warning(f"Could not read checkpoint version: {e}")
        return None


def validate_checkpoint(path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate checkpoint file.
    
    Args:
        path: Path to checkpoint file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not os.path.exists(path):
        return False, "Checkpoint file not found"
    
    try:
        checkpoint = torch.load(path, map_location='cpu')
        
        # Check required fields
        if 'model_state_dict' not in checkpoint:
            return False, "Missing 'model_state_dict'"
        if 'config' not in checkpoint:
            return False, "Missing 'config'"
        
        # Check version
        version = checkpoint.get('version', '0.9')
        if version not in SUPPORTED_VERSIONS:
            return False, f"Unsupported version: {version}"
        
        return True, None
    except Exception as e:
        return False, f"Error loading checkpoint: {e}"

