"""
Training callbacks for GPT Kit.

Contains callback classes for training monitoring and control.
"""

import logging
from typing import Optional, List
import os

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping callback to stop training when validation loss stops improving.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' for loss (lower is better) or 'max' for metrics (higher is better)
        restore_best_weights: Whether to restore best weights when stopping
        
    Example:
        >>> early_stopping = EarlyStopping(patience=5, min_delta=0.001)
        >>> for epoch in range(num_epochs):
        ...     val_loss = validate(model, val_loader)
        ...     if early_stopping(val_loss, model):
        ...         break
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.early_stop = False
        
        if mode not in ['min', 'max']:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
    
    def __call__(self, score: float, model) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score (loss or metric)
            model: Model instance (for saving weights)
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    logger.info("Restored best model weights")
                logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best score."""
        if self.mode == 'min':
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta
    
    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.early_stop = False


class TensorBoardLogger:
    """
    TensorBoard logging callback.
    
    Args:
        log_dir: Directory to save TensorBoard logs
        comment: Optional comment to append to log directory name
        
    Example:
        >>> logger = TensorBoardLogger("logs/experiment1")
        >>> logger.log_scalar("train/loss", loss_value, step)
        >>> logger.log_model_graph(model, input_sample)
    """
    
    def __init__(self, log_dir: str, comment: Optional[str] = None):
        self.log_dir = log_dir
        self.comment = comment
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir, comment=comment)
            self.enabled = True
        except ImportError:
            logger.warning(
                "TensorBoard not available. Install with: pip install tensorboard"
            )
            self.writer = None
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Log multiple scalar values."""
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_model_graph(self, model, input_sample):
        """Log model graph."""
        if self.enabled:
            try:
                self.writer.add_graph(model, input_sample)
            except Exception as e:
                logger.warning(f"Could not log model graph: {e}")
    
    def log_histogram(self, tag: str, values, step: int):
        """Log histogram of values."""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)
    
    def close(self):
        """Close the TensorBoard writer."""
        if self.enabled:
            self.writer.close()

