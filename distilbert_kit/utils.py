"""
Utility functions for BERT models.

Contains helper functions for tokenization and text processing.
Integrates with transformers library for proper BERT tokenization.
"""

from typing import List, Dict, Optional, Callable, Union, Any, Tuple
import torch
import logging
import os

logger = logging.getLogger(__name__)

# Try to import transformers
try:
    from transformers import DistilBertTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning(
        "transformers library not found. Using simple tokenization. "
        "For production use, install with: pip install transformers"
    )


def get_tokenizer(
    model_name: str = "distilbert-base-uncased",
    use_fast: bool = True
) -> Union[Callable[[str], List[int]], 'DistilBertTokenizer']:
    """
    Get DistilBERT tokenizer, preferring transformers library.
    
    Args:
        model_name: DistilBERT model name (e.g., "distilbert-base-uncased")
        use_fast: Whether to use fast tokenizer (if available)
        
    Returns:
        Tokenizer function or DistilBertTokenizer instance
        
    Example:
        >>> tokenizer = get_tokenizer("distilbert-base-uncased")
        >>> token_ids = tokenizer("Hello world")
    """
    if HAS_TRANSFORMERS:
        try:
            tokenizer = DistilBertTokenizer.from_pretrained(model_name, use_fast=use_fast)
            logger.info(f"Using transformers DistilBertTokenizer: {model_name}")
            return tokenizer
        except Exception as e:
            logger.warning(f"Failed to load transformers tokenizer: {e}")
            logger.warning("Falling back to simple tokenization")
            return simple_tokenize
    else:
        logger.warning("Using simple tokenization. Install transformers for better results.")
        return simple_tokenize


def simple_tokenize(text: str, vocab: Optional[Dict[str, int]] = None) -> List[int]:
    """
    Simple tokenization function (word-level).
    
    Note: This is a basic implementation. For production use with BERT,
    you should use WordPiece tokenization from the transformers library.
    
    Args:
        text: Input text string
        vocab: Optional vocabulary dictionary mapping tokens to IDs
        
    Returns:
        List of token IDs
    """
    # Simple word-level tokenization
    # In practice, BERT uses WordPiece tokenization
    words = text.lower().split()
    
    if vocab is None:
        # Simple hash-based tokenization (not ideal, but works for demo)
        token_ids = [hash(word) % 30522 for word in words]
    else:
        token_ids = [vocab.get(word, 100) for word in words]  # 100 = [UNK]
    
    return token_ids


def create_attention_mask(input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    Create attention mask from input IDs.
    
    Args:
        input_ids: Token IDs tensor of shape (batch_size, seq_len)
        pad_token_id: Padding token ID
        
    Returns:
        Attention mask tensor of shape (batch_size, seq_len)
        where 1 = attend, 0 = mask out
    """
    return (input_ids != pad_token_id).long()


def create_token_type_ids(input_ids: torch.Tensor, sep_token_id: int = 102) -> torch.Tensor:
    """
    Create token type IDs (segment IDs) for sentence pairs.
    
    Args:
        input_ids: Token IDs tensor
        sep_token_id: [SEP] token ID
        
    Returns:
        Token type IDs tensor (0 for first sentence, 1 for second)
    """
    batch_size, seq_len = input_ids.shape
    token_type_ids = torch.zeros_like(input_ids)
    
    for i in range(batch_size):
        # Find [SEP] token positions
        sep_positions = (input_ids[i] == sep_token_id).nonzero(as_tuple=True)[0]
        if len(sep_positions) > 0:
            # Set tokens after first [SEP] to 1
            first_sep = sep_positions[0].item()
            token_type_ids[i, first_sep + 1:] = 1
    
    return token_type_ids


def get_cls_embedding(hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Extract [CLS] token embedding from BERT hidden states.
    
    Args:
        hidden_states: Hidden states from BERT of shape (batch_size, seq_len, hidden_size)
        
    Returns:
        [CLS] embeddings of shape (batch_size, hidden_size)
    """
    return hidden_states[:, 0, :]  # [CLS] is always at position 0


def save_model(
    model: torch.nn.Module,
    path: str,
    config: Dict[str, Any],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    loss: Optional[float] = None,
    **kwargs
) -> None:
    """
    Save BERT model, configuration, and optional training state.
    
    Args:
        model: Model instance to save
        path: Path to save the checkpoint
        config: Model configuration dictionary
        optimizer: Optional optimizer state
        epoch: Optional epoch number
        loss: Optional loss value
        **kwargs: Additional metadata to save
        
    Example:
        >>> save_model(model, "checkpoint.pt", config, optimizer=optimizer, epoch=10)
    """
    checkpoint = {
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
    logger.info(f"Model saved to {path}")


def load_model(
    path: str,
    model_class: type,
    device: Optional[str] = None,
    return_optimizer: bool = False,
    optimizer_class: Optional[type] = None,
    **optimizer_kwargs
) -> tuple:
    """
    Load BERT model, configuration, and optional training state.
    
    Args:
        path: Path to the checkpoint file
        model_class: Model class to instantiate (e.g., BERTModel)
        device: Device to load model to
        return_optimizer: Whether to return optimizer state dict
        optimizer_class: Optimizer class if returning optimizer state
        **optimizer_kwargs: Arguments for optimizer initialization
        
    Returns:
        Tuple of (model, config, optional_optimizer_state_dict)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint['config']
    
    model = model_class(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if device is not None:
        model = model.to(device)
    
    logger.info(f"Model loaded from {path}")
    
    optimizer_state = None
    if return_optimizer and 'optimizer_state_dict' in checkpoint:
        if optimizer_class is None:
            logger.warning("Optimizer class not provided, returning state dict only")
        else:
            optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            optimizer_state = optimizer
        if optimizer_state is None:
            optimizer_state = checkpoint['optimizer_state_dict']
    
    return model, config, optimizer_state

