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
    from transformers import BertTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning(
        "transformers library not found. Using simple tokenization. "
        "For production use, install with: pip install transformers"
    )


def get_tokenizer(
    model_name: str = "bert-base-uncased",
    use_fast: bool = True
) -> Union[Callable[[str], List[int]], 'BertTokenizer']:
    """
    Get BERT tokenizer, preferring transformers library.
    
    Args:
        model_name: BERT model name (e.g., "bert-base-uncased")
        use_fast: Whether to use fast tokenizer (if available)
        
    Returns:
        Tokenizer function or BertTokenizer instance
        
    Example:
        >>> tokenizer = get_tokenizer("bert-base-uncased")
        >>> token_ids = tokenizer("Hello world")
    """
    if HAS_TRANSFORMERS:
        try:
            tokenizer = BertTokenizer.from_pretrained(model_name, use_fast=use_fast)
            logger.info(f"Using transformers BertTokenizer: {model_name}")
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

# Note: save_model and load_model have been moved to llm_common.checkpoint
# Import them from llm_common or use: from llm_common import save_model, load_model

