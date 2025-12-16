"""
Utility functions for RoBERTa models.
"""

from typing import List, Dict, Optional, Callable, Union
import torch
import logging

logger = logging.getLogger(__name__)

try:
    from transformers import RobertaTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning(
        "transformers library not found. Using simple tokenization. "
        "For production use, install with: pip install transformers"
    )


def get_tokenizer(
    model_name: str = "roberta-base",
    use_fast: bool = True
) -> Union[Callable[[str], List[int]], 'RobertaTokenizer']:
    """
    Get RoBERTa tokenizer, preferring transformers library.
    
    Args:
        model_name: RoBERTa model name (e.g., "roberta-base")
        use_fast: Whether to use fast tokenizer
        
    Returns:
        Tokenizer function or RobertaTokenizer instance
    """
    if HAS_TRANSFORMERS:
        try:
            tokenizer = RobertaTokenizer.from_pretrained(model_name, use_fast=use_fast)
            logger.info(f"Using transformers RobertaTokenizer: {model_name}")
            return tokenizer
        except Exception as e:
            logger.warning(f"Failed to load transformers tokenizer: {e}")
            return simple_tokenize
    else:
        logger.warning("Using simple tokenization. Install transformers for better results.")
        return simple_tokenize


def simple_tokenize(text: str) -> List[int]:
    """Simple tokenization function (word-level)."""
    words = text.lower().split()
    token_ids = [hash(word) % 50265 for word in words]  # RoBERTa vocab size
    return token_ids


def create_attention_mask(input_ids: torch.Tensor, pad_token_id: int = 1) -> torch.Tensor:
    """Create attention mask from input IDs."""
    return (input_ids != pad_token_id).long()


def get_cls_embedding(hidden_states: torch.Tensor) -> torch.Tensor:
    """Extract [CLS] token embedding (first token)."""
    return hidden_states[:, 0, :]

