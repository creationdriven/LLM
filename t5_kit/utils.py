"""
Utility functions for T5 models.
"""

from typing import List, Callable, Union
import logging

logger = logging.getLogger(__name__)

try:
    from transformers import T5Tokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning(
        "transformers library not found. Using simple tokenization. "
        "For production use, install with: pip install transformers"
    )


def get_tokenizer(
    model_name: str = "t5-small",
    use_fast: bool = True
) -> Union[Callable[[str], List[int]], 'T5Tokenizer']:
    """Get T5 tokenizer, preferring transformers library."""
    if HAS_TRANSFORMERS:
        try:
            tokenizer = T5Tokenizer.from_pretrained(model_name, use_fast=use_fast)
            logger.info(f"Using transformers T5Tokenizer: {model_name}")
            return tokenizer
        except Exception as e:
            logger.warning(f"Failed to load transformers tokenizer: {e}")
            return simple_tokenize
    else:
        logger.warning("Using simple tokenization. Install transformers for better results.")
        return simple_tokenize


def simple_tokenize(text: str) -> List[int]:
    """Simple tokenization function."""
    words = text.lower().split()
    token_ids = [hash(word) % 32128 for word in words]  # T5 vocab size
    return token_ids

