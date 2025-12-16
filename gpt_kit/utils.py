"""
Utility functions for LLM evaluation.

Contains helper functions for text processing, tokenization, and text generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import os
import logging
from typing import Dict, Optional, Tuple, Any

from .config import DEFAULT_TEMPERATURE, EOS_TOKEN_ID

logger = logging.getLogger(__name__)


def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding) -> torch.Tensor:
    """
    Convert text to token IDs.
    
    Args:
        text: Input text string
        tokenizer: Tiktoken tokenizer instance
        
    Returns:
        Token IDs tensor of shape (1, seq_len)
    """
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: tiktoken.Encoding) -> str:
    """
    Convert token IDs to text.
    
    Args:
        token_ids: Token IDs tensor
        tokenizer: Tiktoken tokenizer instance
        
    Returns:
        Decoded text string
    """
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def format_alpaca_instruction(entry: Dict[str, str]) -> str:
    """
    Format instruction dataset entry into Alpaca format.
    
    Args:
        entry: Dictionary with 'instruction' and optionally 'input' keys
        
    Returns:
        Formatted instruction string
        
    Example:
        >>> entry = {"instruction": "Translate to French", "input": "Hello"}
        >>> format_alpaca_instruction(entry)
        "Below is an instruction...\\n\\n### Instruction:\\nTranslate to French\\n\\n### Input:\\nHello"
    """
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
    return instruction_text + input_text


def generate_text_autoregressive(
    model: nn.Module,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: Optional[int] = None,
    eos_token_id: int = EOS_TOKEN_ID
) -> torch.Tensor:
    """
    Generate text using autoregressive sampling.
    
    Args:
        model: GPT model instance
        token_ids: Initial token IDs of shape (1, seq_len)
        max_new_tokens: Maximum number of new tokens to generate
        context_size: Maximum context size to use
        temperature: Sampling temperature (higher = more random)
        top_k: If set, only sample from top-k tokens
        eos_token_id: End-of-sequence token ID
        
    Returns:
        Generated token IDs including input, shape (1, seq_len + num_generated)
    """
    model.eval()
    generated_token_ids = token_ids.clone()
    
    for _ in range(max_new_tokens):
        # Use only the last context_size tokens
        context_token_ids = generated_token_ids[:, -context_size:]
        
        with torch.no_grad():
            logits = model(context_token_ids)
            # Get logits for last token
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering if specified
            if top_k is not None:
                top_k_values, _ = torch.topk(
                    next_token_logits, 
                    min(top_k, next_token_logits.size(-1))
                )
                next_token_logits[next_token_logits < top_k_values[:, [-1]]] = -float('Inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            generated_token_ids = torch.cat((generated_token_ids, next_token_id), dim=1)
            
            # Stop if EOS token generated
            if next_token_id.item() == eos_token_id:
                break
    
    return generated_token_ids

# Note: save_model and load_model have been moved to llm_common.checkpoint
# Import them from llm_common or use: from llm_common import save_model, load_model

