"""
Configuration module for RoBERTa models.

RoBERTa (Robustly Optimized BERT) is an improved version of BERT with:
- Dynamic masking (instead of static)
- No NSP task
- Larger batch sizes
- Better training procedure
"""

from typing import Dict, Any

# ============================================================================
# TOKEN CONSTANTS (RoBERTa uses same tokens as BERT)
# ============================================================================

PAD_TOKEN_ID = 1  # RoBERTa uses 1 for padding (different from BERT)
CLS_TOKEN_ID = 0  # RoBERTa uses 0 for [CLS]
SEP_TOKEN_ID = 2  # RoBERTa uses 2 for [SEP]
MASK_TOKEN_ID = 50264  # RoBERTa mask token
UNK_TOKEN_ID = 3  # RoBERTa unknown token
IGNORE_INDEX = -100
VOCAB_SIZE = 50265  # RoBERTa vocab size

# ============================================================================
# DEFAULT MODEL PARAMETERS
# ============================================================================

DEFAULT_MAX_LENGTH = 512
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_MASK_PROBABILITY = 0.15  # For MLM training

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

MODEL_CONFIGS: Dict[str, Dict[str, int]] = {
    "roberta-base (125M)": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
    },
    "roberta-large (355M)": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
    },
}

BASE_CONFIG: Dict[str, Any] = {
    "vocab_size": VOCAB_SIZE,
    "max_position_embeddings": DEFAULT_MAX_LENGTH,
    "hidden_dropout_prob": DEFAULT_DROPOUT_RATE,
    "attention_probs_dropout_prob": DEFAULT_DROPOUT_RATE,
    "layer_norm_eps": 1e-5,  # RoBERTa uses 1e-5 instead of 1e-12
    "type_vocab_size": 1,  # RoBERTa doesn't use token type IDs
}


def create_model_config(model_name: str) -> Dict[str, Any]:
    """
    Create a complete RoBERTa model configuration.
    
    Args:
        model_name: Name of the model (key in MODEL_CONFIGS)
        
    Returns:
        Complete configuration dictionary
        
    Example:
        >>> config = create_model_config("roberta-base (125M)")
        >>> config["hidden_size"]
        768
    """
    if model_name not in MODEL_CONFIGS:
        raise KeyError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_CONFIGS.keys())}"
        )
    
    config = BASE_CONFIG.copy()
    config.update(MODEL_CONFIGS[model_name])
    return config

