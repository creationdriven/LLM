"""
Configuration module for BERT models.

Contains constants, default values, and model configurations for BERT architectures.
"""

from typing import Dict, Any

# ============================================================================
# TOKEN CONSTANTS (BERT uses different special tokens)
# ============================================================================

PAD_TOKEN_ID = 0
CLS_TOKEN_ID = 101  # [CLS] token
SEP_TOKEN_ID = 102  # [SEP] token
MASK_TOKEN_ID = 103  # [MASK] token
UNK_TOKEN_ID = 100  # [UNK] token
IGNORE_INDEX = -100
VOCAB_SIZE = 30522  # BERT base vocab size

# ============================================================================
# DEFAULT MODEL PARAMETERS
# ============================================================================

DEFAULT_MAX_LENGTH = 512  # BERT's typical max sequence length
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_MASK_PROBABILITY = 0.15  # For MLM training

# ============================================================================
# OLLAMA API DEFAULTS
# ============================================================================

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_OLLAMA_MODEL = "llama3"

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

# BERT model configurations
MODEL_CONFIGS: Dict[str, Dict[str, int]] = {
    "bert-base-uncased (110M)": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
    },
    "bert-large-uncased (340M)": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
    },
    "bert-base-cased (110M)": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
    },
    "bert-large-cased (340M)": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
    },
}

# Base configuration shared across all BERT models
BASE_CONFIG: Dict[str, Any] = {
    "vocab_size": VOCAB_SIZE,
    "max_position_embeddings": DEFAULT_MAX_LENGTH,
    "hidden_dropout_prob": DEFAULT_DROPOUT_RATE,
    "attention_probs_dropout_prob": DEFAULT_DROPOUT_RATE,
    "layer_norm_eps": 1e-12,
    "type_vocab_size": 2,  # For segment embeddings (sentence A/B)
}


def create_model_config(model_name: str) -> Dict[str, Any]:
    """
    Create a complete BERT model configuration from base config and model-specific config.
    
    Args:
        model_name: Name of the model (key in MODEL_CONFIGS)
        
    Returns:
        Complete configuration dictionary
        
    Raises:
        KeyError: If model_name is not in MODEL_CONFIGS
        
    Example:
        >>> config = create_model_config("bert-base-uncased (110M)")
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

