"""
Configuration module for LLM evaluation.

Contains constants, default values, and model configurations.
"""

from typing import Dict, Any

# ============================================================================
# TOKEN CONSTANTS
# ============================================================================

PAD_TOKEN_ID = 50256
EOS_TOKEN_ID = 50256
IGNORE_INDEX = -100
VOCAB_SIZE = 50257

# ============================================================================
# DEFAULT MODEL PARAMETERS
# ============================================================================

DEFAULT_CONTEXT_LENGTH = 1024
DEFAULT_DROPOUT_RATE = 0.0
DEFAULT_TEMPERATURE = 1.0

# ============================================================================
# OLLAMA API DEFAULTS
# ============================================================================

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_OLLAMA_MODEL = "llama3"

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

# Model configurations (GPT-2 variants)
MODEL_CONFIGS: Dict[str, Dict[str, int]] = {
    "gpt2-small (124M)": {
        "emb_dim": 768,
        "n_layers": 12,
        "n_heads": 12
    },
    "gpt2-medium (355M)": {
        "emb_dim": 1024,
        "n_layers": 24,
        "n_heads": 16
    },
    "gpt2-large (774M)": {
        "emb_dim": 1280,
        "n_layers": 36,
        "n_heads": 20
    },
    "gpt2-xl (1558M)": {
        "emb_dim": 1600,
        "n_layers": 48,
        "n_heads": 25
    },
}

# Base configuration shared across all models
BASE_CONFIG: Dict[str, Any] = {
    "vocab_size": VOCAB_SIZE,
    "context_length": DEFAULT_CONTEXT_LENGTH,
    "drop_rate": DEFAULT_DROPOUT_RATE,
    "qkv_bias": True
}


def create_model_config(model_name: str) -> Dict[str, Any]:
    """
    Create a complete model configuration from base config and model-specific config.
    
    Args:
        model_name: Name of the model (key in MODEL_CONFIGS)
        
    Returns:
        Complete configuration dictionary
        
    Raises:
        KeyError: If model_name is not in MODEL_CONFIGS
        
    Example:
        >>> config = create_model_config("gpt2-small (124M)")
        >>> config["emb_dim"]
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

