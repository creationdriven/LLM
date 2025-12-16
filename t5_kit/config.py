"""
Configuration module for T5 models.

T5 (Text-To-Text Transfer Transformer) is an encoder-decoder model
that treats all NLP tasks as text-to-text problems.
"""

from typing import Dict, Any

# T5 uses sentencepiece tokenization
PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 1
UNK_TOKEN_ID = 2
IGNORE_INDEX = -100
VOCAB_SIZE = 32128  # T5 vocab size

DEFAULT_MAX_LENGTH = 512
DEFAULT_DROPOUT_RATE = 0.1

MODEL_CONFIGS: Dict[str, Dict[str, int]] = {
    "t5-small (60M)": {
        "d_model": 512,
        "d_ff": 2048,
        "num_layers": 6,
        "num_decoder_layers": 6,
        "num_heads": 8,
    },
    "t5-base (220M)": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_decoder_layers": 12,
        "num_heads": 12,
    },
    "t5-large (770M)": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_decoder_layers": 24,
        "num_heads": 16,
    },
}

BASE_CONFIG: Dict[str, Any] = {
    "vocab_size": VOCAB_SIZE,
    "max_position_embeddings": DEFAULT_MAX_LENGTH,
    "dropout_rate": DEFAULT_DROPOUT_RATE,
    "layer_norm_eps": 1e-6,
    "relative_attention_num_buckets": 32,
    "relative_attention_max_distance": 128,
}


def create_model_config(model_name: str) -> Dict[str, Any]:
    """Create a complete T5 model configuration."""
    if model_name not in MODEL_CONFIGS:
        raise KeyError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_CONFIGS.keys())}"
        )
    
    config = BASE_CONFIG.copy()
    config.update(MODEL_CONFIGS[model_name])
    return config

