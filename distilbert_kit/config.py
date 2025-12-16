"""
Configuration module for DistilBERT models.

DistilBERT is a smaller, faster version of BERT achieved through knowledge distillation.
- 40% smaller than BERT
- 60% faster inference
- 97% of BERT's performance
"""

from typing import Dict, Any

# Token constants (same as BERT)
PAD_TOKEN_ID = 0
CLS_TOKEN_ID = 101
SEP_TOKEN_ID = 102
MASK_TOKEN_ID = 103
UNK_TOKEN_ID = 100
IGNORE_INDEX = -100
VOCAB_SIZE = 30522

DEFAULT_MAX_LENGTH = 512
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_MASK_PROBABILITY = 0.15

MODEL_CONFIGS: Dict[str, Dict[str, int]] = {
    "distilbert-base-uncased (66M)": {
        "hidden_size": 768,
        "num_hidden_layers": 6,  # Half of BERT-base's 12 layers
        "num_attention_heads": 12,
        "intermediate_size": 3072,
    },
}

BASE_CONFIG: Dict[str, Any] = {
    "vocab_size": VOCAB_SIZE,
    "max_position_embeddings": DEFAULT_MAX_LENGTH,
    "hidden_dropout_prob": DEFAULT_DROPOUT_RATE,
    "attention_probs_dropout_prob": DEFAULT_DROPOUT_RATE,
    "layer_norm_eps": 1e-12,
    "type_vocab_size": 2,
}


def create_model_config(model_name: str) -> Dict[str, Any]:
    """Create a complete DistilBERT model configuration."""
    if model_name not in MODEL_CONFIGS:
        raise KeyError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_CONFIGS.keys())}"
        )
    
    config = BASE_CONFIG.copy()
    config.update(MODEL_CONFIGS[model_name])
    return config

