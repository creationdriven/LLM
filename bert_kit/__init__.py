"""
BERT Kit - A comprehensive toolkit for BERT models

A Python package for building, training, fine-tuning, and evaluating BERT-style
language models with bidirectional attention.

Modules:
    - model: BERT model architecture components
    - data: Dataset classes and data loading utilities
    - training: Training utilities and loss computation
    - evaluation: Model evaluation functions
    - utils: Utility functions for text processing
    - config: Configuration constants and model configs
"""

from .model import (
    LayerNorm,
    GELU,
    FeedForward,
    MultiHeadAttention,
    TransformerEncoderBlock,
    BERTModel,
    BERTForMaskedLM,
    BERTForSequenceClassification
)

from .data import (
    MLMDataset,
    ClassificationDataset,
    create_mlm_dataloader,
    create_classification_dataloader
)

from .training import (
    compute_mlm_loss,
    compute_classification_loss,
    compute_mlm_loader_loss,
    compute_classification_loader_loss
)

from .evaluation import (
    evaluate_classification,
    evaluate_mlm_perplexity
)

from .utils import (
    simple_tokenize,
    get_tokenizer,
    create_attention_mask,
    create_token_type_ids,
    get_cls_embedding,
)

from llm_common.checkpoint import (
    save_model,
    load_model,
)

from .common import (
    get_device,
    count_parameters,
    print_model_summary
)

from .metrics import (
    compute_precision_recall_f1,
    compute_confusion_matrix,
    evaluate_classification_metrics
)

from .amp import (
    AMPTrainer,
    train_mlm_step_with_amp,
    train_classification_step_with_amp
)

from .config import (
    PAD_TOKEN_ID,
    CLS_TOKEN_ID,
    SEP_TOKEN_ID,
    MASK_TOKEN_ID,
    UNK_TOKEN_ID,
    IGNORE_INDEX,
    VOCAB_SIZE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_MASK_PROBABILITY,
    DEFAULT_OLLAMA_URL,
    DEFAULT_OLLAMA_MODEL,
    MODEL_CONFIGS,
    BASE_CONFIG,
    create_model_config
)

__version__ = "1.0.0"

__all__ = [
    # Model components
    "LayerNorm",
    "GELU",
    "FeedForward",
    "MultiHeadAttention",
    "TransformerEncoderBlock",
    "BERTModel",
    "BERTForMaskedLM",
    "BERTForSequenceClassification",
    # Data loading
    "MLMDataset",
    "ClassificationDataset",
    "create_mlm_dataloader",
    "create_classification_dataloader",
    # Training
    "compute_mlm_loss",
    "compute_classification_loss",
    "compute_mlm_loader_loss",
    "compute_classification_loader_loss",
    # Evaluation
    "evaluate_classification",
    "evaluate_mlm_perplexity",
    # Utils
    "simple_tokenize",
    "get_tokenizer",
    "create_attention_mask",
    "create_token_type_ids",
    "get_cls_embedding",
    "save_model",
    "load_model",
    # Common utilities
    "get_device",
    "count_parameters",
    "print_model_summary",
    # Metrics
    "compute_precision_recall_f1",
    "compute_confusion_matrix",
    "evaluate_classification_metrics",
    # Mixed Precision Training
    "AMPTrainer",
    "train_mlm_step_with_amp",
    "train_classification_step_with_amp",
    # Config
    "PAD_TOKEN_ID",
    "CLS_TOKEN_ID",
    "SEP_TOKEN_ID",
    "MASK_TOKEN_ID",
    "UNK_TOKEN_ID",
    "IGNORE_INDEX",
    "VOCAB_SIZE",
    "DEFAULT_MAX_LENGTH",
    "DEFAULT_DROPOUT_RATE",
    "DEFAULT_MASK_PROBABILITY",
    "DEFAULT_OLLAMA_URL",
    "DEFAULT_OLLAMA_MODEL",
    "MODEL_CONFIGS",
    "BASE_CONFIG",
    "create_model_config",
]

