"""
RoBERTa Kit - Robustly Optimized BERT Pretraining Approach.

A Python toolkit for building, training, and fine-tuning RoBERTa models.
RoBERTa improves upon BERT with:
- Dynamic masking (instead of static)
- No NSP task
- Larger batch sizes
- Better training procedure
"""

__version__ = "1.0.0"

from .model import (
    RoBERTaModel,
    RoBERTaForMaskedLM,
    RoBERTaForSequenceClassification,
    LayerNorm,
    GELU,
    FeedForward,
    MultiHeadAttention,
    TransformerEncoderBlock,
)

from .config import (
    create_model_config,
    BASE_CONFIG,
    MODEL_CONFIGS,
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
)

from .data import (
    MLMDataset,
    ClassificationDataset,
    create_mlm_dataloader,
    create_classification_dataloader,
)

from .training import (
    compute_mlm_loss,
    compute_classification_loss,
    compute_mlm_loader_loss,
    compute_classification_loader_loss,
)

from .utils import (
    get_tokenizer,
    simple_tokenize,
    create_attention_mask,
    get_cls_embedding,
)

from .amp import (
    AMPTrainer,
    train_mlm_step_with_amp,
    train_classification_step_with_amp,
)

from .metrics import (
    compute_precision_recall_f1,
    compute_confusion_matrix,
    evaluate_classification_metrics,
)

from .common import (
    get_device,
    count_parameters,
    print_model_summary,
)

from llm_common.checkpoint import (
    save_model,
    load_model,
)

__all__ = [
    # Models
    "RoBERTaModel",
    "RoBERTaForMaskedLM",
    "RoBERTaForSequenceClassification",
    # Components
    "LayerNorm",
    "GELU",
    "FeedForward",
    "MultiHeadAttention",
    "TransformerEncoderBlock",
    # Config
    "create_model_config",
    "BASE_CONFIG",
    "MODEL_CONFIGS",
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
    # Data
    "MLMDataset",
    "ClassificationDataset",
    "create_mlm_dataloader",
    "create_classification_dataloader",
    # Training
    "compute_mlm_loss",
    "compute_classification_loss",
    "compute_mlm_loader_loss",
    "compute_classification_loader_loss",
    # Utils
    "get_tokenizer",
    "simple_tokenize",
    "create_attention_mask",
    "get_cls_embedding",
    # Mixed Precision Training
    "AMPTrainer",
    "train_mlm_step_with_amp",
    "train_classification_step_with_amp",
    # Metrics
    "compute_precision_recall_f1",
    "compute_confusion_matrix",
    "evaluate_classification_metrics",
    # Common utilities
    "get_device",
    "count_parameters",
    "print_model_summary",
    "save_model",
    "load_model",
]

