"""
T5 Kit - Text-To-Text Transfer Transformer.

A Python toolkit for building, training, and fine-tuning T5 models.
T5 treats all NLP tasks as text-to-text problems.
"""

__version__ = "1.0.0"

from .model import (
    T5Model,
    T5ForConditionalGeneration,
    T5Stack,
    T5Block,
    T5Attention,
    T5FeedForward,
    LayerNorm,
    RelativePositionBias,
)

from .config import (
    create_model_config,
    BASE_CONFIG,
    MODEL_CONFIGS,
    PAD_TOKEN_ID,
    EOS_TOKEN_ID,
    UNK_TOKEN_ID,
    IGNORE_INDEX,
    VOCAB_SIZE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_DROPOUT_RATE,
)

from .data import (
    T5Dataset,
    create_t5_dataloader,
)

from .training import (
    compute_t5_loss,
    compute_t5_loader_loss,
)

from .utils import (
    get_tokenizer,
    simple_tokenize,
)

from .amp import (
    AMPTrainer,
    train_t5_step_with_amp,
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

from .metrics import (
    compute_bleu_score,
    compute_rouge_score,
    evaluate_t5_generation,
    tokenize_text,
)

__all__ = [
    # Models
    "T5Model",
    "T5ForConditionalGeneration",
    "T5Stack",
    "T5Block",
    "T5Attention",
    "T5FeedForward",
    "LayerNorm",
    "RelativePositionBias",
    # Config
    "create_model_config",
    "BASE_CONFIG",
    "MODEL_CONFIGS",
    "PAD_TOKEN_ID",
    "EOS_TOKEN_ID",
    "UNK_TOKEN_ID",
    "IGNORE_INDEX",
    "VOCAB_SIZE",
    "DEFAULT_MAX_LENGTH",
    "DEFAULT_DROPOUT_RATE",
    # Data
    "T5Dataset",
    "create_t5_dataloader",
    # Training
    "compute_t5_loss",
    "compute_t5_loader_loss",
    # Utils
    "get_tokenizer",
    "simple_tokenize",
    # Mixed Precision Training
    "AMPTrainer",
    "train_t5_step_with_amp",
    # Common utilities
    "get_device",
    "count_parameters",
    "print_model_summary",
    "save_model",
    "load_model",
    # Metrics
    "compute_bleu_score",
    "compute_rouge_score",
    "evaluate_t5_generation",
    "tokenize_text",
]

