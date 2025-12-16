"""
GPT Kit - A comprehensive toolkit for GPT models

A Python package for building, training, fine-tuning, and evaluating GPT-style
language models.

Modules:
    - model: GPT model architecture components
    - data: Dataset classes and data loading utilities
    - training: Training utilities and loss computation
    - evaluation: Model response evaluation using Ollama
    - utils: Utility functions for text processing and generation
    - config: Configuration constants and model configs
"""

from .model import (
    LayerNorm,
    GELU,
    FeedForward,
    MultiHeadAttention,
    TransformerBlock,
    GPTModel
)

from .data import (
    GPTPretrainingDataset,
    InstructionDataset,
    create_pretraining_dataloader,
    create_instruction_collate_fn
)

from .evaluation import (
    check_ollama_running,
    query_ollama_model,
    evaluate_model_responses
)

from .utils import (
    text_to_token_ids,
    token_ids_to_text,
    format_alpaca_instruction,
    generate_text_autoregressive,
)

from llm_common.checkpoint import (
    save_model,
    load_model,
)

from .training import (
    compute_batch_loss,
    compute_loader_loss,
    train_step_with_gradient_clipping,
    train_step_with_accumulation
)

from .common import (
    get_device,
    count_parameters,
    print_model_summary,
    compute_perplexity
)

from .schedulers import (
    get_cosine_scheduler,
    get_linear_scheduler,
    get_constant_scheduler
)

from .callbacks import (
    EarlyStopping,
    TensorBoardLogger
)

from .amp import (
    AMPTrainer,
    train_step_with_amp
)

from .config import (
    PAD_TOKEN_ID,
    EOS_TOKEN_ID,
    IGNORE_INDEX,
    VOCAB_SIZE,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_TEMPERATURE,
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
    "TransformerBlock",
    "GPTModel",
    # Data loading
    "GPTPretrainingDataset",
    "InstructionDataset",
    "create_pretraining_dataloader",
    "create_instruction_collate_fn",
    # Training
    "compute_batch_loss",
    "compute_loader_loss",
    "train_step_with_gradient_clipping",
    "train_step_with_accumulation",
    # Evaluation
    "check_ollama_running",
    "query_ollama_model",
    "evaluate_model_responses",
    # Utils
    "text_to_token_ids",
    "token_ids_to_text",
    "format_alpaca_instruction",
    "generate_text_autoregressive",
    "save_model",
    "load_model",
    # Common utilities
    "get_device",
    "count_parameters",
    "print_model_summary",
    "compute_perplexity",
    # Schedulers
    "get_cosine_scheduler",
    "get_linear_scheduler",
    "get_constant_scheduler",
    # Callbacks
    "EarlyStopping",
    "TensorBoardLogger",
    # Mixed Precision Training
    "AMPTrainer",
    "train_step_with_amp",
    # Config
    "PAD_TOKEN_ID",
    "EOS_TOKEN_ID",
    "IGNORE_INDEX",
    "VOCAB_SIZE",
    "DEFAULT_CONTEXT_LENGTH",
    "DEFAULT_DROPOUT_RATE",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_OLLAMA_URL",
    "DEFAULT_OLLAMA_MODEL",
    "MODEL_CONFIGS",
    "BASE_CONFIG",
    "create_model_config",
]

