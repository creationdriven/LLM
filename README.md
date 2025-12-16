# LLM Kits

A comprehensive Python toolkit for building, training, and fine-tuning Large Language Models (LLMs). This package provides implementations of GPT, BERT, RoBERTa, DistilBERT, and T5 models with advanced training utilities, evaluation metrics, and production-ready features.

## 🚀 Features

- **5 Complete LLM Kits**: GPT, BERT, RoBERTa, DistilBERT, and T5
- **Mixed Precision Training**: FP16/BF16 support for faster training and reduced memory
- **Advanced Training Utilities**: Gradient clipping, accumulation, learning rate schedulers, early stopping
- **Comprehensive Evaluation**: Classification metrics, BLEU/ROUGE scores, perplexity
- **Model Management**: Save/load checkpoints, model quantization, distributed training
- **Production Ready**: Comprehensive tests, benchmarks, and examples

## 📦 Installation

### Basic Installation

```bash
pip install -r requirements.txt
```

### Core Dependencies

- `torch>=2.0.0,<3.0.0` - PyTorch framework
- `tiktoken>=0.5.0` - GPT tokenization
- `numpy>=1.24.0,<2.0.0` - Numerical operations
- `psutil>=5.9.0` - System utilities
- `tqdm>=4.65.0` - Progress bars

### Optional Dependencies

For enhanced functionality, install optional dependencies:

```bash
# For BERT/RoBERTa/DistilBERT/T5 tokenization
pip install transformers>=4.30.0

# For training visualization
pip install tensorboard>=2.13.0

# For model summaries
pip install torchsummary>=1.5.1

# For T5 evaluation metrics
pip install nltk>=3.8 rouge-score>=0.1.2
```

## 🎯 Quick Start

### GPT Model

```python
from gpt_kit import GPTModel, create_model_config, create_pretraining_dataloader

# Create model
config = create_model_config("gpt2-small (124M)")
model = GPTModel(config)

# Create data loader
text = "Your training text here..."
train_loader = create_pretraining_dataloader(
    text=text, batch_size=4, max_length=256
)

# Train
import torch
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for input_batch, target_batch in train_loader:
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_batch.view(-1)
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### BERT Classification

```python
from bert_kit import (
    BERTForSequenceClassification,
    create_model_config,
    create_classification_dataloader,
    get_tokenizer
)

# Create model
config = create_model_config("bert-base-uncased (110M)")
model = BERTForSequenceClassification(config, num_labels=2)

# Create data loader
texts = ["Great movie!", "Terrible experience"]
labels = [1, 0]
tokenizer = get_tokenizer("bert-base-uncased")
train_loader = create_classification_dataloader(
    texts=texts, labels=labels, tokenizer=tokenizer, batch_size=8
)

# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
for batch in train_loader:
    logits = model(batch["input_ids"], batch["attention_mask"])
    loss = torch.nn.functional.cross_entropy(logits, batch["labels"])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## 📚 Package Structure

### GPT Kit
```
gpt_kit/
├── __init__.py          # Package initialization and exports
├── config.py            # Configuration constants and model configs
├── model.py             # GPT model architecture components
├── data.py              # Dataset classes and data loading
├── training.py          # Training utilities
├── evaluation.py        # Model evaluation functions
├── utils.py             # Utility functions
├── amp.py               # Mixed precision training
├── callbacks.py         # EarlyStopping, TensorBoard
├── schedulers.py        # Learning rate schedulers
└── common.py            # Common utilities
```

### BERT Kit
```
bert_kit/
├── __init__.py          # Package initialization and exports
├── config.py            # Configuration constants and model configs
├── model.py             # BERT model architecture components
├── data.py              # Dataset classes and data loading
├── training.py          # Training utilities
├── evaluation.py        # Model evaluation functions
├── utils.py             # Utility functions
├── metrics.py           # Evaluation metrics
├── amp.py               # Mixed precision training
└── common.py            # Common utilities
```

### RoBERTa Kit
```
roberta_kit/
├── __init__.py          # Package initialization and exports
├── config.py            # Configuration constants and model configs
├── model.py             # RoBERTa model architecture components
├── data.py              # Dataset classes with dynamic masking
├── training.py          # Training utilities
├── utils.py             # Utility functions
├── amp.py               # Mixed precision training
├── metrics.py           # Evaluation metrics
└── common.py            # Common utilities
```

### DistilBERT Kit
```
distilbert_kit/
├── __init__.py          # Package initialization and exports
├── config.py            # Configuration constants and model configs
├── model.py             # DistilBERT model architecture components
├── data.py              # Dataset classes and data loading
├── training.py          # Training utilities
├── utils.py             # Utility functions
├── amp.py               # Mixed precision training
├── metrics.py           # Evaluation metrics
└── common.py            # Common utilities
```

### T5 Kit
```
t5_kit/
├── __init__.py          # Package initialization and exports
├── config.py            # Configuration constants and model configs
├── model.py             # T5 encoder-decoder architecture
├── data.py              # Text-to-text dataset classes
├── training.py          # Training utilities
├── utils.py             # Utility functions
├── amp.py               # Mixed precision training
├── metrics.py           # BLEU/ROUGE evaluation metrics
└── common.py            # Common utilities
```

### Shared Package (`llm_common/`)
```
llm_common/
├── __init__.py          # Package exports
├── device.py            # Device management
├── model_utils.py       # Model utilities (count_parameters, print_model_summary)
├── checkpoint.py        # Model checkpointing (save_model, load_model)
├── quantization.py      # Model quantization (dynamic/static)
└── distributed.py       # Distributed training utilities
```

### Tests (`tests/`)
```
tests/
├── __init__.py
├── test_common.py      # Tests for shared utilities
├── test_kits.py        # Tests for all LLM kits
├── test_integration.py # Integration tests (complete workflows)
├── benchmarks.py       # Performance benchmarks
└── README.md           # Test documentation
```

### Examples (`examples/`)
```
examples/
├── README.md                    # Examples documentation
├── train_gpt.py                 # GPT training example
├── train_bert_classification.py # BERT classification example
├── train_roberta.py             # RoBERTa classification example
├── train_distilbert.py          # DistilBERT classification example
├── train_t5.py                  # T5 training example
├── train_t5_with_metrics.py    # T5 with BLEU/ROUGE evaluation
├── quantize_model.py            # Model quantization example
└── distributed_training.py      # Multi-GPU training example
```

## 🎓 Model Configurations

### GPT Models
- `gpt2-small (124M)` - 768 embedding dim, 12 layers, 12 heads
- `gpt2-medium (355M)` - 1024 embedding dim, 24 layers, 16 heads
- `gpt2-large (774M)` - 1280 embedding dim, 36 layers, 20 heads
- `gpt2-xl (1558M)` - 1600 embedding dim, 48 layers, 25 heads

### BERT Models
- `bert-base-uncased (110M)` - 768 hidden size, 12 layers, 12 heads
- `bert-large-uncased (340M)` - 1024 hidden size, 24 layers, 16 heads
- `bert-base-cased (110M)` - 768 hidden size, 12 layers, 12 heads
- `bert-large-cased (340M)` - 1024 hidden size, 24 layers, 16 heads

### RoBERTa Models
- `roberta-base (125M)` - 768 hidden size, 12 layers, 12 heads
- `roberta-large (355M)` - 1024 hidden size, 24 layers, 16 heads

### DistilBERT Models
- `distilbert-base-uncased (66M)` - 768 hidden size, 6 layers, 12 heads

### T5 Models
- `t5-small (60M)` - 512 d_model, 6 encoder/decoder layers, 8 heads
- `t5-base (220M)` - 768 d_model, 12 encoder/decoder layers, 12 heads

## 🔥 Advanced Features

### Mixed Precision Training

Both kits support automatic mixed precision (AMP) training for faster training and reduced memory usage.

**Benefits:**
- **2x faster training** on compatible GPUs (V100, A100, RTX series)
- **~50% memory reduction** allowing larger batch sizes
- **Minimal accuracy impact** when used correctly

**Usage:**

**GPT Kit:**
```python
from gpt_kit import AMPTrainer

trainer = AMPTrainer(dtype='float16')  # Use 'bfloat16' for A100/H100
for batch in dataloader:
    loss = trainer.train_step(
        input_batch, target_batch, model, optimizer, device,
        max_grad_norm=1.0
    )
```

**BERT/RoBERTa/DistilBERT Kit:**
```python
from bert_kit import AMPTrainer

trainer = AMPTrainer(dtype='float16')
# For MLM
loss = trainer.train_mlm_step(input_ids, labels, model, optimizer, device)
# For classification
loss = trainer.train_classification_step(
    input_ids, attention_mask, labels, model, optimizer, device
)
```

**T5 Kit:**
```python
from t5_kit import AMPTrainer

trainer = AMPTrainer(dtype='float16')
loss = trainer.train_step(
    input_ids, decoder_input_ids, labels, model, optimizer, device
)
```

### Model Quantization

Reduce model size and speed up inference with quantization:

```python
from llm_common.quantization import quantize_model_dynamic, compare_model_sizes

# Quantize model
quantized_model = quantize_model_dynamic(model, dtype=torch.qint8)

# Compare sizes
comparison = compare_model_sizes(model, quantized_model)
print(f"Compression ratio: {comparison['compression_ratio']:.2f}x")
print(f"Size reduction: {comparison['size_reduction_percent']:.1f}%")
```

See `examples/quantize_model.py` for a complete example.

### Distributed Training

Train on multiple GPUs:

```python
from llm_common.distributed import wrap_model_for_distributed

# Wrap model for multi-GPU training
model = wrap_model_for_distributed(model, device_ids=[0, 1])

# Model automatically uses DataParallel or DistributedDataParallel
```

See `examples/distributed_training.py` for a complete example.

### T5 Evaluation Metrics

Evaluate T5 text generation with BLEU and ROUGE:

```python
from t5_kit import compute_bleu_score, compute_rouge_score

# Compute BLEU scores
bleu_scores = compute_bleu_score(references_tokens, candidates_tokens)
print(f"BLEU-4: {bleu_scores['bleu_4']:.4f}")

# Compute ROUGE scores
rouge_scores = compute_rouge_score(references_texts, candidates_texts)
print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
```

**Note**: Install `nltk` for BLEU and `rouge-score` for ROUGE:
```bash
pip install nltk rouge-score
```

### Performance Benchmarks

Run benchmarks to measure model performance:

```bash
# Model performance benchmarks
python tests/benchmarks.py

# Checkpoint operation benchmarks
python tests/benchmark_checkpoints.py
```

**Model Benchmarks** measure:
- Model size (MB)
- Inference speed (samples/second)
- Training speed (tokens/second)
- Memory usage

**Checkpoint Benchmarks** measure:
- Checkpoint save/load performance
- Checkpoint file sizes
- Throughput (checkpoints/second)
- Performance with optimizer state

### Integration Tests

Run integration tests for complete workflows:

```bash
# Basic integration tests
python tests/test_integration.py

# Comprehensive integration tests
python tests/test_integration_comprehensive.py
```

**Basic Tests** cover:
- Complete training workflows
- Model saving/loading
- Evaluation metrics
- Mixed precision training
- Shared utilities

**Comprehensive Tests** cover:
- End-to-end training pipelines
- Cross-kit checkpoint compatibility
- Evaluation workflows for all kits
- Mixed precision training workflows
- Shared utilities across kits

## 📖 Complete Training Examples

### GPT Kit - Full Training Loop with Advanced Features

```python
import torch
from gpt_kit import (
    GPTModel, create_model_config, create_pretraining_dataloader,
    AMPTrainer, get_cosine_scheduler, EarlyStopping, save_model, get_device
)

# Configuration
config = create_model_config("gpt2-small (124M)")
model = GPTModel(config).to(get_device())
train_loader = create_pretraining_dataloader(text="...", batch_size=4, max_length=256)

# Setup
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = get_cosine_scheduler(optimizer, num_warmup_steps=100, num_training_steps=1000)
trainer = AMPTrainer(dtype='float16')
early_stopping = EarlyStopping(patience=3)

# Training loop
for epoch in range(10):
    total_loss = 0.0
    for input_batch, target_batch in train_loader:
        loss = trainer.train_step(
            input_batch, target_batch, model, optimizer, "cuda",
            max_grad_norm=1.0
        )
        total_loss += loss
        scheduler.step()
    
    avg_loss = total_loss / len(train_loader)
    if early_stopping(avg_loss, model):
        break

save_model(model, "checkpoint.pt", config)
```

### BERT Kit - Classification Training with Metrics

```python
from bert_kit import (
    BERTForSequenceClassification, create_model_config,
    create_classification_dataloader, get_tokenizer,
    AMPTrainer, evaluate_classification_metrics
)

# Setup
config = create_model_config("bert-base-uncased (110M)")
model = BERTForSequenceClassification(config, num_labels=2)
tokenizer = get_tokenizer("bert-base-uncased")
train_loader = create_classification_dataloader(
    texts=["...", "..."], labels=[1, 0], tokenizer=tokenizer
)

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
trainer = AMPTrainer(dtype='float16')

for epoch in range(3):
    for batch in train_loader:
        loss = trainer.train_classification_step(
            batch["input_ids"], batch["attention_mask"],
            batch["labels"], model, optimizer, "cuda"
        )
    
    # Evaluate
    metrics = evaluate_classification_metrics(model, val_loader, "cuda", num_classes=2)
    print(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
```

### BERT Kit - MLM Pretraining with Mixed Precision

```python
from bert_kit import (
    BERTForMaskedLM, create_model_config, create_mlm_dataloader,
    AMPTrainer, compute_mlm_loader_loss
)

# Setup
config = create_model_config("bert-base-uncased (110M)")
model = BERTForMaskedLM(config)
train_loader = create_mlm_dataloader(texts=["..."], tokenizer=tokenizer)

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
trainer = AMPTrainer(dtype='float16')

for epoch in range(3):
    for batch in train_loader:
        loss = trainer.train_mlm_step(
            batch["input_ids"], batch["labels"], model, optimizer, "cuda"
        )
    
    # Evaluate perplexity
    perplexity = compute_mlm_loader_loss(model, val_loader, "cuda")
    print(f"Epoch {epoch+1} - Perplexity: {perplexity:.2f}")
```

## 🔧 API Reference

### GPT Kit

**Models:**
- `GPTModel(config)` - Base GPT model
- `create_model_config(name)` - Create model configuration

**Data:**
- `create_pretraining_dataloader(text, batch_size, max_length, ...)` - Create pretraining dataloader
- `create_instruction_collate_fn(...)` - Instruction dataset collate function

**Training:**
- `AMPTrainer(dtype='float16')` - Mixed precision trainer
- `train_step_with_accumulation(...)` - Training step with gradient accumulation
- `get_cosine_scheduler(...)` - Cosine learning rate scheduler
- `EarlyStopping(patience=3)` - Early stopping callback

**Utilities:**
- `save_model(model, path, config, ...)` - Save model checkpoint
- `load_model(path, model_class, ...)` - Load model checkpoint
- `generate_text_autoregressive(model, prompt, ...)` - Generate text

### BERT/RoBERTa/DistilBERT Kit

**Models:**
- `BERTModel(config)` / `RoBERTaModel(config)` / `DistilBERTModel(config)` - Base models
- `BERTForMaskedLM(config)` / `RoBERTaForMaskedLM(config)` / `DistilBERTForMaskedLM(config)` - MLM models
- `BERTForSequenceClassification(config, num_labels)` - Classification models

**Data:**
- `create_mlm_dataloader(texts, tokenizer, ...)` - MLM dataloader
- `create_classification_dataloader(texts, labels, tokenizer, ...)` - Classification dataloader

**Training:**
- `AMPTrainer(dtype='float16')` - Mixed precision trainer
- `train_mlm_step(...)` / `train_classification_step(...)` - Training steps

**Evaluation:**
- `evaluate_classification_metrics(model, dataloader, device, num_classes)` - Classification metrics
- `compute_precision_recall_f1(predictions, labels, ...)` - Precision/recall/F1
- `compute_confusion_matrix(predictions, labels, ...)` - Confusion matrix

### T5 Kit

**Models:**
- `T5Model(config)` - Base T5 encoder-decoder model
- `T5ForConditionalGeneration(config)` - Text-to-text generation model

**Data:**
- `create_t5_dataloader(inputs, targets, tokenizer, ...)` - Text-to-text dataloader

**Training:**
- `AMPTrainer(dtype='float16')` - Mixed precision trainer
- `train_step(...)` - Training step

**Evaluation:**
- `compute_bleu_score(references, candidates, ...)` - BLEU scores
- `compute_rouge_score(references, candidates, ...)` - ROUGE scores

### Shared Utilities (`llm_common`)

**Device Management:**
- `get_device(device=None)` - Get best available device

**Model Utilities:**
- `count_parameters(model, trainable_only=False)` - Count parameters
- `print_model_summary(model, input_size=None)` - Print model summary

**Checkpointing:**
- `save_model(model, path, config, ...)` - Save checkpoint
- `load_model(path, model_class, ...)` - Load checkpoint

**Quantization:**
- `quantize_model_dynamic(model, dtype=torch.qint8)` - Dynamic quantization
- `get_model_size_mb(model)` - Get model size
- `compare_model_sizes(original, quantized)` - Compare sizes

**Distributed Training:**
- `wrap_model_for_distributed(model, device_ids=None)` - Wrap for multi-GPU
- `setup_distributed(rank, world_size, backend='nccl')` - Setup distributed
- `get_world_size()` - Get number of processes
- `is_main_process()` - Check if main process

## 🧪 Testing

### Unit Tests

```bash
# Test shared utilities
python tests/test_common.py

# Test all kits
python tests/test_kits.py
```

### Integration Tests

```bash
# Test complete workflows
python tests/test_integration.py
```

### Benchmarks

```bash
# Run performance benchmarks
python tests/benchmarks.py
```

## 📝 Examples

See the `examples/` directory for complete, runnable examples:

- `train_gpt.py` - GPT training with advanced features
- `train_bert_classification.py` - BERT classification
- `train_roberta.py` - RoBERTa classification
- `train_distilbert.py` - DistilBERT classification
- `train_t5.py` - T5 text-to-text training
- `train_t5_with_metrics.py` - T5 with BLEU/ROUGE evaluation
- `quantize_model.py` - Model quantization
- `distributed_training.py` - Multi-GPU training

Run examples:
```bash
python examples/train_gpt.py
python examples/train_bert_classification.py
# ... etc
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory (OOM) Errors

**Symptoms:** `RuntimeError: CUDA out of memory`

**Solutions:**
- **Reduce batch size**: Decrease `batch_size` in DataLoader creation
- **Use gradient accumulation**: Accumulate gradients over multiple batches
- **Enable mixed precision training**: Reduces memory usage significantly
- **Use smaller model**: Try `gpt2-small` instead of `gpt2-large`
- **Reduce sequence length**: Lower `max_length` parameter

#### 2. Slow Training Speed

**Solutions:**
- **Enable mixed precision training**: Use FP16/BF16 for faster training
- **Use larger batch sizes**: If memory allows
- **Enable gradient accumulation**: Simulate larger batches
- **Use quantization**: For faster CPU inference

#### 3. Import Errors

**Solutions:**
- Install missing dependencies: `pip install transformers tensorboard`
- Check Python version: Requires Python 3.8+
- Verify PyTorch installation: `python -c "import torch; print(torch.__version__)"`

#### 4. Tokenizer Issues

**Solutions:**
- Install transformers: `pip install transformers`
- Falls back to simple tokenization if transformers not available
- Use appropriate tokenizer for each model type

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Code Style**: Follow PEP 8, use type hints
2. **Testing**: Add tests for new features
3. **Documentation**: Update docstrings and README
4. **Examples**: Add examples for new features

## 📄 License

This package is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- GPT architecture based on OpenAI's GPT-2
- BERT architecture based on Google's BERT
- RoBERTa improvements from Facebook AI
- DistilBERT from Hugging Face
- T5 from Google Research

## 📚 Additional Resources

- **Examples**: See `examples/README.md`
- **Tests**: See `tests/README.md`
- **Codebase Review**: See `CODEBASE_REVIEW.md`

---

**Status**: 🟢 Production Ready

**Last Updated**: December 2025
