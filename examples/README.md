# Example Scripts

This directory contains example scripts demonstrating how to use each LLM kit.

## Available Examples

### Training Examples

#### `train_gpt.py`
Complete GPT training example with:
- Model initialization
- Data loading
- Mixed precision training
- Learning rate scheduling
- Early stopping
- Model checkpointing

**Usage:**
```bash
python examples/train_gpt.py
```

#### `train_bert_classification.py`
BERT classification training example with:
- Text classification setup
- Data preparation
- Training with evaluation metrics
- Model evaluation

**Usage:**
```bash
python examples/train_bert_classification.py
```

#### `train_roberta.py`
RoBERTa classification training example with:
- Dynamic masking demonstration
- Classification training
- Mixed precision training
- Evaluation metrics

**Usage:**
```bash
python examples/train_roberta.py
```

#### `train_roberta_mlm.py`
RoBERTa MLM pretraining example with:
- Masked Language Modeling
- Dynamic masking (pattern changes each epoch)
- Perplexity evaluation

**Usage:**
```bash
python examples/train_roberta_mlm.py
```

#### `train_distilbert.py`
DistilBERT classification training example with:
- Fast training with smaller model
- Performance comparisons
- Mixed precision training

**Usage:**
```bash
python examples/train_distilbert.py
```

#### `train_distilbert_mlm.py`
DistilBERT MLM pretraining example with:
- Smaller, faster MLM training
- Performance benchmarking
- Model size comparisons

**Usage:**
```bash
python examples/train_distilbert_mlm.py
```

#### `train_t5.py`
T5 text-to-text training example with:
- Text-to-text data format
- Encoder-decoder training
- Mixed precision support

**Usage:**
```bash
python examples/train_t5.py
```

#### `train_t5_with_metrics.py`
T5 training with BLEU/ROUGE evaluation:
- Text-to-text training
- BLEU score computation
- ROUGE score computation

**Usage:**
```bash
python examples/train_t5_with_metrics.py
```

### Utility Examples

#### `load_checkpoint.py`
Checkpoint loading and usage examples:
- Loading saved models
- Resuming training from checkpoints
- Loading optimizer state
- Accessing checkpoint metadata

**Usage:**
```bash
python examples/load_checkpoint.py
```

#### `generate_text_gpt.py`
Text generation with GPT:
- Loading trained models
- Text generation with different sampling strategies
- Temperature and top-k sampling
- Generating from prompts

**Usage:**
```bash
python examples/generate_text_gpt.py
```

#### `evaluate_classification.py`
Comprehensive classification evaluation:
- Loading trained models
- Computing evaluation metrics (accuracy, precision, recall, F1)
- Confusion matrix analysis
- Per-class metrics

**Usage:**
```bash
python examples/evaluate_classification.py
```

#### `quantize_model.py`
Model quantization example:
- Dynamic quantization
- Model size comparison
- Memory usage reduction

**Usage:**
```bash
python examples/quantize_model.py
```

#### `distributed_training.py`
Distributed/multi-GPU training example:
- Multi-GPU setup
- Model wrapping for DataParallel/DDP
- Distributed utilities

**Usage:**
```bash
python examples/distributed_training.py
```

## Running Examples

All examples are executable and can be run directly:

```bash
# Make scripts executable (if needed)
chmod +x examples/*.py

# Run training examples
python examples/train_gpt.py
python examples/train_bert_classification.py
python examples/train_roberta.py
python examples/train_distilbert.py
python examples/train_t5.py

# Run utility examples
python examples/load_checkpoint.py
python examples/generate_text_gpt.py
python examples/evaluate_classification.py
python examples/quantize_model.py
python examples/distributed_training.py
```

## Example Categories

### Training Examples (8)
- GPT training with advanced features
- BERT classification
- RoBERTa classification and MLM
- DistilBERT classification and MLM
- T5 text-to-text with metrics

### Utility Examples (5)
- Checkpoint loading
- Text generation
- Model evaluation
- Model quantization
- Distributed training

## Notes

- Examples use CPU by default if CUDA is not available
- Sample data is included for demonstration
- Models are saved to `examples/` directory
- Adjust hyperparameters in scripts as needed
- Some examples require pre-trained checkpoints (they will create new models if not found)

