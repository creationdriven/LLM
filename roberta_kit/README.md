# RoBERTa Kit

A Python package for building, training, fine-tuning, and evaluating RoBERTa (Robustly Optimized BERT Pretraining Approach) models.

## 🎯 What Makes RoBERTa Unique?

RoBERTa is an optimized version of BERT that removes the Next Sentence Prediction (NSP) task and uses **dynamic masking** during pretraining.

### Key Characteristics:

1. **No NSP Task**: Removed Next Sentence Prediction, focusing solely on MLM
2. **Dynamic Masking**: Masking pattern changes each epoch (vs. static masking in BERT)
3. **Larger Batch Sizes**: Trained with larger batches for better performance
4. **More Training Data**: Trained on more data than BERT
5. **Longer Training**: Trained for more steps than BERT

### Best Use Cases:

- **Text Classification**: Sentiment analysis, topic classification
- **Named Entity Recognition**: Extract entities from text
- **Question Answering**: Extract answers from context
- **Sentence Similarity**: Measure semantic similarity
- **All BERT tasks**: Generally outperforms BERT on most tasks

## 🏗️ Model Architecture

### High-Level Structure:

```
Input Tokens → Token Embeddings → Position Embeddings → 
Transformer Encoder Blocks → [CLS] Token → Task-Specific Head
```

### Core Components:

1. **Embeddings**:
   - **Token Embeddings**: Convert token IDs to dense vectors
   - **Position Embeddings**: Learned positional encodings
   - **No Segment Embeddings**: Unlike BERT, RoBERTa doesn't use segment embeddings
2. **Transformer Encoder Blocks** (N layers):
   - **Bidirectional Multi-Head Attention**: Full bidirectional self-attention
   - **Feed-Forward Network**: Two linear layers with GELU activation
   - **Layer Normalization**: Applied after attention and FFN (post-norm)
   - **Residual Connections**: Skip connections around attention and FFN
3. **Task-Specific Heads**: Same as BERT (MLM, Classification)

### Architecture Details:

- **Special Tokens**: [CLS], [SEP], [MASK] (same as BERT)
- **Attention**: Full bidirectional (no masking)
- **Layer Normalization**: Uses epsilon=1e-5 (vs. 1e-12 in BERT)
- **Activation**: GELU
- **Dynamic Masking**: Masking pattern regenerated each epoch

## 📁 Code Structure

### `model.py` - Model Architecture

**Key Classes:**

- `LayerNorm`: Layer normalization with RoBERTa-specific epsilon (1e-5)
- `GELU`: Gaussian Error Linear Unit activation
- `FeedForward`: Two-layer MLP with GELU and dropout
- `MultiHeadAttention`: Bidirectional multi-head self-attention
- `TransformerEncoderBlock`: Single encoder layer
- `RoBERTaModel`: Base RoBERTa encoder model
- `RoBERTaForMaskedLM`: RoBERTa with MLM head
- `RoBERTaForSequenceClassification`: RoBERTa with classification head

**How It Works:**

1. **Input Processing**: Token IDs → Embeddings → Add Position Embeddings (no segment embeddings)
2. **Encoder Layers**: Each layer applies bidirectional attention then FFN
3. **Output**: Same as BERT (all hidden states, MLM logits, or classification logits)

### `data.py` - Data Loading

**Key Classes:**

- `MLMDataset`: Dataset with dynamic masking for MLM pretraining
- `ClassificationDataset`: Dataset for classification fine-tuning

**How It Works:**

- **Dynamic Masking**: Masking pattern changes each epoch (regenerated on-the-fly)
- **MLM**: Randomly masks 15% of tokens (same percentages as BERT)
- **Classification**: Formats text-label pairs with [CLS] token
- **Tokenization**: Uses `transformers.RobertaTokenizer` (BPE tokenization)

### `training.py` - Training Utilities

**Key Functions:**

- `compute_mlm_loss`: Calculate MLM loss with dynamic masking
- `compute_classification_loss`: Calculate classification loss
- `compute_mlm_loader_loss`: Evaluate average MLM loss

**How It Works:**

- Same loss computation as BERT
- Dynamic masking handled in dataset (regenerated each epoch)

### `metrics.py` - Evaluation Metrics

**Key Functions:**

- `compute_precision_recall_f1`: Calculate precision, recall, F1 scores
- `compute_confusion_matrix`: Generate confusion matrix
- `evaluate_classification_metrics`: Comprehensive classification evaluation

**How It Works:**

- Same evaluation metrics as BERT
- Computes metrics from predictions and true labels

### `amp.py` - Mixed Precision Training

**Key Classes:**

- `AMPTrainer`: Automatic Mixed Precision trainer for RoBERTa

**How It Works:**

- Supports both MLM and classification training steps
- Uses FP16/BF16 with gradient scaling

## 🔑 Key Differences from BERT

| Feature | RoBERTa | BERT |
|---------|---------|------|
| NSP Task | ❌ Removed | ✅ Included |
| Masking | Dynamic (changes each epoch) | Static (fixed per example) |
| Segment Embeddings | ❌ Not used | ✅ Used |
| LayerNorm Epsilon | 1e-5 | 1e-12 |
| Training Data | More | Less |
| Training Steps | More | Less |
| Batch Size | Larger | Smaller |

## 📝 Usage Examples

### Basic Model Creation

```python
from roberta_kit import RoBERTaModel, create_model_config

config = create_model_config("roberta-base (125M)")
model = RoBERTaModel(config)
```

### MLM Pretraining with Dynamic Masking

```python
from roberta_kit import RoBERTaForMaskedLM, create_mlm_dataloader, get_tokenizer

config = create_model_config("roberta-base (125M)")
model = RoBERTaForMaskedLM(config)

tokenizer = get_tokenizer("roberta-base")
train_loader = create_mlm_dataloader(
    texts=["Your training text..."],
    tokenizer=tokenizer,
    batch_size=8,
    mask_probability=0.15
)

# Dynamic masking: pattern changes each epoch
for epoch in range(num_epochs):
    # Masking pattern regenerated automatically
    for batch in train_loader:
        # Train...
        pass
```

### Classification Fine-tuning

```python
from roberta_kit import (
    RoBERTaForSequenceClassification,
    create_classification_dataloader,
    get_tokenizer
)

config = create_model_config("roberta-base (125M)")
model = RoBERTaForSequenceClassification(config, num_labels=2)

texts = ["Great movie!", "Terrible experience"]
labels = [1, 0]

tokenizer = get_tokenizer("roberta-base")
train_loader = create_classification_dataloader(
    texts=texts, labels=labels, tokenizer=tokenizer, batch_size=8
)
```

## 🎓 Understanding the Code Flow

### Dynamic Masking:

1. **Epoch 1**: Text "The cat sat" → Masks "cat" → "The [MASK] sat"
2. **Epoch 2**: Same text → Masks "sat" → "The cat [MASK]"
3. **Epoch 3**: Same text → Masks "The" → "[MASK] cat sat"

This forces the model to learn from different masking patterns, improving robustness.

### Why Remove NSP?

Research showed that NSP didn't help performance and actually hurt it in some cases. Removing it simplifies training and improves results.

## 🔧 Configuration

Model configurations are defined in `config.py`:

- `vocab_size`: Vocabulary size (50,265 for RoBERTa)
- `hidden_size`: Hidden dimension (768 for RoBERTa-base)
- `num_hidden_layers`: Number of encoder layers (12 for RoBERTa-base)
- `num_attention_heads`: Number of attention heads (12 for RoBERTa-base)
- `max_position_embeddings`: Maximum sequence length (514 for RoBERTa)

## 📚 Further Reading

- RoBERTa Paper: "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
- RoBERTa GitHub: https://github.com/pytorch/fairseq/tree/main/examples/roberta

## 🚀 Quick Start

See `examples/train_roberta.py` for a complete classification example.

