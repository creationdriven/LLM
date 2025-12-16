# BERT Kit

A Python package for building, training, fine-tuning, and evaluating BERT-style bidirectional language models.

## 🎯 What Makes BERT Unique?

BERT (Bidirectional Encoder Representations from Transformers) is an **encoder-only** model designed for **understanding** tasks rather than generation.

### Key Characteristics:

1. **Bidirectional Context**: Processes text in both directions simultaneously, seeing full context
2. **Masked Language Modeling (MLM)**: Pre-trained by predicting masked tokens using bidirectional context
3. **Next Sentence Prediction (NSP)**: Pre-trained to understand sentence relationships
4. **No Generation**: Unlike GPT, BERT doesn't generate text - it produces contextualized embeddings

### Best Use Cases:

- **Text Classification**: Sentiment analysis, topic classification, spam detection
- **Named Entity Recognition (NER)**: Extract entities from text
- **Question Answering**: Extract answers from context
- **Sentence Similarity**: Measure semantic similarity between sentences
- **Token Classification**: Part-of-speech tagging, NER at token level

## 🏗️ Model Architecture

### High-Level Structure:

```
Input Tokens → Token Embeddings → Segment Embeddings → Position Embeddings → 
Transformer Encoder Blocks → [CLS] Token → Task-Specific Head
```

### Core Components:

1. **Embeddings**:
   - **Token Embeddings**: Convert token IDs to dense vectors
   - **Segment Embeddings**: Distinguish sentence A from sentence B (for NSP)
   - **Position Embeddings**: Learned positional encodings
2. **Transformer Encoder Blocks** (N layers):
   - **Bidirectional Multi-Head Attention**: Full bidirectional self-attention
   - **Feed-Forward Network**: Two linear layers with GELU activation
   - **Layer Normalization**: Applied after attention and FFN (post-norm)
   - **Residual Connections**: Skip connections around attention and FFN
3. **Task-Specific Heads**:
   - **MLM Head**: Predict masked tokens
   - **Classification Head**: Use [CLS] token for classification
   - **Token Classification Head**: Predict label for each token

### Architecture Details:

- **Special Tokens**: [CLS] (classification), [SEP] (separator), [MASK] (masking)
- **Attention**: Full bidirectional (no masking)
- **Layer Normalization**: Uses epsilon=1e-12 (BERT-specific), applied after sub-layers
- **Activation**: GELU
- **Dropout**: Applied to embeddings, attention, and FFN

## 📁 Code Structure

### `model.py` - Model Architecture

**Key Classes:**

- `LayerNorm`: Layer normalization with BERT-specific epsilon (1e-12)
- `GELU`: Gaussian Error Linear Unit activation
- `FeedForward`: Two-layer MLP with GELU and dropout
- `MultiHeadAttention`: Bidirectional multi-head self-attention
- `TransformerEncoderBlock`: Single encoder layer (attention + FFN)
- `BERTModel`: Base BERT encoder model
- `BERTForMaskedLM`: BERT with MLM head
- `BERTForSequenceClassification`: BERT with classification head

**How It Works:**

1. **Input Processing**: Token IDs + Segment IDs → Embeddings → Sum all embeddings
2. **Encoder Layers**: Each layer applies bidirectional attention then FFN
3. **Output**: 
   - **Base Model**: Returns all hidden states
   - **MLM**: Returns logits for masked token prediction
   - **Classification**: Returns logits for classification task

### `data.py` - Data Loading

**Key Classes:**

- `MLMDataset`: Dataset for Masked Language Modeling pretraining
- `ClassificationDataset`: Dataset for classification fine-tuning
- `NSPDataset`: Dataset for Next Sentence Prediction (optional)

**How It Works:**

- **MLM**: Randomly masks 15% of tokens (80% [MASK], 10% random, 10% unchanged)
- **Classification**: Formats text-label pairs with [CLS] and [SEP] tokens
- **Tokenization**: Uses `transformers.BertTokenizer` (WordPiece tokenization)

### `training.py` - Training Utilities

**Key Functions:**

- `compute_mlm_loss`: Calculate MLM loss (cross-entropy for masked tokens)
- `compute_classification_loss`: Calculate classification loss
- `compute_mlm_loader_loss`: Evaluate average MLM loss on dataloader

**How It Works:**

- **MLM Loss**: Only computed for masked positions (ignore_index=-100 for non-masked)
- **Classification Loss**: Standard cross-entropy for classification tasks
- Supports gradient clipping and accumulation

### `metrics.py` - Evaluation Metrics

**Key Functions:**

- `compute_precision_recall_f1`: Calculate precision, recall, F1 scores
- `compute_confusion_matrix`: Generate confusion matrix
- `evaluate_classification_metrics`: Comprehensive classification evaluation

**How It Works:**

- Computes metrics from predictions and true labels
- Handles multi-class classification
- Provides per-class metrics

### `amp.py` - Mixed Precision Training

**Key Classes:**

- `AMPTrainer`: Automatic Mixed Precision trainer for BERT

**How It Works:**

- Supports both MLM and classification training steps
- Uses FP16/BF16 with gradient scaling
- Handles attention masks and token type IDs

## 🔑 Key Differences from Other Kits

| Feature | BERT | GPT | T5 |
|---------|------|-----|-----|
| Architecture | Encoder-only | Decoder-only | Encoder-Decoder |
| Attention | Bidirectional | Causal (masked) | Bidirectional (encoder) |
| Task | Understanding | Generation | Text-to-text |
| Direction | Both directions | Left-to-right | Both (encoder) |
| Pre-training | MLM + NSP | Next token prediction | Span corruption |
| Use Case | Classification/QA | Text generation | Translation/Summarization |

## 📝 Usage Examples

### Basic Model Creation

```python
from bert_kit import BERTModel, create_model_config

config = create_model_config("bert-base-uncased (110M)")
model = BERTModel(config)
```

### MLM Pretraining

```python
from bert_kit import BERTForMaskedLM, create_mlm_dataloader, get_tokenizer

config = create_model_config("bert-base-uncased (110M)")
model = BERTForMaskedLM(config)

tokenizer = get_tokenizer("bert-base-uncased")
train_loader = create_mlm_dataloader(
    texts=["Your training text..."],
    tokenizer=tokenizer,
    batch_size=8,
    mask_probability=0.15
)
```

### Classification Fine-tuning

```python
from bert_kit import (
    BERTForSequenceClassification,
    create_classification_dataloader,
    get_tokenizer
)

config = create_model_config("bert-base-uncased (110M)")
model = BERTForSequenceClassification(config, num_labels=2)

texts = ["Great movie!", "Terrible experience"]
labels = [1, 0]

tokenizer = get_tokenizer("bert-base-uncased")
train_loader = create_classification_dataloader(
    texts=texts, labels=labels, tokenizer=tokenizer, batch_size=8
)
```

### Evaluation

```python
from bert_kit import evaluate_classification_metrics

metrics = evaluate_classification_metrics(
    model, val_loader, device="cuda", num_classes=2
)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
```

## 🎓 Understanding the Code Flow

### MLM Pretraining Flow:

1. **Data Loading**: `create_mlm_dataloader` → Creates batches with masked tokens
2. **Forward Pass**: `model(input_ids)` → Returns logits for masked positions
3. **Loss Calculation**: Cross-entropy only for masked tokens
4. **Backward Pass**: Compute gradients
5. **Optimization**: Update weights

### Classification Flow:

1. **Data Loading**: `create_classification_dataloader` → Creates (text, label) pairs
2. **Forward Pass**: `model(input_ids, attention_mask)` → Returns classification logits
3. **Loss Calculation**: Cross-entropy between logits and labels
4. **Prediction**: Use [CLS] token representation for classification

### Why [CLS] Token?

The [CLS] token is placed at the beginning of the input sequence and is trained to aggregate information from the entire sequence. Its final hidden state is used for classification tasks.

## 🔧 Configuration

Model configurations are defined in `config.py`:

- `vocab_size`: Vocabulary size (30,522 for BERT-base)
- `hidden_size`: Hidden dimension (768 for BERT-base)
- `num_hidden_layers`: Number of encoder layers (12 for BERT-base)
- `num_attention_heads`: Number of attention heads (12 for BERT-base)
- `max_position_embeddings`: Maximum sequence length (512 for BERT)
- `type_vocab_size`: Number of segment types (2 for sentence pairs)

## 📚 Further Reading

- Original BERT Paper: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- BERT GitHub: https://github.com/google-research/bert

## 🚀 Quick Start

See `examples/train_bert_classification.py` for a complete classification example.

