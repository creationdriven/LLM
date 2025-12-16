# DistilBERT Kit

A Python package for building, training, fine-tuning, and evaluating DistilBERT models - a smaller, faster, and lighter version of BERT.

## 🎯 What Makes DistilBERT Unique?

DistilBERT is a **distilled** version of BERT that achieves 97% of BERT's performance with **60% fewer parameters** and **60% faster inference**.

### Key Characteristics:

1. **Knowledge Distillation**: Trained to mimic BERT's behavior using a smaller architecture
2. **Smaller Architecture**: 6 layers instead of 12 (half the size)
3. **No Token Type Embeddings**: Removed segment embeddings to reduce parameters
4. **Faster Inference**: 60% faster than BERT while maintaining performance
5. **Lower Memory**: Requires less memory for training and inference

### Best Use Cases:

- **Resource-Constrained Environments**: Mobile devices, edge computing
- **Real-Time Applications**: Where speed is critical
- **Large-Scale Deployment**: When deploying many models simultaneously
- **All BERT Tasks**: Classification, NER, QA (with slight performance trade-off)

## 🏗️ Model Architecture

### High-Level Structure:

```
Input Tokens → Token Embeddings → Position Embeddings → 
Transformer Encoder Blocks (6 layers) → [CLS] Token → Task-Specific Head
```

### Core Components:

1. **Embeddings**:
   - **Token Embeddings**: Convert token IDs to dense vectors
   - **Position Embeddings**: Learned positional encodings
   - **No Segment Embeddings**: Removed to reduce parameters
2. **Transformer Encoder Blocks** (6 layers, vs. 12 in BERT):
   - **Bidirectional Multi-Head Attention**: Full bidirectional self-attention
   - **Feed-Forward Network**: Two linear layers with GELU activation
   - **Layer Normalization**: Applied after attention and FFN
   - **Residual Connections**: Skip connections
3. **Task-Specific Heads**: Same as BERT (MLM, Classification)

### Architecture Details:

- **Half the Layers**: 6 encoder layers instead of 12
- **Same Hidden Size**: 768 (same as BERT-base)
- **Same Attention Heads**: 12 heads per layer
- **No Segment Embeddings**: Removed to save parameters
- **Layer Normalization**: Uses epsilon=1e-12 (same as BERT)

## 📁 Code Structure

### `model.py` - Model Architecture

**Key Classes:**

- `LayerNorm`: Layer normalization with BERT-style epsilon (1e-12)
- `GELU`: Gaussian Error Linear Unit activation
- `FeedForward`: Two-layer MLP with GELU and dropout
- `MultiHeadAttention`: Bidirectional multi-head self-attention
- `TransformerEncoderBlock`: Single encoder layer
- `DistilBERTModel`: Base DistilBERT encoder model
- `DistilBERTForMaskedLM`: DistilBERT with MLM head
- `DistilBERTForSequenceClassification`: DistilBERT with classification head

**How It Works:**

1. **Input Processing**: Token IDs → Embeddings → Add Position Embeddings (no segment embeddings)
2. **Encoder Layers**: 6 layers (half of BERT) apply bidirectional attention then FFN
3. **Output**: Same as BERT (all hidden states, MLM logits, or classification logits)

### `data.py` - Data Loading

**Key Classes:**

- `MLMDataset`: Dataset for MLM pretraining
- `ClassificationDataset`: Dataset for classification fine-tuning

**How It Works:**

- Same data loading as BERT
- No segment embeddings needed
- **Tokenization**: Uses `transformers.DistilBertTokenizer` (WordPiece tokenization)

### `training.py` - Training Utilities

**Key Functions:**

- `compute_mlm_loss`: Calculate MLM loss
- `compute_classification_loss`: Calculate classification loss
- `compute_mlm_loader_loss`: Evaluate average MLM loss

**How It Works:**

- Same loss computation as BERT
- Faster training due to fewer parameters

### `metrics.py` - Evaluation Metrics

**Key Functions:**

- `compute_precision_recall_f1`: Calculate precision, recall, F1 scores
- `compute_confusion_matrix`: Generate confusion matrix
- `evaluate_classification_metrics`: Comprehensive classification evaluation

**How It Works:**

- Same evaluation metrics as BERT

### `amp.py` - Mixed Precision Training

**Key Classes:**

- `AMPTrainer`: Automatic Mixed Precision trainer for DistilBERT

**How It Works:**

- Supports both MLM and classification training steps
- Uses FP16/BF16 with gradient scaling

## 🔑 Key Differences from BERT

| Feature | DistilBERT | BERT |
|---------|------------|------|
| Layers | 6 | 12 |
| Parameters | ~66M | ~110M |
| Speed | 60% faster | Baseline |
| Performance | ~97% of BERT | 100% |
| Segment Embeddings | ❌ Not used | ✅ Used |
| Memory Usage | Lower | Higher |

## 📝 Usage Examples

### Basic Model Creation

```python
from distilbert_kit import DistilBERTModel, create_model_config

config = create_model_config("distilbert-base-uncased (66M)")
model = DistilBERTModel(config)

# Check parameter count
from llm_common import count_parameters
print(f"Parameters: {count_parameters(model):,}")  # ~66M
```

### MLM Pretraining

```python
from distilbert_kit import DistilBERTForMaskedLM, create_mlm_dataloader, get_tokenizer

config = create_model_config("distilbert-base-uncased (66M)")
model = DistilBERTForMaskedLM(config)

tokenizer = get_tokenizer("distilbert-base-uncased")
train_loader = create_mlm_dataloader(
    texts=["Your training text..."],
    tokenizer=tokenizer,
    batch_size=16,  # Can use larger batches due to smaller model
    mask_probability=0.15
)
```

### Classification Fine-tuning

```python
from distilbert_kit import (
    DistilBERTForSequenceClassification,
    create_classification_dataloader,
    get_tokenizer
)

config = create_model_config("distilbert-base-uncased (66M)")
model = DistilBERTForSequenceClassification(config, num_labels=2)

texts = ["Great movie!", "Terrible experience"]
labels = [1, 0]

tokenizer = get_tokenizer("distilbert-base-uncased")
train_loader = create_classification_dataloader(
    texts=texts, labels=labels, tokenizer=tokenizer, batch_size=16  # Larger batches possible
)
```

## 🎓 Understanding the Code Flow

### Knowledge Distillation:

DistilBERT is trained using knowledge distillation:
1. **Teacher Model**: BERT (large, accurate)
2. **Student Model**: DistilBERT (small, fast)
3. **Training**: Student learns to mimic teacher's predictions
4. **Result**: Small model with BERT-like performance

### Why It's Faster:

1. **Fewer Layers**: 6 vs. 12 layers means fewer computations
2. **Fewer Parameters**: 66M vs. 110M means less memory access
3. **No Segment Embeddings**: Saves computation in embedding layer
4. **Same Architecture**: Still bidirectional, just smaller

## 🔧 Configuration

Model configurations are defined in `config.py`:

- `vocab_size`: Vocabulary size (30,522 for DistilBERT)
- `hidden_size`: Hidden dimension (768, same as BERT-base)
- `num_hidden_layers`: Number of encoder layers (6, half of BERT)
- `num_attention_heads`: Number of attention heads (12, same as BERT-base)
- `max_position_embeddings`: Maximum sequence length (512, same as BERT)

## 📚 Further Reading

- DistilBERT Paper: "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"
- DistilBERT GitHub: https://github.com/huggingface/transformers/tree/main/models/distilbert

## 🚀 Quick Start

See `examples/train_distilbert.py` for a complete classification example.

