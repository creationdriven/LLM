# T5 Kit

A Python package for building, training, fine-tuning, and evaluating T5 (Text-To-Text Transfer Transformer) models.

## 🎯 What Makes T5 Unique?

T5 is an **encoder-decoder** model that frames all NLP tasks as **text-to-text** problems, making it extremely versatile.

### Key Characteristics:

1. **Text-to-Text Framework**: All tasks (translation, summarization, classification) are framed as text generation
2. **Encoder-Decoder Architecture**: Separate encoder and decoder stacks
3. **Relative Position Embeddings**: Uses relative rather than absolute position encodings
4. **Span Corruption**: Pre-trained by corrupting spans of text and learning to reconstruct them
5. **Task Prefixes**: Uses prefixes like "translate:", "summarize:" to specify tasks

### Best Use Cases:

- **Translation**: Machine translation between languages
- **Summarization**: Abstractive text summarization
- **Question Answering**: Generate answers from context
- **Classification**: Text classification (as text generation)
- **Text-to-Text Tasks**: Any task that can be framed as text generation

## 🏗️ Model Architecture

### High-Level Structure:

```
Input Text → Encoder → Encoder Hidden States → 
Decoder (with Cross-Attention) → Output Text
```

### Core Components:

1. **Encoder Stack**:
   - **Token Embeddings**: Convert input token IDs to vectors
   - **Relative Position Embeddings**: Learn relative positions between tokens
   - **Encoder Blocks** (N layers):
     - Bidirectional self-attention
     - Feed-forward network
     - Layer normalization
2. **Decoder Stack**:
   - **Token Embeddings**: Convert output token IDs to vectors
   - **Relative Position Embeddings**: Relative positions for decoder
   - **Decoder Blocks** (N layers):
     - **Causal Self-Attention**: Masked attention (can't see future tokens)
     - **Cross-Attention**: Attend to encoder outputs
     - Feed-forward network
     - Layer normalization
3. **Output Layer**: Linear projection to vocabulary size

### Architecture Details:

- **Relative Position Bias**: Learned relative position embeddings (not absolute)
- **Cross-Attention**: Decoder attends to encoder outputs
- **Layer Normalization**: Uses epsilon=1e-6, applied before sub-layers
- **Activation**: ReLU (Rectified Linear Unit)
- **Shared Embeddings**: Input and output embeddings share weights

## 📁 Code Structure

### `model.py` - Model Architecture

**Key Classes:**

- `RelativePositionBias`: Computes relative position bias for attention
- `T5Attention`: Multi-head attention with relative position bias
- `T5FeedForward`: Feed-forward network with ReLU activation
- `T5EncoderBlock`: Single encoder layer
- `T5DecoderBlock`: Single decoder layer (with cross-attention)
- `T5Model`: Base encoder-decoder model
- `T5ForConditionalGeneration`: T5 model with language modeling head

**How It Works:**

1. **Encoder**: Processes input text bidirectionally
2. **Decoder**: Generates output text autoregressively
3. **Cross-Attention**: Decoder attends to encoder outputs at each step
4. **Output**: Generates text token by token

### `data.py` - Data Loading

**Key Classes:**

- `T5Dataset`: Dataset for text-to-text tasks

**How It Works:**

- Formats input-output pairs with task prefixes
- Handles tokenization and padding
- Creates decoder input IDs (shifted for teacher forcing)

### `training.py` - Training Utilities

**Key Functions:**

- `compute_t5_loss`: Calculate cross-entropy loss for text generation
- `compute_t5_loader_loss`: Evaluate average loss on dataloader

**How It Works:**

- Uses standard cross-entropy for next-token prediction
- Handles padding via ignore_index
- Supports teacher forcing during training

### `metrics.py` - Evaluation Metrics

**Key Functions:**

- `compute_bleu_score`: Calculate BLEU scores for generation quality
- `compute_rouge_score`: Calculate ROUGE scores for summarization
- `evaluate_t5_generation`: Comprehensive generation evaluation

**How It Works:**

- **BLEU**: Measures n-gram overlap between generated and reference text
- **ROUGE**: Measures recall-oriented metrics (good for summarization)
- Requires `nltk` for BLEU and `rouge-score` for ROUGE

### `amp.py` - Mixed Precision Training

**Key Classes:**

- `AMPTrainer`: Automatic Mixed Precision trainer for T5

**How It Works:**

- Supports encoder-decoder training with FP16/BF16
- Handles both encoder and decoder inputs
- Manages attention masks for both stacks

## 🔑 Key Differences from Other Kits

| Feature | T5 | GPT | BERT |
|---------|----|-----|------|
| Architecture | Encoder-Decoder | Decoder-only | Encoder-only |
| Task Framework | Text-to-text | Next token | Understanding |
| Position Encoding | Relative | Absolute/Cached | Learned |
| Attention | Bidirectional (encoder) + Causal (decoder) | Causal | Bidirectional |
| Pre-training | Span corruption | Next token | MLM + NSP |
| Use Case | Translation/Summarization | Generation | Classification |

## 📝 Usage Examples

### Basic Model Creation

```python
from t5_kit import T5ForConditionalGeneration, create_model_config

config = create_model_config("t5-small (60M)")
model = T5ForConditionalGeneration(config)
```

### Text-to-Text Training

```python
from t5_kit import create_t5_dataloader, get_tokenizer

# Translation task
inputs = ["translate English to French: Hello", "translate English to French: Goodbye"]
targets = ["Bonjour", "Au revoir"]

tokenizer = get_tokenizer("t5-small")
train_loader = create_t5_dataloader(
    inputs=inputs, targets=targets, tokenizer=tokenizer, batch_size=4
)
```

### Generation

```python
# T5 generates text autoregressively
# Input: "translate English to French: Hello"
# Output: "Bonjour"
```

### Evaluation

```python
from t5_kit import compute_bleu_score, compute_rouge_score

# BLEU scores
bleu = compute_bleu_score(references, candidates)
print(f"BLEU-4: {bleu['bleu_4']:.4f}")

# ROUGE scores
rouge = compute_rouge_score(references, candidates)
print(f"ROUGE-L: {rouge['rougeL']:.4f}")
```

## 🎓 Understanding the Code Flow

### Training Flow:

1. **Data Loading**: `create_t5_dataloader` → Creates (input, decoder_input, labels) triplets
2. **Encoder Forward**: Process input text → Encoder hidden states
3. **Decoder Forward**: Generate output text (autoregressive) with cross-attention to encoder
4. **Loss Calculation**: Cross-entropy for next-token prediction
5. **Backward Pass**: Compute gradients through both encoder and decoder

### Generation Flow:

1. **Encoder**: Process input text → Encoder hidden states
2. **Iterative Decoding**: For each step:
   - Decoder attends to encoder outputs (cross-attention)
   - Generate next token
   - Append to sequence
   - Repeat until EOS or max length

### Why Relative Position Embeddings?

Relative position embeddings learn relationships between tokens (e.g., "token A is 3 positions before token B") rather than absolute positions. This allows the model to generalize better to longer sequences and different contexts.

## 🔧 Configuration

Model configurations are defined in `config.py`:

- `vocab_size`: Vocabulary size (32,128 for T5)
- `d_model`: Model dimension (512 for T5-small)
- `num_encoder_layers`: Number of encoder layers (6 for T5-small)
- `num_decoder_layers`: Number of decoder layers (6 for T5-small)
- `num_heads`: Number of attention heads (8 for T5-small)
- `relative_attention_num_buckets`: Number of relative position buckets (32)

## 📚 Further Reading

- Original T5 Paper: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
- T5 GitHub: https://github.com/google-research/text-to-text-transfer-transformer

## 🚀 Quick Start

See `examples/train_t5.py` for a complete training example.

