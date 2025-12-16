# GPT Kit

A Python package for building, training, fine-tuning, and evaluating GPT-style language models.

## 🎯 What Makes GPT Unique?

GPT (Generative Pre-trained Transformer) is a **decoder-only** autoregressive language model designed for **text generation** tasks.

### Key Characteristics:

1. **Autoregressive Generation**: Generates text one token at a time, using previously generated tokens as context
2. **Causal Attention**: Uses masked self-attention to prevent the model from seeing future tokens during training
3. **Unidirectional Context**: Processes text from left to right, making it ideal for generation tasks
4. **No Encoder**: Unlike BERT/T5, GPT doesn't have a separate encoder - it's a pure decoder architecture

### Best Use Cases:

- **Text Generation**: Creative writing, story continuation, code generation
- **Completion Tasks**: Sentence completion, text infilling
- **Instruction Following**: Fine-tuning on instruction-response pairs
- **Conversational AI**: Chatbots and dialogue systems

## 🏗️ Model Architecture

### High-Level Structure:

```
Input Tokens → Token Embeddings → Position Embeddings → Transformer Blocks → Output Logits
```

### Core Components:

1. **Token Embeddings**: Convert token IDs to dense vectors
2. **Position Embeddings**: Add positional information (cached for efficiency)
3. **Transformer Blocks** (N layers):
   - **Causal Multi-Head Attention**: Masked self-attention (prevents seeing future tokens)
   - **Feed-Forward Network**: Two linear layers with GELU activation
   - **Layer Normalization**: Applied before attention and FFN (pre-norm architecture)
   - **Residual Connections**: Skip connections around attention and FFN
4. **Output Layer**: Linear projection to vocabulary size

### Architecture Details:

- **Attention Mask**: Lower triangular mask ensures causal attention
- **Position Embeddings**: Cached position IDs for efficiency (not learned)
- **Layer Normalization**: Uses epsilon=1e-5, applied before sub-layers
- **Activation**: GELU (Gaussian Error Linear Unit)
- **Dropout**: Applied to embeddings, attention, and FFN

## 📁 Code Structure

### `model.py` - Model Architecture

**Key Classes:**

- `LayerNorm`: Layer normalization with configurable epsilon
- `GELU`: Gaussian Error Linear Unit activation
- `FeedForward`: Two-layer MLP with GELU and dropout
- `MultiHeadAttention`: Causal (masked) multi-head self-attention
- `TransformerBlock`: Single transformer layer (attention + FFN)
- `GPTModel`: Complete GPT model

**How It Works:**

1. **Input Processing**: Token IDs → Embeddings → Add Position Embeddings
2. **Transformer Layers**: Each layer applies causal attention then FFN
3. **Output**: Final hidden states → Logits (vocab_size predictions per position)

### `data.py` - Data Loading

**Key Classes:**

- `GPTPretrainingDataset`: Sliding window dataset for pretraining
- `InstructionDataset`: Dataset for instruction fine-tuning

**How It Works:**

- **Pretraining**: Creates overlapping sequences using sliding window approach
- **Instruction Fine-tuning**: Formats instruction-input-output triplets
- **Tokenization**: Uses `tiktoken` (GPT-2 tokenizer)

### `training.py` - Training Utilities

**Key Functions:**

- `compute_batch_loss`: Calculate cross-entropy loss for a batch
- `train_step_with_gradient_clipping`: Training step with gradient clipping
- `compute_loader_loss`: Evaluate average loss on a dataloader

**How It Works:**

- Uses standard cross-entropy loss for next-token prediction
- Supports gradient clipping for training stability
- Handles padding tokens via `ignore_index`

### `evaluation.py` - Text Generation

**Key Functions:**

- `generate_text_autoregressive`: Generate text using autoregressive sampling
- `text_to_token_ids`: Convert text to token IDs
- `token_ids_to_text`: Convert token IDs back to text

**How It Works:**

- **Autoregressive Sampling**: Generates one token at a time
- **Temperature**: Controls randomness (higher = more creative)
- **Top-k Sampling**: Limits sampling to top-k most likely tokens
- **EOS Handling**: Stops generation when end-of-sequence token is generated

### `amp.py` - Mixed Precision Training

**Key Classes:**

- `AMPTrainer`: Automatic Mixed Precision trainer

**How It Works:**

- Uses `torch.cuda.amp.autocast` for FP16/BF16 training
- `GradScaler` prevents gradient underflow
- Automatically handles loss scaling and gradient unscaling

## 🔑 Key Differences from Other Kits

| Feature | GPT | BERT | T5 |
|---------|-----|------|-----|
| Architecture | Decoder-only | Encoder-only | Encoder-Decoder |
| Attention | Causal (masked) | Bidirectional | Causal (decoder) |
| Task | Generation | Understanding | Text-to-text |
| Direction | Left-to-right | Bidirectional | Bidirectional (encoder) |
| Use Case | Text generation | Classification/MLM | Translation/Summarization |

## 📝 Usage Examples

### Basic Model Creation

```python
from gpt_kit import GPTModel, create_model_config

config = create_model_config("gpt2-small (124M)")
model = GPTModel(config)
```

### Pretraining

```python
from gpt_kit import create_pretraining_dataloader

text = "Your training text here..."
train_loader = create_pretraining_dataloader(
    text=text, batch_size=4, max_length=256, stride=128
)
```

### Text Generation

```python
from gpt_kit import generate_text_autoregressive, text_to_token_ids
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
prompt_ids = text_to_token_ids("The future of AI is", tokenizer)

generated_ids = generate_text_autoregressive(
    model=model,
    token_ids=prompt_ids,
    max_new_tokens=50,
    temperature=0.8,
    top_k=50
)
```

### Instruction Fine-tuning

```python
from gpt_kit import InstructionDataset, create_instruction_collate_fn
from torch.utils.data import DataLoader

instruction_data = [
    {"instruction": "Translate", "input": "Hello", "output": "Bonjour"}
]

dataset = InstructionDataset(instruction_data, tokenizer, format_fn)
collate_fn = create_instruction_collate_fn(pad_token_id=50256)
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
```

## 🎓 Understanding the Code Flow

### Training Flow:

1. **Data Loading**: `create_pretraining_dataloader` → Creates batches of (input, target) pairs
2. **Forward Pass**: `model(input_batch)` → Returns logits of shape `(batch, seq_len, vocab_size)`
3. **Loss Calculation**: Cross-entropy between logits and targets
4. **Backward Pass**: Compute gradients
5. **Optimization**: Update weights via optimizer

### Generation Flow:

1. **Initialization**: Start with prompt token IDs
2. **Iterative Generation**: For each step:
   - Forward pass through model
   - Sample next token from logits
   - Append to sequence
   - Repeat until EOS or max length
3. **Decoding**: Convert token IDs back to text

## 🔧 Configuration

Model configurations are defined in `config.py`:

- `vocab_size`: Vocabulary size (50,257 for GPT-2)
- `n_embd`: Embedding dimension (768 for GPT-2 small)
- `n_layer`: Number of transformer layers (12 for GPT-2 small)
- `n_head`: Number of attention heads (12 for GPT-2 small)
- `max_position_embeddings`: Maximum sequence length (1024 for GPT-2)

## 📚 Further Reading

- Original GPT Paper: "Improving Language Understanding by Generative Pre-Training"
- GPT-2 Paper: "Language Models are Unsupervised Multitask Learners"
- GPT-3 Paper: "Language Models are Few-Shot Learners"

## 🚀 Quick Start

See `examples/train_gpt.py` for a complete training example.

