# GPT Architecture Diagrams

## High-Level Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         GPT Model Flow                           │
└─────────────────────────────────────────────────────────────────┘

Input Text
    │
    ▼
┌─────────────────┐
│ Tokenization    │  Convert text to token IDs using tiktoken
│ (tiktoken)      │  Example: "Hello" → [15496]
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Token           │  Embed each token ID into dense vector
│ Embedding       │  Shape: (batch, seq_len) → (batch, seq_len, emb_dim)
│ (vocab_size ×   │
│  emb_dim)       │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Position        │  Add positional information (cached position IDs)
│ Embedding       │  Shape: (seq_len) → (seq_len, emb_dim)
│ (cached)        │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Embedding       │  token_emb + pos_emb → (batch, seq_len, emb_dim)
│ Sum + Dropout   │
└─────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│              Transformer Blocks (N layers)                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Block 1:                                              │  │
│  │   ┌──────────────┐                                    │  │
│  │   │ LayerNorm    │  Pre-norm: normalize before       │  │
│  │   └──────────────┘                                    │  │
│  │         │                                             │  │
│  │         ▼                                             │  │
│  │   ┌──────────────────────────────────────┐           │  │
│  │   │ Causal Multi-Head Attention          │           │  │
│  │   │ - Q, K, V projections                │           │  │
│  │   │ - Split into heads                   │           │  │
│  │   │ - Causal mask (lower triangular)     │           │  │
│  │   │ - Softmax → attention weights        │           │  │
│  │   │ - Apply to values                    │           │  │
│  │   └──────────────────────────────────────┘           │  │
│  │         │                                             │  │
│  │         ▼                                             │  │
│  │   ┌──────────────┐                                    │  │
│  │   │ Dropout +    │  Residual connection: x + attn(x) │  │
│  │   │ Residual     │                                    │  │
│  │   └──────────────┘                                    │  │
│  │         │                                             │  │
│  │         ▼                                             │  │
│  │   ┌──────────────┐                                    │  │
│  │   │ LayerNorm    │  Pre-norm: normalize before       │  │
│  │   └──────────────┘                                    │  │
│  │         │                                             │  │
│  │         ▼                                             │  │
│  │   ┌──────────────────────────────────────┐           │  │
│  │   │ Feed-Forward Network                 │           │  │
│  │   │ - Linear(emb_dim → 4×emb_dim)        │           │  │
│  │   │ - GELU activation                    │           │  │
│  │   │ - Linear(4×emb_dim → emb_dim)        │           │  │
│  │   └──────────────────────────────────────┘           │  │
│  │         │                                             │  │
│  │         ▼                                             │  │
│  │   ┌──────────────┐                                    │  │
│  │   │ Dropout +    │  Residual connection: x + ffn(x)   │  │
│  │   │ Residual     │                                    │  │
│  │   └──────────────┘                                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ...                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Block N: (same structure as Block 1)                 │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────┐
│ Final LayerNorm │  Normalize final hidden states
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Output Head     │  Linear projection to vocabulary
│ (emb_dim →      │  Shape: (batch, seq_len, emb_dim) →
│  vocab_size)    │         (batch, seq_len, vocab_size)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Logits          │  Probability distribution over vocabulary
│ (batch, seq_len,│  for each position
│  vocab_size)    │
└─────────────────┘
```

## Causal Attention Mechanism

```
┌─────────────────────────────────────────────────────────────┐
│              Causal Attention Visualization                 │
└─────────────────────────────────────────────────────────────┘

Sequence: ["The", "cat", "sat", "on", "mat"]

Attention Matrix (what each token can see):

        The   cat   sat   on   mat
The      ✓     ✗    ✗    ✗    ✗
cat      ✓     ✓    ✗    ✗    ✗
sat      ✓     ✓    ✓    ✗    ✗
on       ✓     ✓    ✓    ✓    ✗
mat      ✓     ✓    ✓    ✓    ✓

Lower Triangular Mask:
┌─────────────────────────┐
│  1   0   0   0   0  │  The can only see itself
│  1   1   0   0   0  │  cat can see The and itself
│  1   1   1   0   0  │  sat can see The, cat, itself
│  1   1   1   1   0  │  on can see The, cat, sat, itself
│  1   1   1   1   1  │  mat can see all previous tokens
└─────────────────────────┘

This ensures autoregressive generation - each token only
depends on previous tokens, not future ones.
```

## Multi-Head Attention Detail

```
┌─────────────────────────────────────────────────────────────┐
│           Multi-Head Attention Internal Flow                 │
└─────────────────────────────────────────────────────────────┘

Input: (batch, seq_len, emb_dim)
    │
    ▼
┌─────────────────────────────────────────┐
│ QKV Projection                          │
│ Linear(emb_dim → 3 × emb_dim)           │
└─────────────────────────────────────────┘
    │
    ▼ Split into Q, K, V
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Query    │  │ Key      │  │ Value    │
│ (Q)      │  │ (K)       │  │ (V)      │
└──────────┘  └──────────┘  └──────────┘
    │              │              │
    ▼              ▼              ▼
┌─────────────────────────────────────────┐
│ Reshape for Multi-Head                  │
│ (batch, seq_len, emb_dim) →            │
│ (batch, num_heads, seq_len, head_dim)   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Compute Attention Scores                │
│ scores = Q @ K^T / sqrt(head_dim)       │
│ Shape: (batch, heads, seq_len, seq_len)│
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Apply Causal Mask                       │
│ Set upper triangle to -inf              │
│ (prevents seeing future tokens)         │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Softmax                                 │
│ Convert scores to probabilities         │
│ Each row sums to 1                      │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Weighted Sum of Values                  │
│ context = attention_weights @ V        │
│ Shape: (batch, heads, seq_len, head_dim)│
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Concatenate Heads                       │
│ (batch, heads, seq_len, head_dim) →    │
│ (batch, seq_len, emb_dim)               │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Output Projection                       │
│ Linear(emb_dim → emb_dim)               │
└─────────────────────────────────────────┘
    │
    ▼
Output: (batch, seq_len, emb_dim)
```

## Training Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    GPT Training Flow                         │
└─────────────────────────────────────────────────────────────┘

Training Data (Text)
    │
    ▼
┌─────────────────────────────────────────┐
│ Sliding Window Dataset                  │
│ - Split text into overlapping windows  │
│ - Example: "The cat sat on the mat"    │
│   Window 1: "The cat sat"               │
│   Window 2: "cat sat on"                │
│   Window 3: "sat on the"                │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Create Input/Target Pairs               │
│ Input:  [The, cat, sat]                 │
│ Target: [cat, sat, on]                  │
│ (shifted by 1 position)                 │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Forward Pass                            │
│ model(input) → logits                   │
│ Shape: (batch, seq_len, vocab_size)     │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Compute Loss                            │
│ CrossEntropy(logits, targets)           │
│ - Reshape: (batch*seq_len, vocab_size)  │
│ - Reshape targets: (batch*seq_len,)     │
│ - Compute per-token loss                │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Backward Pass                           │
│ loss.backward()                         │
│ - Compute gradients                    │
│ - Store in .grad attributes            │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Gradient Clipping (optional)            │
│ clip_grad_norm_(max_norm=1.0)          │
│ - Prevents exploding gradients          │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Optimizer Step                          │
│ optimizer.step()                        │
│ - Update model parameters              │
│ - Using AdamW optimizer                 │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Zero Gradients                          │
│ optimizer.zero_grad()                   │
│ - Clear gradients for next iteration   │
└─────────────────────────────────────────┘
```

## Text Generation Flow

```
┌─────────────────────────────────────────────────────────────┐
│                  Autoregressive Generation                   │
└─────────────────────────────────────────────────────────────┘

Prompt: "The cat"
    │
    ▼
┌─────────────────────────────────────────┐
│ Tokenize Prompt                         │
│ "The cat" → [464, 2361]                 │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Forward Pass                            │
│ model([464, 2361]) → logits             │
│ Shape: (1, 2, vocab_size)               │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Extract Last Position Logits            │
│ logits[:, -1, :] → (vocab_size,)        │
│ (logits for next token prediction)       │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Apply Temperature (optional)            │
│ logits = logits / temperature           │
│ - temperature > 1: more random          │
│ - temperature < 1: more deterministic   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Top-k Sampling (optional)               │
│ Keep only top-k logits, mask others    │
│ - Reduces randomness                    │
│ - Focuses on likely tokens              │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Softmax                                 │
│ Convert to probability distribution     │
│ probs = softmax(logits)                 │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Sample Token                            │
│ next_token = sample(probs)              │
│ Example: 2361 → "sat"                    │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Append to Sequence                      │
│ [464, 2361] + [2361] → [464, 2361, 2361]│
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Check Stopping Condition                │
│ - EOS token reached?                    │
│ - Max length reached?                   │
│ If not, repeat from Forward Pass        │
└─────────────────────────────────────────┘
    │
    ▼
Generated: "The cat sat on the mat"
```

## Pre-norm vs Post-norm Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Pre-norm (GPT) vs Post-norm (BERT)             │
└─────────────────────────────────────────────────────────────┘

PRE-NORM (GPT-style):
─────────────────────
Input
  │
  ├─→ LayerNorm ──→ Attention ──→ Dropout ──┐
  │                                           │
  └───────────────────────────────────────────┴─→ Output
         (residual connection)

  ├─→ LayerNorm ──→ FFN ──→ Dropout ──┐
  │                                    │
  └────────────────────────────────────┴─→ Output
         (residual connection)

Benefits:
- Better gradient flow
- More stable training
- Allows deeper networks


POST-NORM (BERT-style):
───────────────────────
Input
  │
  ├─→ Attention ──→ LayerNorm ──┐
  │                             │
  └─────────────────────────────┴─→ Output
         (residual connection)

  ├─→ FFN ──→ LayerNorm ──┐
  │                        │
  └────────────────────────┴─→ Output
         (residual connection)

Benefits:
- Simpler structure
- Standard in BERT
- Works well for encoder models
```

