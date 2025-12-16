# T5 Architecture Diagrams

## High-Level Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         T5 Model Flow                           │
└─────────────────────────────────────────────────────────────────┘

Input Text (with task prefix)
    │
    ▼
┌─────────────────┐
│ Tokenization    │  Convert text to token IDs
│ (SentencePiece) │  Example: "translate: Hello" → [token_ids]
└─────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    ENCODER STACK                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Input Embeddings                                      │  │
│  │ - Token Embeddings                                    │  │
│  │ - Relative Position Embeddings                        │  │
│  └──────────────────────────────────────────────────────┘  │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Encoder Blocks (N layers)                            │  │
│  │  ┌──────────────────────────────────────────────┐   │  │
│  │  │ Block 1:                                      │   │  │
│  │  │   ┌──────────────────────────────────────┐   │   │  │
│  │  │   │ Bidirectional Self-Attention         │   │   │  │
│  │  │   │ - Q, K, V projections                │   │   │  │
│  │  │   │ - Relative position bias              │   │   │  │
│  │  │   │ - NO causal mask (bidirectional)     │   │   │  │
│  │  │   │ - All tokens attend to all tokens    │   │   │  │
│  │  │   └──────────────────────────────────────┘   │   │  │
│  │  │         │                                     │   │  │
│  │  │         ▼                                     │   │  │
│  │  │   ┌──────────────┐                           │   │  │
│  │  │   │ LayerNorm +  │  Pre-norm + residual      │   │  │
│  │  │   │ Residual     │                           │   │  │
│  │  │   └──────────────┘                           │   │  │
│  │  │         │                                     │   │  │
│  │  │         ▼                                     │   │  │
│  │  │   ┌──────────────────────────────────────┐   │   │  │
│  │  │   │ Feed-Forward Network                 │   │   │  │
│  │  │   │ - Linear(d_model → d_ff)             │   │   │  │
│  │  │   │ - ReLU activation                     │   │   │  │
│  │  │   │ - Linear(d_ff → d_model)             │   │   │  │
│  │  │   └──────────────────────────────────────┘   │   │  │
│  │  │         │                                     │   │  │
│  │  │         ▼                                     │   │  │
│  │  │   ┌──────────────┐                           │   │  │
│  │  │   │ LayerNorm +  │  Pre-norm + residual      │   │  │
│  │  │   │ Residual     │                           │   │  │
│  │  │   └──────────────┘                           │   │  │
│  │  └──────────────────────────────────────────────┘   │  │
│  │                          ...                         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────┐
│ Encoder Output  │  Contextualized representations
│ Hidden States   │  Shape: (batch, seq_len, d_model)
└─────────────────┘
    │
    │ (passed to decoder via cross-attention)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    DECODER STACK                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Output Embeddings                                    │  │
│  │ - Token Embeddings (shared with encoder)             │  │
│  │ - Relative Position Embeddings                       │  │
│  └──────────────────────────────────────────────────────┘  │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Decoder Blocks (N layers)                            │  │
│  │  ┌──────────────────────────────────────────────┐   │  │
│  │  │ Block 1:                                      │   │  │
│  │  │   ┌──────────────────────────────────────┐   │   │  │
│  │  │   │ Causal Self-Attention                 │   │   │  │
│  │  │   │ - Q, K, V projections                 │   │   │  │
│  │  │   │ - Relative position bias              │   │   │  │
│  │  │   │ - Causal mask (can't see future)      │   │   │  │
│  │  │   └──────────────────────────────────────┘   │   │  │
│  │  │         │                                     │   │  │
│  │  │         ▼                                     │   │  │
│  │  │   ┌──────────────┐                           │   │  │
│  │  │   │ LayerNorm +  │  Pre-norm + residual      │   │  │
│  │  │   │ Residual     │                           │   │  │
│  │  │   └──────────────┘                           │   │  │
│  │  │         │                                     │   │  │
│  │  │         ▼                                     │   │  │
│  │  │   ┌──────────────────────────────────────┐   │   │  │
│  │  │   │ Cross-Attention                      │   │   │  │
│  │  │   │ - Q from decoder                     │   │   │  │
│  │  │   │ - K, V from encoder                  │   │   │  │
│  │  │   │ - Decoder attends to encoder outputs │   │   │  │
│  │  │   └──────────────────────────────────────┘   │   │  │
│  │  │         │                                     │   │  │
│  │  │         ▼                                     │   │  │
│  │  │   ┌──────────────┐                           │   │  │
│  │  │   │ LayerNorm +  │  Pre-norm + residual      │   │  │
│  │  │   │ Residual     │                           │   │  │
│  │  │   └──────────────┘                           │   │  │
│  │  │         │                                     │   │  │
│  │  │         ▼                                     │   │  │
│  │  │   ┌──────────────────────────────────────┐   │   │  │
│  │  │   │ Feed-Forward Network                 │   │   │  │
│  │  │   │ - Linear(d_model → d_ff)             │   │   │  │
│  │  │   │ - ReLU activation                     │   │   │  │
│  │  │   │ - Linear(d_ff → d_model)             │   │   │  │
│  │  │   └──────────────────────────────────────┘   │   │  │
│  │  │         │                                     │   │  │
│  │  │         ▼                                     │   │  │
│  │  │   ┌──────────────┐                           │   │  │
│  │  │   │ LayerNorm +  │  Pre-norm + residual      │   │  │
│  │  │   │ Residual     │                           │   │  │
│  │  │   └──────────────┘                           │   │  │
│  │  └──────────────────────────────────────────────┘   │  │
│  │                          ...                         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────┐
│ Output Head     │  Linear projection to vocabulary
│ (d_model →      │  Shape: (batch, seq_len, d_model) →
│  vocab_size)    │         (batch, seq_len, vocab_size)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Logits          │  Probability distribution over vocabulary
│ (batch, seq_len,│  for each output position
│  vocab_size)    │
└─────────────────┘
```

## Encoder-Decoder Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Encoder-Decoder Interaction                    │
└─────────────────────────────────────────────────────────────┘

ENCODER (Bidirectional):
────────────────────────
Input: "translate English to French: Hello"

[CLS] translate English to French : Hello [SEP]
  │      │        │       │    │    │    │
  └──────┴────────┴───────┴────┴────┴────┘
         │
         ▼
  Bidirectional Attention
  (all tokens see all tokens)
         │
         ▼
  Encoder Hidden States
  (contextualized representations)

DECODER (Autoregressive):
─────────────────────────
Step 1: Generate first token
  Input: [BOS] (beginning of sequence)
         │
         ▼
  Causal Self-Attention (only sees [BOS])
         │
         ▼
  Cross-Attention → Encoder Hidden States
         │
         ▼
  Output: "Bonjour"

Step 2: Generate second token
  Input: [BOS] Bonjour
         │
         ▼
  Causal Self-Attention (sees [BOS], Bonjour)
         │
         ▼
  Cross-Attention → Encoder Hidden States
         │
         ▼
  Output: [EOS] (end of sequence)

Final Output: "Bonjour"
```

## Relative Position Embeddings

```
┌─────────────────────────────────────────────────────────────┐
│          Relative Position Embeddings (T5)                  │
└─────────────────────────────────────────────────────────────┘

Unlike absolute position embeddings (GPT, BERT), T5 uses relative
position embeddings that encode the relationship between tokens.

Example Sequence: "The cat sat"

Absolute Positions (GPT/BERT):
  The: position 0
  cat: position 1
  sat: position 2

Relative Positions (T5):
  The → cat: relative position +1
  The → sat: relative position +2
  cat → The: relative position -1
  cat → sat: relative position +1
  sat → The: relative position -2
  sat → cat: relative position -1

Relative Position Buckets:
  ┌─────────────────────────────────────┐
  │ Distance │ Bucket                    │
  │──────────┼───────────────────────────│
  │ 0        │ Bucket 0 (same position)  │
  │ ±1       │ Bucket 1                  │
  │ ±2       │ Bucket 2                  │
  │ ±3       │ Bucket 3                  │
  │ ...      │ ...                       │
  │ ±N       │ Logarithmic bucketing     │
  └─────────────────────────────────────┘

Benefits:
- Generalizes better to longer sequences
- Learns relative relationships (e.g., "3 positions before")
- More flexible than absolute positions
```

## Cross-Attention Mechanism

```
┌─────────────────────────────────────────────────────────────┐
│                  Cross-Attention Detail                     │
└─────────────────────────────────────────────────────────────┘

Cross-attention allows the decoder to attend to encoder outputs.

ENCODER OUTPUTS:
────────────────
Hidden States: [h1, h2, h3, h4]
                │   │   │   │
                └───┴───┴───┘
                    │
                    ▼
DECODER CROSS-ATTENTION:
────────────────────────
Decoder Position: "Bonjour" (generating)

Query (Q): From decoder hidden state
  Q = Linear(decoder_hidden)

Key (K): From encoder hidden states
  K = Linear([h1, h2, h3, h4])

Value (V): From encoder hidden states
  V = Linear([h1, h2, h3, h4])

Attention Scores:
  scores = Q @ K^T / sqrt(head_dim)
  Shape: (1, 4) - one score per encoder position

Attention Weights:
  weights = softmax(scores)
  Example: [0.1, 0.2, 0.5, 0.2]
           (attends most to h3 - "Hello")

Context Vector:
  context = weights @ V
  Weighted combination of encoder hidden states

This allows decoder to:
- Focus on relevant parts of input
- Translate: attend to source words
- Summarize: attend to important sentences
- Answer: attend to relevant context
```

## Text-to-Text Framework

```
┌─────────────────────────────────────────────────────────────┐
│              Text-to-Text Task Formatting                   │
└─────────────────────────────────────────────────────────────┘

All tasks are framed as text generation:

TRANSLATION:
────────────
Input:  "translate English to French: Hello"
Output: "Bonjour"

SUMMARIZATION:
──────────────
Input:  "summarize: Long article text here..."
Output: "Summary of the article"

CLASSIFICATION:
───────────────
Input:  "classify: This movie is great!"
Output: "positive"

QUESTION ANSWERING:
───────────────────
Input:  "question: What is the capital of France? context: France is..."
Output: "Paris"

SENTIMENT ANALYSIS:
──────────────────
Input:  "sentiment: I love this product!"
Output: "positive"

The task prefix ("translate:", "summarize:", etc.) tells the model
what task to perform. The model learns to interpret these prefixes
during training.
```

## Training Flow (Span Corruption)

```
┌─────────────────────────────────────────────────────────────┐
│                    T5 Pretraining Flow                      │
└─────────────────────────────────────────────────────────────┘

Original Text: "Thank you for inviting me to your party last week."
    │
    ▼
┌─────────────────────────────────────────┐
│ Span Corruption (T5's pretraining task) │
│                                         │
│ Randomly select spans of text to mask: │
│ "Thank you <X> me to <Y> last week."   │
│                                         │
│ <X> = "for inviting"                    │
│ <Y> = "your party"                      │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Create Input/Output Pair                │
│                                         │
│ Input:  "Thank you <X> me to <Y> last week."│
│ Output: "for inviting your party"       │
│                                         │
│ Task: Reconstruct corrupted spans       │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Encoder Processes Input                 │
│ - Sees corrupted text                  │
│ - Creates contextualized representations│
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Decoder Generates Output                │
│ - Autoregressively generates spans      │
│ - Uses cross-attention to encoder       │
│ - Learns to reconstruct masked text     │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Compute Loss                            │
│ CrossEntropy(predicted_spans, true_spans)│
│                                         │
│ Model learns to:                        │
│ - Understand context                    │
│ - Generate coherent text                │
│ - Handle various tasks                  │
└─────────────────────────────────────────┘
```

## Generation Flow (Autoregressive)

```
┌─────────────────────────────────────────────────────────────┐
│              T5 Autoregressive Generation                   │
└─────────────────────────────────────────────────────────────┘

Input: "translate English to French: Hello"
    │
    ▼
┌─────────────────────────────────────────┐
│ ENCODER: Process Input                  │
│ - Tokenize: [translate, :, Hello]       │
│ - Pass through encoder blocks           │
│ - Create encoder hidden states           │
│ - Shape: (batch, seq_len, d_model)      │
└─────────────────────────────────────────┘
    │
    │ (encoder states available for all decoder steps)
    │
    ▼
┌─────────────────────────────────────────┐
│ DECODER: Generate Output Token by Token│
│                                         │
│ Step 1:                                 │
│   Decoder Input: [BOS]                  │
│   ├─→ Causal Self-Attention             │
│   ├─→ Cross-Attention → Encoder States  │
│   ├─→ FFN                               │
│   └─→ Logits → Sample → "Bonjour"      │
│                                         │
│ Step 2:                                 │
│   Decoder Input: [BOS] Bonjour          │
│   ├─→ Causal Self-Attention             │
│   ├─→ Cross-Attention → Encoder States  │
│   ├─→ FFN                               │
│   └─→ Logits → Sample → [EOS]          │
│                                         │
│ Stop: [EOS] token generated             │
└─────────────────────────────────────────┘
    │
    ▼
Output: "Bonjour"
```

## Key Differences from Other Architectures

```
┌─────────────────────────────────────────────────────────────┐
│         T5 vs GPT vs BERT Architecture Comparison           │
└─────────────────────────────────────────────────────────────┘

ENCODER-DECODER (T5):
─────────────────────
Input → Encoder → Encoder States → Decoder → Output
         │                          │
         └──────────────────────────┘
              (cross-attention)

- Encoder: Bidirectional (sees all input)
- Decoder: Causal (autoregressive generation)
- Cross-attention: Decoder attends to encoder
- Use case: Text-to-text tasks


DECODER-ONLY (GPT):
───────────────────
Input → Decoder → Output
         │
         └─ (causal self-attention only)

- Single stack: Decoder only
- Causal attention: Can't see future
- No cross-attention
- Use case: Text generation


ENCODER-ONLY (BERT):
────────────────────
Input → Encoder → Hidden States
         │
         └─ (bidirectional self-attention)

- Single stack: Encoder only
- Bidirectional attention: Sees all tokens
- No generation capability
- Use case: Understanding tasks
```

## Shared Embeddings

```
┌─────────────────────────────────────────────────────────────┐
│                  Shared Embedding Weights                   │
└─────────────────────────────────────────────────────────────┘

T5 uses weight tying between input/output embeddings and output head:

┌─────────────────┐
│ Input Embedding │  ─┐
│ (vocab × d_model)│   │
└─────────────────┘   │
                       │ Shared Weights
┌─────────────────┐   │
│ Output Embedding│  ─┤
│ (vocab × d_model)│   │
└─────────────────┘   │
                       │
┌─────────────────┐   │
│ Output Head     │  ─┘
│ (d_model × vocab)│
└─────────────────┘

Benefits:
- Reduces parameters
- Improves training efficiency
- Better generalization
- Standard in T5 architecture
```

