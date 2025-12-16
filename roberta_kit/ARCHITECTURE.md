# RoBERTa Architecture Diagrams

## High-Level Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                       RoBERTa Model Flow                       │
└─────────────────────────────────────────────────────────────────┘

Input Text
    │
    ▼
┌─────────────────┐
│ Tokenization    │  Convert text to token IDs using BPE
│ (BPE)           │  Example: "Hello world" → [token_ids]
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Token           │  Embed each token ID into dense vector
│ Embedding       │  Shape: (batch, seq_len) → (batch, seq_len, hidden_size)
│ (vocab_size ×   │
│  hidden_size)   │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Position        │  Learned positional encodings
│ Embedding       │  Shape: (seq_len) → (seq_len, hidden_size)
│ (learned)       │
│                 │
│ NOTE: No Segment│  Unlike BERT, RoBERTa doesn't use segment embeddings
│ Embeddings      │  (removed to simplify architecture)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Embedding       │  token_emb + pos_emb (no segment_emb)
│ Sum + LayerNorm │  Shape: (batch, seq_len, hidden_size)
│ + Dropout       │  LayerNorm epsilon: 1e-5 (vs 1e-12 in BERT)
└─────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│              Transformer Encoder Blocks (N layers)           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Block 1:                                              │  │
│  │   ┌──────────────────────────────────────┐           │  │
│  │   │ Bidirectional Multi-Head Attention  │           │  │
│  │   │ - Q, K, V projections                │           │  │
│  │   │ - NO causal mask (bidirectional)     │           │  │
│  │   │ - All tokens attend to all tokens    │           │  │
│  │   │ - Same as BERT                       │           │  │
│  │   └──────────────────────────────────────┘           │  │
│  │         │                                             │  │
│  │         ▼                                             │  │
│  │   ┌──────────────┐                                    │  │
│  │   │ LayerNorm +  │  Post-norm: normalize after        │  │
│  │   │ Residual     │  Residual: x + attn(x)             │  │
│  │   │ (eps=1e-5)   │  Note: Different epsilon than BERT │  │
│  │   └──────────────┘                                    │  │
│  │         │                                             │  │
│  │         ▼                                             │  │
│  │   ┌──────────────────────────────────────┐           │  │
│  │   │ Feed-Forward Network                 │           │  │
│  │   │ - Linear(hidden_size → intermediate) │           │  │
│  │   │ - GELU activation                    │           │  │
│  │   │ - Linear(intermediate → hidden_size) │           │  │
│  │   │ - Dropout                            │           │  │
│  │   └──────────────────────────────────────┘           │  │
│  │         │                                             │  │
│  │         ▼                                             │  │
│  │   ┌──────────────┐                                    │  │
│  │   │ LayerNorm +  │  Post-norm: normalize after        │  │
│  │   │ Residual     │  Residual: x + ffn(x)              │  │
│  │   │ (eps=1e-5)   │                                     │  │
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
│ Hidden States   │  Contextualized representations
│ (batch, seq_len,│  Each token has context from entire sequence
│  hidden_size)   │
└─────────────────┘
    │
    ├─────────────────────────────────────┐
    │                                     │
    ▼                                     ▼
┌─────────────────┐            ┌─────────────────┐
│ [CLS] Token     │            │ All Tokens      │
│ (for            │            │ (for token      │
│ classification) │            │ classification) │
└─────────────────┘            └─────────────────┘
```

## Dynamic Masking Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                  Dynamic Masking (RoBERTa)                   │
└─────────────────────────────────────────────────────────────┘

RoBERTa uses DYNAMIC masking - the masking pattern changes each epoch.

Original Text: "The cat sat on the mat"

EPOCH 1:
─────────
Masking Pattern: [The, [MASK], sat, on, the, mat]
                 (cat is masked)

EPOCH 2:
─────────
Same Text, Different Pattern: [The, cat, [MASK], on, the, mat]
                               (sat is masked)

EPOCH 3:
─────────
Same Text, Different Pattern: [[MASK], cat, sat, on, the, mat]
                               (The is masked)

Benefits:
- Model sees different masking patterns for same text
- Prevents overfitting to specific masking patterns
- Forces model to learn robust representations
- Better generalization

Comparison with BERT:
- BERT: Static masking (same pattern every epoch)
- RoBERTa: Dynamic masking (pattern changes each epoch)
```

## MLM Training Flow

```
┌─────────────────────────────────────────────────────────────┐
│              RoBERTa MLM Pretraining Flow                   │
└─────────────────────────────────────────────────────────────┘

Original Text: "The cat sat on the mat"
    │
    ▼
┌─────────────────────────────────────────┐
│ Tokenize                                │
│ [CLS] The cat sat on the mat [SEP]      │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Dynamic Masking (15% of tokens)         │
│ Pattern changes each epoch              │
│                                         │
│ Example (Epoch 1):                      │
│ [CLS] The [MASK] sat on the mat [SEP]   │
│         (cat is masked)                 │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Forward Pass                            │
│ model(masked_input) → logits            │
│ Shape: (batch, seq_len, vocab_size)     │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Compute MLM Loss                        │
│ Only for masked positions               │
│ Target: "cat" (original token)          │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Key Difference from BERT:               │
│ - No NSP task (removed)                 │
│ - Focus solely on MLM                  │
│ - Dynamic masking (vs static)           │
│ - More training data                    │
│ - Longer training                       │
└─────────────────────────────────────────┘
```

## Differences from BERT

```
┌─────────────────────────────────────────────────────────────┐
│              RoBERTa vs BERT Architecture                  │
└─────────────────────────────────────────────────────────────┘

EMBEDDING LAYER:
────────────────

BERT:
  Token Embeddings
  + Segment Embeddings  ← Used for sentence pairs
  + Position Embeddings
  = Combined Embeddings

RoBERTa:
  Token Embeddings
  + Position Embeddings
  = Combined Embeddings
  (No Segment Embeddings - removed)

LAYER NORMALIZATION:
────────────────────

BERT:
  LayerNorm epsilon = 1e-12

RoBERTa:
  LayerNorm epsilon = 1e-5
  (More standard value)

MASKING:
────────

BERT:
  Static masking
  - Same pattern every epoch
  - Fixed during dataset creation

RoBERTa:
  Dynamic masking
  - Pattern changes each epoch
  - Regenerated on-the-fly

PRE-TRAINING TASKS:
───────────────────

BERT:
  - Masked Language Modeling (MLM)
  - Next Sentence Prediction (NSP)  ← Removed in RoBERTa

RoBERTa:
  - Masked Language Modeling (MLM) only
  - No NSP task

TRAINING:
─────────

BERT:
  - Smaller batch size
  - Less training data
  - Fewer training steps

RoBERTa:
  - Larger batch size (8K vs 256)
  - More training data
  - More training steps (500K vs 1M)
```

## Classification Flow

```
┌─────────────────────────────────────────────────────────────┐
│              RoBERTa Classification Flow                    │
└─────────────────────────────────────────────────────────────┘

Input Text: "Great movie!"
    │
    ▼
┌─────────────────────────────────────────┐
│ Tokenize & Format                       │
│ [CLS] Great movie! [SEP]                │
│                                         │
│ Token IDs: [0, 2307, 3185, 999, 2]     │
│ Attention Mask: [1, 1, 1, 1, 1]        │
│                                         │
│ Note: No segment IDs needed             │
│ (RoBERTa doesn't use segment embeddings)│
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Forward Pass Through RoBERTa            │
│ model(input_ids, attention_mask)         │
│ → hidden_states                         │
│ Shape: (batch, seq_len, hidden_size)    │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Extract [CLS] Token                     │
│ cls_hidden = hidden_states[:, 0, :]     │
│ Shape: (batch, hidden_size)             │
│                                         │
│ [CLS] token aggregates information      │
│ from entire sequence                    │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Apply Dropout                           │
│ cls_hidden = dropout(cls_hidden)        │
│ (regularization)                        │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Classification Head                     │
│ Linear(hidden_size → num_labels)        │
│ logits = classifier(cls_hidden)         │
│ Shape: (batch, num_labels)              │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Prediction                              │
│ pred = argmax(logits)                   │
│ Example: 1 (positive sentiment)         │
└─────────────────────────────────────────┘
```

## Why Remove NSP?

```
┌─────────────────────────────────────────────────────────────┐
│              Why RoBERTa Removed NSP Task                   │
└─────────────────────────────────────────────────────────────┘

BERT's NSP Task:
────────────────
Input: Sentence A + Sentence B
Task: Predict if B follows A

Problems Found:
───────────────
1. NSP task too easy
   - Model learns to distinguish sentence pairs vs single sentences
   - Not learning meaningful relationships

2. NSP hurts performance
   - Removing NSP improves results on downstream tasks
   - MLM alone is sufficient

3. NSP adds complexity
   - Requires sentence pair data
   - Segment embeddings needed
   - More preprocessing

RoBERTa Solution:
─────────────────
- Remove NSP task entirely
- Focus solely on MLM
- Use longer sequences (single sentences or concatenated)
- Simpler architecture (no segment embeddings)
- Better performance

Result:
───────
RoBERTa outperforms BERT on most tasks without NSP!
```

## Training Improvements

```
┌─────────────────────────────────────────────────────────────┐
│              RoBERTa Training Optimizations                 │
└─────────────────────────────────────────────────────────────┘

BERT Training:
──────────────
Batch Size: 256
Training Steps: 1M
Data: BookCorpus + English Wikipedia
Masking: Static

RoBERTa Training:
─────────────────
Batch Size: 8,000 (32x larger!)
Training Steps: 500K (but more effective)
Data: BookCorpus + CC-News + OpenWebText + Stories
      (10x more data!)
Masking: Dynamic

Key Improvements:
─────────────────
1. Larger batches → more stable gradients
2. More data → better generalization
3. Dynamic masking → more robust representations
4. No NSP → simpler, more focused training
5. Longer sequences → better context understanding

Result: Better performance with same architecture!
```

