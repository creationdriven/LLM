# BERT Architecture Diagrams

## High-Level Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         BERT Model Flow                         │
└─────────────────────────────────────────────────────────────────┘

Input Text(s)
    │
    ▼
┌─────────────────┐
│ Tokenization    │  Convert text to token IDs using WordPiece
│ (WordPiece)     │  Add special tokens: [CLS] at start, [SEP] between sentences
│                 │  Example: "Hello world" → [CLS] Hello world [SEP]
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
│ Segment         │  Distinguish sentence A from sentence B
│ Embedding       │  Shape: (batch, seq_len) → (batch, seq_len, hidden_size)
│ (type_vocab_size│  [0, 0, 0, 1, 1, 1] for two sentences
│  × hidden_size) │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Position        │  Learned positional encodings
│ Embedding       │  Shape: (seq_len) → (seq_len, hidden_size)
│ (learned)       │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Embedding       │  token_emb + segment_emb + pos_emb
│ Sum + LayerNorm │  Shape: (batch, seq_len, hidden_size)
│ + Dropout       │
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
│  │   │ - Softmax → attention weights        │           │  │
│  │   │ - Apply to values                    │           │  │
│  │   └──────────────────────────────────────┘           │  │
│  │         │                                             │  │
│  │         ▼                                             │  │
│  │   ┌──────────────┐                                    │  │
│  │   │ LayerNorm +  │  Post-norm: normalize after        │  │
│  │   │ Residual     │  Residual: x + attn(x)             │  │
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
│ (batch, seq_len,│  Each token now has context from entire sequence
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

## Bidirectional Attention Mechanism

```
┌─────────────────────────────────────────────────────────────┐
│          Bidirectional Attention Visualization              │
└─────────────────────────────────────────────────────────────┘

Sequence: ["[CLS]", "The", "cat", "sat", "[SEP]"]

Attention Matrix (what each token can see):

        [CLS]  The   cat   sat   [SEP]
[CLS]     ✓     ✓     ✓     ✓     ✓
The       ✓     ✓     ✓     ✓     ✓
cat       ✓     ✓     ✓     ✓     ✓
sat       ✓     ✓     ✓     ✓     ✓
[SEP]     ✓     ✓     ✓     ✓     ✓

Full Bidirectional Mask:
┌─────────────────────────┐
│  1   1   1   1   1  │  All tokens can attend to all tokens
│  1   1   1   1   1  │  This allows understanding full context
│  1   1   1   1   1  │  Each token sees entire sequence
│  1   1   1   1   1  │
│  1   1   1   1   1  │
└─────────────────────────┘

This enables BERT to understand relationships between
all words in the sequence, not just previous words.
```

## Masked Language Modeling (MLM) Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    MLM Pretraining Flow                      │
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
│ Randomly Mask 15% of Tokens              │
│ Strategy:                                │
│ - 80% → [MASK] token                    │
│ - 10% → Random token                    │
│ - 10% → Unchanged                       │
│                                         │
│ Example:                                │
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
│ Extract Logits for Masked Positions     │
│ Only compute loss for [MASK] positions  │
│ Ignore non-masked tokens (ignore_index)│
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Compute MLM Loss                         │
│ CrossEntropy(logits[masked_pos],        │
│              true_tokens[masked_pos])   │
│                                         │
│ Target: "cat" (original token)         │
│ Prediction: probability over vocab      │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Backward Pass & Update                  │
│ Learn to predict masked tokens using    │
│ bidirectional context                   │
└─────────────────────────────────────────┘
```

## Classification Task Flow

```
┌─────────────────────────────────────────────────────────────┐
│                  Classification Fine-tuning                  │
└─────────────────────────────────────────────────────────────┘

Input Text: "Great movie!"
    │
    ▼
┌─────────────────────────────────────────┐
│ Tokenize & Format                       │
│ [CLS] Great movie! [SEP]                │
│                                         │
│ Token IDs: [101, 2307, 3185, 999, 102] │
│ Attention Mask: [1, 1, 1, 1, 1]        │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Forward Pass Through BERT               │
│ model(input_ids, attention_mask)        │
│ → hidden_states                         │
│ Shape: (batch, seq_len, hidden_size)   │
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
│ Softmax (for probabilities)             │
│ probs = softmax(logits)                 │
│ Example: [0.1, 0.9] → positive sentiment│
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Prediction                              │
│ pred = argmax(logits)                   │
│ Example: 1 (positive)                   │
└─────────────────────────────────────────┘
```

## Next Sentence Prediction (NSP) Flow

```
┌─────────────────────────────────────────────────────────────┐
│              Next Sentence Prediction (NSP)                │
└─────────────────────────────────────────────────────────────┘

Sentence Pair:
  Sentence A: "The cat sat"
  Sentence B: "It was happy"
    │
    ▼
┌─────────────────────────────────────────┐
│ Format with Special Tokens              │
│ [CLS] The cat sat [SEP] It was happy [SEP]│
│                                         │
│ Segment IDs:                            │
│ [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]         │
│ (0 = sentence A, 1 = sentence B)        │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Forward Pass                            │
│ model(input_ids, segment_ids)           │
│ → hidden_states                         │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Extract [CLS] Token                     │
│ cls_hidden = hidden_states[:, 0, :]     │
│ (aggregates info from both sentences)   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ NSP Head                                │
│ Linear(hidden_size → 2)                 │
│ logits = nsp_head(cls_hidden)           │
│                                         │
│ Output:                                 │
│ - 0: IsNext (sentences are consecutive)│
│ - 1: NotNext (sentences are random)     │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Training                                │
│ Learn to distinguish consecutive vs     │
│ random sentence pairs                   │
└─────────────────────────────────────────┘
```

## Embedding Layer Detail

```
┌─────────────────────────────────────────────────────────────┐
│                    BERT Embedding Layer                     │
└─────────────────────────────────────────────────────────────┘

Input: "Hello world"
    │
    ▼
┌─────────────────────────────────────────┐
│ Token IDs                               │
│ [101, 7592, 2088, 102]                 │
│ ([CLS], Hello, world, [SEP])           │
└─────────────────────────────────────────┘
    │
    ├──────────────┬──────────────┬──────────────┐
    │              │              │              │
    ▼              ▼              ▼              ▼
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│ Token   │  │ Segment│  │ Position│  │         │
│ Embed   │  │ Embed   │  │ Embed   │  │         │
│         │  │         │  │         │  │         │
│ [CLS]→  │  │ [0,0,0, │  │ [0,1,2, │  │         │
│ Hello→  │  │  0,1,1, │  │  3]     │  │         │
│ world→  │  │  1]     │  │         │  │         │
│ [SEP]→  │  │         │  │         │  │         │
└─────────┘  └─────────┘  └─────────┘  └─────────┘
    │              │              │              │
    └──────────────┴──────────────┴──────────────┘
                    │
                    ▼
            ┌───────────────┐
            │ Sum All Three │
            │ Embeddings    │
            └───────────────┘
                    │
                    ▼
            ┌───────────────┐
            │ LayerNorm     │
            │ (eps=1e-12)   │
            └───────────────┘
                    │
                    ▼
            ┌───────────────┐
            │ Dropout       │
            └───────────────┘
                    │
                    ▼
        Output: (batch, seq_len, hidden_size)
```

