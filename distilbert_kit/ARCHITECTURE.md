# DistilBERT Architecture Diagrams

## High-Level Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     DistilBERT Model Flow                      │
└─────────────────────────────────────────────────────────────────┘

Input Text
    │
    ▼
┌─────────────────┐
│ Tokenization    │  Convert text to token IDs using WordPiece
│ (WordPiece)     │  Example: "Hello world" → [token_ids]
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
│ NOTE: No Segment│  Removed to reduce parameters
│ Embeddings      │  (simplification for distillation)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Embedding       │  token_emb + pos_emb (no segment_emb)
│ Sum + LayerNorm │  Shape: (batch, seq_len, hidden_size)
│ + Dropout       │  LayerNorm epsilon: 1e-12 (same as BERT)
└─────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│        Transformer Encoder Blocks (6 layers - HALF of BERT) │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Block 1:                                              │  │
│  │   ┌──────────────────────────────────────┐           │  │
│  │   │ Bidirectional Multi-Head Attention  │           │  │
│  │   │ - Q, K, V projections                │           │  │
│  │   │ - NO causal mask (bidirectional)     │           │  │
│  │   │ - Same attention mechanism as BERT    │           │  │
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
│  │ Block 6: (last layer - BERT has 12)                 │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────┐
│ Hidden States   │  Contextualized representations
│ (batch, seq_len,│  Each token has context from entire sequence
│  hidden_size)   │
└─────────────────┘
```

## Size Comparison

```
┌─────────────────────────────────────────────────────────────┐
│              DistilBERT vs BERT Size Comparison             │
└─────────────────────────────────────────────────────────────┘

ARCHITECTURE:
─────────────

BERT-base:
  Layers: 12
  Hidden Size: 768
  Attention Heads: 12
  Parameters: ~110M

DistilBERT:
  Layers: 6 (50% reduction)
  Hidden Size: 768 (same)
  Attention Heads: 12 (same)
  Parameters: ~66M (40% reduction)

COMPUTATION:
────────────

BERT-base:
  Forward Pass: 12 layers × computation
  Memory: Full model size
  Speed: Baseline

DistilBERT:
  Forward Pass: 6 layers × computation (50% fewer)
  Memory: 40% less
  Speed: 60% faster

PERFORMANCE:
────────────

BERT-base:
  Performance: 100% (baseline)
  Accuracy: Full BERT performance

DistilBERT:
  Performance: ~97% of BERT
  Accuracy: Slight trade-off for speed
```

## Knowledge Distillation Process

```
┌─────────────────────────────────────────────────────────────┐
│              Knowledge Distillation Flow                    │
└─────────────────────────────────────────────────────────────┘

TEACHER MODEL (BERT):
─────────────────────
Input: "The cat sat"
    │
    ▼
BERT (12 layers, 110M params)
    │
    ▼
Output: [logits for "cat"]
        High confidence, accurate predictions

STUDENT MODEL (DistilBERT):
───────────────────────────
Input: "The cat sat"
    │
    ▼
DistilBERT (6 layers, 66M params)
    │
    ▼
Output: [logits for "cat"]
        Learning to mimic teacher

TRAINING PROCESS:
─────────────────
1. Teacher makes predictions (soft labels)
2. Student makes predictions
3. Loss = α × Hard Loss + (1-α) × Distillation Loss
   
   Hard Loss: Cross-entropy with true labels
   Distillation Loss: KL divergence with teacher's soft labels
   
4. Student learns to:
   - Match teacher's predictions (soft labels)
   - Match true labels (hard labels)
   - Use fewer parameters

RESULT:
───────
Student (DistilBERT) achieves:
- 97% of teacher's performance
- 40% fewer parameters
- 60% faster inference
```

## Why It's Faster

```
┌─────────────────────────────────────────────────────────────┐
│              Why DistilBERT is Faster                       │
└─────────────────────────────────────────────────────────────┘

COMPUTATION REDUCTION:
──────────────────────

BERT (12 layers):
  Layer 1 → Layer 2 → ... → Layer 12
  Total: 12 × (Attention + FFN) operations

DistilBERT (6 layers):
  Layer 1 → Layer 2 → ... → Layer 6
  Total: 6 × (Attention + FFN) operations
  
Reduction: 50% fewer operations

MEMORY REDUCTION:
─────────────────

BERT:
  Parameters: 110M
  Memory: ~440MB (FP32)
  
DistilBERT:
  Parameters: 66M
  Memory: ~264MB (FP32)
  
Reduction: 40% less memory

PARAMETER REDUCTION:
────────────────────

Removed Components:
  - 6 encoder layers (50% reduction)
  - Segment embeddings (token type embeddings)
  
Kept Components:
  - Same hidden size (768)
  - Same attention heads (12)
  - Same vocabulary size

SPEED IMPROVEMENTS:
───────────────────
- Fewer layers = faster forward pass
- Less memory = better cache utilization
- Smaller model = faster loading
- 60% faster inference overall
```

## Training Flow

```
┌─────────────────────────────────────────────────────────────┐
│              DistilBERT Training Flow                       │
└─────────────────────────────────────────────────────────────┘

Training Data
    │
    ▼
┌─────────────────────────────────────────┐
│ Create Training Examples                │
│ - Masked Language Modeling (MLM)        │
│ - Same as BERT                          │
└─────────────────────────────────────────┘
    │
    ├──────────────────────┬──────────────────────┐
    │                      │                      │
    ▼                      ▼                      ▼
┌──────────┐      ┌──────────────┐      ┌──────────────┐
│ Teacher  │      │ Student      │      │ True         │
│ (BERT)   │      │ (DistilBERT) │      │ Labels       │
└──────────┘      └──────────────┘      └──────────────┘
    │                      │                      │
    ▼                      ▼                      ▼
┌──────────┐      ┌──────────────┐      ┌──────────────┐
│ Teacher  │      │ Student      │      │ Hard Loss    │
│ Logits   │      │ Logits       │      │ (CE with     │
│ (soft)   │      │              │      │  true labels)│
└──────────┘      └──────────────┘      └──────────────┘
    │                      │                      │
    └──────────────────────┴──────────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Combined Loss   │
                  │                 │
                  │ Loss =          │
                  │   α × Hard Loss │
                  │   + (1-α) ×     │
                  │   Distillation  │
                  │   Loss          │
                  └─────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Backward Pass   │
                  │ Update Student  │
                  │ Parameters      │
                  └─────────────────┘
```

## Use Case Comparison

```
┌─────────────────────────────────────────────────────────────┐
│              When to Use DistilBERT vs BERT                 │
└─────────────────────────────────────────────────────────────┘

USE DISTILBERT WHEN:
────────────────────
✓ Speed is critical
  - Real-time applications
  - High-throughput systems
  - Edge devices

✓ Memory is limited
  - Mobile devices
  - Embedded systems
  - Multiple models in memory

✓ Performance trade-off acceptable
  - 97% performance is sufficient
  - Slight accuracy loss OK

✓ Large-scale deployment
  - Many concurrent requests
  - Cost-sensitive applications

USE BERT WHEN:
──────────────
✓ Maximum accuracy needed
  - Research applications
  - Critical decision systems
  - Benchmark competitions

✓ Resources available
  - Sufficient GPU memory
  - No strict latency requirements
  - Single model deployment

✓ Complex tasks
  - Tasks requiring full model capacity
  - Fine-tuned for specific domains
```

## Architecture Simplifications

```
┌─────────────────────────────────────────────────────────────┐
│         DistilBERT Architecture Simplifications             │
└─────────────────────────────────────────────────────────────┘

REMOVED:
────────
1. Segment Embeddings
   BERT: token_emb + segment_emb + pos_emb
   DistilBERT: token_emb + pos_emb
   
   Reason: Not essential, reduces parameters

2. 6 Encoder Layers
   BERT: 12 layers
   DistilBERT: 6 layers
   
   Reason: Knowledge distillation allows fewer layers
           while maintaining performance

KEPT:
─────
1. Hidden Size: 768 (same as BERT-base)
   Reason: Maintains representation capacity

2. Attention Heads: 12 (same as BERT-base)
   Reason: Multi-head attention still important

3. Vocabulary Size: Same as BERT
   Reason: Compatible tokenization

4. Architecture Style: Same as BERT
   Reason: Easier distillation, familiar structure

RESULT:
───────
Simpler architecture that:
- Trains faster
- Runs faster
- Uses less memory
- Maintains 97% performance
```

