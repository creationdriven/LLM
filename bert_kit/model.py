"""
BERT Model Architecture.

Contains the complete BERT model implementation including:
- LayerNorm
- GELU activation
- FeedForward network
- MultiHeadAttention (bidirectional, no causal mask)
- TransformerEncoderBlock
- BERTModel
- BERTForClassification
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class LayerNorm(nn.Module):
    """
    Layer normalization module (same as GPT but BERT uses different epsilon).
    
    Args:
        hidden_size: Dimension of the hidden states
        eps: Epsilon value for numerical stability
    """
    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Normalized tensor of same shape
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * normalized_x + self.bias


class GELU(nn.Module):
    """
    GELU (Gaussian Error Linear Unit) activation function used in BERT.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GELU activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Activated tensor
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """
    Feedforward neural network module with GELU activation.
    
    Args:
        config: Model configuration dictionary
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        hidden_size = config["hidden_size"]
        intermediate_size = config["intermediate_size"]
        
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.activation = GELU()
        self.dense_output = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feedforward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Output tensor of same shape
        """
        x = self.dense(x)
        x = self.activation(x)
        x = self.dense_output(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head bidirectional self-attention mechanism for BERT.
    
    Unlike GPT's causal attention, BERT uses bidirectional attention where each
    token can attend to ALL other tokens in the sequence (both past and future).
    This allows BERT to understand the full context of each word.
    
    Architecture:
    1. Project input to Q (query), K (key), V (value) vectors
    2. Split into multiple heads for parallel attention computation
    3. Compute attention scores: Q @ K^T / sqrt(head_dim)
    4. NO causal mask - all tokens can attend to all tokens
    5. Apply softmax to get attention weights
    6. Weighted sum of values using attention weights
    7. Concatenate heads and project to output dimension
    
    This bidirectional attention is what makes BERT powerful for understanding
    tasks like classification and question answering, where context from both
    directions is important.
    
    Args:
        config: Model configuration dictionary
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        hidden_size = config["hidden_size"]
        num_attention_heads = config["num_attention_heads"]
        
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = hidden_size
        
        # Q, K, V projections
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        self.dense = nn.Linear(hidden_size, hidden_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose tensor for multi-head attention."""
        batch_size, seq_len = x.size(0), x.size(1)
        x = x.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        return x.transpose(1, 2)  # (batch, heads, seq_len, head_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through multi-head attention.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional mask tensor of shape (batch_size, seq_len)
                           where 1 = attend, 0 = mask out
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Compute attention scores: (batch, heads, seq_len, seq_len)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(
            torch.tensor(self.attention_head_size, dtype=torch.float32)
        )
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask to match attention scores shape
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape and combine heads
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.view(batch_size, seq_len, self.all_head_size)
        
        # Final projection
        output = self.dense(context_layer)
        return output


class TransformerEncoderBlock(nn.Module):
    """
    BERT transformer encoder block with bidirectional attention.
    
    Args:
        config: Model configuration dictionary
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feedforward = FeedForward(config)
        self.layer_norm1 = LayerNorm(config["hidden_size"], config["layer_norm_eps"])
        self.layer_norm2 = LayerNorm(config["hidden_size"], config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer encoder block.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor of same shape
        """
        # Self-attention with residual connection
        attention_output = self.attention(self.layer_norm1(hidden_states), attention_mask)
        hidden_states = hidden_states + self.dropout(attention_output)
        
        # Feedforward with residual connection
        feedforward_output = self.feedforward(self.layer_norm2(hidden_states))
        hidden_states = hidden_states + self.dropout(feedforward_output)
        
        return hidden_states


class BERTModel(nn.Module):
    """
    BERT model architecture (encoder only).
    
    Args:
        config: Model configuration dictionary
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"]
        max_position_embeddings = config["max_position_embeddings"]
        type_vocab_size = config["type_vocab_size"]
        
        # Embeddings
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        self.layer_norm = LayerNorm(hidden_size, config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        
        # Transformer encoder blocks
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(config)
            for _ in range(config["num_hidden_layers"])
        ])
        
        # Initialize embeddings
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.word_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.token_type_embeddings.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through BERT model.
        
        This method implements the complete BERT forward pass:
        1. Convert token IDs to embeddings
        2. Add segment (token type) embeddings (for sentence pairs)
        3. Add positional embeddings (learned)
        4. Sum all embeddings and apply layer norm + dropout
        5. Pass through N encoder blocks (each with bidirectional attention)
        6. Return contextualized hidden states
        
        Unlike GPT, BERT uses bidirectional attention, so each token can attend
        to all other tokens in the sequence (both past and future). This makes
        BERT powerful for understanding tasks but unsuitable for generation.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
                     Each value is an integer index into the vocabulary
            attention_mask: Optional mask of shape (batch_size, seq_len)
                          where 1 = attend, 0 = mask out (for padding)
                          If None, all tokens are attended to
            token_type_ids: Optional segment IDs of shape (batch_size, seq_len)
                          Used to distinguish sentence A (0) from sentence B (1)
                          If None, assumes single sentence (all zeros)
            
        Returns:
            Hidden states of shape (batch_size, seq_len, hidden_size)
            Each token now has a contextualized representation that includes
            information from all other tokens in the sequence
        """
        batch_size, seq_len = input_ids.shape
        
        # Step 1: Convert token IDs to dense embeddings
        # Shape: (batch_size, seq_len) → (batch_size, seq_len, hidden_size)
        # Each token ID is mapped to a learned embedding vector
        word_embeddings = self.word_embeddings(input_ids)
        
        # Step 2: Create position IDs and get positional embeddings
        # BERT uses learned positional embeddings (not sinusoidal)
        # Generate position IDs: [0, 1, 2, ..., seq_len-1] for each sequence
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        # Shape: (batch_size, seq_len) → (batch_size, seq_len, hidden_size)
        position_embeddings = self.position_embeddings(position_ids)
        
        # Step 3: Get segment (token type) embeddings
        # Used to distinguish sentence A from sentence B in NSP task
        # If not provided, assume single sentence (all zeros)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        # Shape: (batch_size, seq_len) → (batch_size, seq_len, hidden_size)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # Step 4: Combine all three embeddings
        # Sum them element-wise (broadcasting handles dimensions)
        # Shape: (batch_size, seq_len, hidden_size)
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        
        # Step 5: Apply layer normalization and dropout
        # Layer norm stabilizes training (BERT uses eps=1e-12)
        embeddings = self.layer_norm(embeddings)
        # Dropout provides regularization
        embeddings = self.dropout(embeddings)
        
        # Step 6: Pass through transformer encoder blocks
        # Each block applies bidirectional attention and feed-forward network
        # The attention_mask (if provided) masks out padding tokens
        # Unlike GPT, there's NO causal mask - all tokens can attend to all tokens
        hidden_states = embeddings
        for encoder_layer in self.encoder_layers:
            # Each layer refines the representations using bidirectional context
            # Token at position i can see tokens at all positions (0 to seq_len-1)
            hidden_states = encoder_layer(hidden_states, attention_mask)
        
        # Return contextualized hidden states
        # Each token's representation now includes information from the entire sequence
        # This bidirectional context makes BERT powerful for understanding tasks
        return hidden_states


class BERTForMaskedLM(nn.Module):
    """
    BERT model for Masked Language Modeling (MLM) pretraining.
    
    Args:
        config: Model configuration dictionary
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.bert = BERTModel(config)
        self.cls = nn.Linear(config["hidden_size"], config["vocab_size"])
        
        # Tie weights between word embeddings and output layer
        self.cls.weight = self.bert.word_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for MLM.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            token_type_ids: Optional segment IDs
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        hidden_states = self.bert(input_ids, attention_mask, token_type_ids)
        logits = self.cls(hidden_states)
        return logits


class BERTForSequenceClassification(nn.Module):
    """
    BERT model for sequence classification tasks.
    
    Args:
        config: Model configuration dictionary
        num_labels: Number of classification labels
    """
    def __init__(self, config: Dict[str, Any], num_labels: int = 2):
        super().__init__()
        self.bert = BERTModel(config)
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.classifier = nn.Linear(config["hidden_size"], num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for classification.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            token_type_ids: Optional segment IDs
            
        Returns:
            Logits of shape (batch_size, num_labels)
        """
        hidden_states = self.bert(input_ids, attention_mask, token_type_ids)
        
        # Use [CLS] token for classification
        cls_hidden_state = hidden_states[:, 0, :]
        cls_hidden_state = self.dropout(cls_hidden_state)
        logits = self.classifier(cls_hidden_state)
        
        return logits

