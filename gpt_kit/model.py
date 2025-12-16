"""
GPT Model Architecture.

Contains the complete GPT model implementation including:
- LayerNorm
- GELU activation
- FeedForward network
- MultiHeadAttention
- TransformerBlock
- GPTModel
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class LayerNorm(nn.Module):
    """
    Layer normalization module.
    
    Args:
        embedding_dim: Dimension of the embeddings
    """
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)
            
        Returns:
            Normalized tensor of same shape
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * normalized_x + self.shift


class GELU(nn.Module):
    """
    GELU (Gaussian Error Linear Unit) activation function used in GPT models.
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
        embedding_dim = config["emb_dim"]
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feedforward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)
            
        Returns:
            Output tensor of same shape
        """
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension (must be divisible by num_heads)
        context_length: Maximum context length for causal masking
        dropout_rate: Dropout rate
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in Q, K, V projections
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        context_length: int, 
        dropout_rate: float, 
        num_heads: int = 2, 
        qkv_bias: bool = False
    ):
        super().__init__()
        if output_dim % num_heads != 0:
            raise ValueError(f"output_dim ({output_dim}) must be divisible by num_heads ({num_heads})")
        
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.context_length = context_length
        
        self.W_query = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_key = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_value = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask on-the-fly (more memory efficient)."""
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, num_tokens, input_dim)
            attention_mask: Optional attention mask of shape (batch_size, num_tokens)
                           where 1 = attend, 0 = mask out
            
        Returns:
            Output tensor of shape (batch_size, num_tokens, output_dim)
        """
        batch_size, num_tokens, input_dim = x.shape
        
        # Project to Q, K, V
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        # Reshape for multi-head attention: (batch, num_tokens, num_heads, head_dim)
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = queries @ keys.transpose(2, 3) / (self.head_dim ** 0.5)
        
        # Apply causal mask (generate on-the-fly for memory efficiency)
        causal_mask = self._generate_causal_mask(num_tokens, x.device)
        attention_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), -torch.inf)
        
        # Apply attention mask if provided (for padding)
        if attention_mask is not None:
            # Expand mask: (batch, 1, 1, seq_len)
            mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores.masked_fill_(mask_expanded == 0, -torch.inf)
        
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context_vectors = attention_weights @ values
        
        # Reshape and combine heads: (batch, num_tokens, output_dim)
        context_vectors = context_vectors.transpose(1, 2).contiguous().view(
            batch_size, num_tokens, self.output_dim
        )
        return self.out_proj(context_vectors)


class TransformerBlock(nn.Module):
    """
    Transformer block with attention and feedforward layers.
    
    Args:
        config: Model configuration dictionary
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.attention = MultiHeadAttention(
            input_dim=config["emb_dim"],
            output_dim=config["emb_dim"],
            context_length=config["context_length"],
            dropout_rate=config["drop_rate"],
            num_heads=config["n_heads"],
            qkv_bias=config.get("qkv_bias", False)
        )
        self.feedforward = FeedForward(config)
        self.norm1 = LayerNorm(config["emb_dim"])
        self.norm2 = LayerNorm(config["emb_dim"])
        self.dropout_residual = nn.Dropout(config["drop_rate"])

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer block with residual connections.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor of same shape
        """
        # Pre-norm architecture with residual connections
        x = x + self.dropout_residual(self.attention(self.norm1(x), attention_mask))
        x = x + self.dropout_residual(self.feedforward(self.norm2(x)))
        return x


class GPTModel(nn.Module):
    """
    GPT model architecture.
    
    Args:
        config: Model configuration dictionary
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        vocab_size = config["vocab_size"]
        embedding_dim = config["emb_dim"]
        context_length = config["context_length"]
        dropout_rate = config["drop_rate"]
        num_layers = config["n_layers"]
        
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)
        self.dropout_embedding = nn.Dropout(dropout_rate)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(num_layers)
        ])
        
        self.final_layer_norm = LayerNorm(embedding_dim)
        self.output_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        
        # Cache position IDs for efficiency (up to context_length)
        self.register_buffer(
            'position_ids_cache',
            torch.arange(context_length, dtype=torch.long)
        )

    def forward(
        self, 
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through GPT model.
        
        Args:
            token_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)
                           where 1 = attend, 0 = mask out
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape
        
        # Get embeddings
        token_embeddings = self.token_embedding(token_ids)
        
        # Use cached position IDs if seq_len <= context_length, otherwise generate on-the-fly
        if seq_len <= self.position_ids_cache.size(0):
            position_ids = self.position_ids_cache[:seq_len].to(token_ids.device)
        else:
            # Fallback for sequences longer than context_length
            position_ids = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
        
        position_embeddings = self.position_embedding(position_ids)
        
        # Combine embeddings
        hidden_states = token_embeddings + position_embeddings
        hidden_states = self.dropout_embedding(hidden_states)
        
        # Pass through transformer blocks (using ModuleList to pass attention_mask)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, attention_mask)
        
        hidden_states = self.final_layer_norm(hidden_states)
        
        # Generate logits
        logits = self.output_head(hidden_states)
        return logits

