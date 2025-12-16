"""
RoBERTa Model Architecture.

RoBERTa (Robustly Optimized BERT) is similar to BERT but with:
- No token type embeddings (single sentence format)
- Different layer norm epsilon (1e-5)
- Dynamic masking during training
- No NSP task
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .config import BASE_CONFIG


class LayerNorm(nn.Module):
    """Layer normalization with RoBERTa's epsilon (1e-5)."""
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * normalized_x + self.bias


class GELU(nn.Module):
    """GELU activation function."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """Feedforward network with GELU activation."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        hidden_size = config["hidden_size"]
        intermediate_size = config["intermediate_size"]
        
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.activation = GELU()
        self.dense_output = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        x = self.activation(x)
        x = self.dense_output(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention (bidirectional, no causal mask)."""
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
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        self.dense = nn.Linear(hidden_size, hidden_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size(0), x.size(1)
        x = x.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        return x.transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(
            torch.tensor(self.attention_head_size, dtype=torch.float32)
        )
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask
        
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.view(batch_size, seq_len, self.all_head_size)
        
        output = self.dense(context_layer)
        return output


class TransformerEncoderBlock(nn.Module):
    """RoBERTa transformer encoder block."""
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
        attention_output = self.attention(self.layer_norm1(hidden_states), attention_mask)
        hidden_states = hidden_states + self.dropout(attention_output)
        
        feedforward_output = self.feedforward(self.layer_norm2(hidden_states))
        hidden_states = hidden_states + self.dropout(feedforward_output)
        
        return hidden_states


class RoBERTaModel(nn.Module):
    """
    RoBERTa model architecture (encoder only, no token type embeddings).
    
    Args:
        config: Model configuration dictionary
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"]
        max_position_embeddings = config["max_position_embeddings"]
        
        # Embeddings (no token type embeddings for RoBERTa)
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        self.layer_norm = LayerNorm(hidden_size, config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        
        # Transformer encoder blocks
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(config)
            for _ in range(config["num_hidden_layers"])
        ])
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.word_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through RoBERTa model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)
            
        Returns:
            Hidden states of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len = input_ids.shape
        
        word_embeddings = self.word_embeddings(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)
        
        # No token type embeddings for RoBERTa
        embeddings = word_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        hidden_states = embeddings
        for encoder_layer in self.encoder_layers:
            hidden_states = encoder_layer(hidden_states, attention_mask)
        
        return hidden_states


class RoBERTaForMaskedLM(nn.Module):
    """
    RoBERTa model for Masked Language Modeling (MLM) pretraining.
    
    Args:
        config: Model configuration dictionary
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.roberta = RoBERTaModel(config)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"])
        
        # Tie weights between word embeddings and output layer
        self.lm_head.weight = self.roberta.word_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for MLM.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        hidden_states = self.roberta(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)
        return logits


class RoBERTaForSequenceClassification(nn.Module):
    """
    RoBERTa model for sequence classification tasks.
    
    Args:
        config: Model configuration dictionary
        num_labels: Number of classification labels
    """
    def __init__(self, config: Dict[str, Any], num_labels: int = 2):
        super().__init__()
        self.roberta = RoBERTaModel(config)
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.classifier = nn.Linear(config["hidden_size"], num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for classification.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            
        Returns:
            Logits of shape (batch_size, num_labels)
        """
        hidden_states = self.roberta(input_ids, attention_mask)
        
        # Use first token ([CLS]) for classification
        cls_hidden_state = hidden_states[:, 0, :]
        cls_hidden_state = self.dropout(cls_hidden_state)
        logits = self.classifier(cls_hidden_state)
        
        return logits

