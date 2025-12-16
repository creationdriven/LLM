"""
T5 Model Architecture.

T5 (Text-To-Text Transfer Transformer) is an encoder-decoder model
that treats all NLP tasks as text-to-text problems.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Any, Optional, Tuple


class LayerNorm(nn.Module):
    """T5 Layer Normalization (no bias)."""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps)


class RelativePositionBias(nn.Module):
    """Relative position bias for T5 attention."""
    def __init__(self, num_buckets: int, num_heads: int, max_distance: int = 128):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """Convert relative position to bucket index."""
        num_buckets = self.num_buckets
        max_exact = num_buckets // 2
        
        # Handle negative positions
        relative_position = relative_position.abs()
        is_small = relative_position < max_exact
        
        # For large positions, use logarithmic bucketing
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float().clamp(min=1) / max_exact) /
            math.log(self.max_distance / max_exact) *
            (num_buckets - max_exact)
        ).long()
        relative_position_if_large = torch.clamp(relative_position_if_large, 0, num_buckets - 1)
        
        result = torch.where(is_small, relative_position, relative_position_if_large)
        return torch.clamp(result, 0, num_buckets - 1)

    def forward(self, query_len: int, key_len: int) -> torch.Tensor:
        """Compute relative position bias."""
        device = self.relative_attention_bias.weight.device
        q_pos = torch.arange(query_len, dtype=torch.long, device=device)
        k_pos = torch.arange(key_len, dtype=torch.long, device=device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        
        rel_pos_bucket = self._relative_position_bucket(rel_pos)
        values = self.relative_attention_bias(rel_pos_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values


class T5Attention(nn.Module):
    """T5 Multi-head attention with relative position bias."""
    def __init__(self, config: Dict[str, Any], is_decoder: bool = False):
        super().__init__()
        self.d_model = config["d_model"]
        self.num_heads = config["num_heads"]
        self.head_dim = self.d_model // self.num_heads
        self.is_decoder = is_decoder
        
        self.q = nn.Linear(self.d_model, self.d_model)
        self.k = nn.Linear(self.d_model, self.d_model)
        self.v = nn.Linear(self.d_model, self.d_model)
        self.o = nn.Linear(self.d_model, self.d_model)
        
        self.dropout = nn.Dropout(config["dropout_rate"])
        self.relative_attention_bias = RelativePositionBias(
            config["relative_attention_num_buckets"],
            self.num_heads,
            config["relative_attention_max_distance"]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len = hidden_states.shape[:2]
        
        if key_value_states is None:
            key_value_states = hidden_states
        
        # Project to Q, K, V
        query = self.q(hidden_states)
        key = self.k(key_value_states)
        value = self.v(key_value_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # Add relative position bias
        rel_bias = self.relative_attention_bias(seq_len, key.shape[2])
        scores = scores + rel_bias
        
        # Apply causal mask for decoder
        if self.is_decoder:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1)
            scores.masked_fill_(causal_mask.bool().unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply attention mask
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply to values
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        output = self.o(attn_output)
        return output, None


class T5FeedForward(nn.Module):
    """T5 Feed-forward network with GELU."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.wi = nn.Linear(config["d_model"], config["d_ff"])
        self.wo = nn.Linear(config["d_ff"], config["d_model"])
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.wi(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5Block(nn.Module):
    """T5 Transformer block."""
    def __init__(self, config: Dict[str, Any], is_decoder: bool = False):
        super().__init__()
        self.is_decoder = is_decoder
        self.self_attention = T5Attention(config, is_decoder=is_decoder)
        if is_decoder:
            self.cross_attention = T5Attention(config, is_decoder=False)
        self.feed_forward = T5FeedForward(config)
        self.layer_norm = nn.ModuleList([
            LayerNorm(config["d_model"], config["layer_norm_eps"]),
            LayerNorm(config["d_model"], config["layer_norm_eps"]),
            LayerNorm(config["d_model"], config["layer_norm_eps"]),
        ])
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention
        normed_hidden_states = self.layer_norm[0](hidden_states)
        attention_output, _ = self.self_attention(
            normed_hidden_states,
            attention_mask=attention_mask
        )
        hidden_states = hidden_states + self.dropout(attention_output)
        
        # Cross-attention (decoder only)
        if self.is_decoder and encoder_hidden_states is not None:
            normed_hidden_states = self.layer_norm[1](hidden_states)
            cross_attention_output, _ = self.cross_attention(
                normed_hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask
            )
            hidden_states = hidden_states + self.dropout(cross_attention_output)
        
        # Feed-forward
        normed_hidden_states = self.layer_norm[-1](hidden_states)
        ff_output = self.feed_forward(normed_hidden_states)
        hidden_states = hidden_states + self.dropout(ff_output)
        
        return hidden_states


class T5Stack(nn.Module):
    """Stack of T5 blocks (encoder or decoder)."""
    def __init__(self, config: Dict[str, Any], is_decoder: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([
            T5Block(config, is_decoder=is_decoder)
            for _ in range(config["num_layers"] if not is_decoder else config["num_decoder_layers"])
        ])
        self.final_layer_norm = LayerNorm(config["d_model"], config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        
        hidden_states = inputs_embeds
        
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask
            )
        
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states


class T5Model(nn.Module):
    """T5 encoder-decoder model."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.shared = nn.Embedding(config["vocab_size"], config["d_model"])
        
        self.encoder = T5Stack(config, is_decoder=False)
        self.decoder = T5Stack(config, is_decoder=True)
        
        self.config = config

    def get_input_embeddings(self):
        return self.shared

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Encoder
        encoder_inputs_embeds = self.shared(input_ids)
        encoder_hidden_states = self.encoder(
            inputs_embeds=encoder_inputs_embeds,
            attention_mask=attention_mask
        )
        
        # Decoder
        decoder_inputs_embeds = self.shared(decoder_input_ids)
        decoder_hidden_states = self.decoder(
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask
        )
        
        return {
            "encoder_last_hidden_state": encoder_hidden_states,
            "decoder_last_hidden_state": decoder_hidden_states,
        }


class T5ForConditionalGeneration(nn.Module):
    """T5 model for conditional generation (text-to-text)."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.model = T5Model(config)
        self.lm_head = nn.Linear(config["d_model"], config["vocab_size"], bias=False)
        
        # Tie weights
        self.lm_head.weight = self.model.shared.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        outputs = self.model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask
        )
        
        logits = self.lm_head(outputs["decoder_last_hidden_state"])
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {
            "loss": loss,
            "logits": logits,
            "encoder_last_hidden_state": outputs["encoder_last_hidden_state"],
            "decoder_last_hidden_state": outputs["decoder_last_hidden_state"],
        }

