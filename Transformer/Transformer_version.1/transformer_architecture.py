"""A compact, from-scratch Transformer encoder-decoder in PyTorch.

Architecture: Vaswani et al., "Attention Is All You Need".
References used while organizing this implementation:
  - https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/transformer.py
  - https://github.com/harvardnlp/annotated-transformer

Tensor convention throughout: (batch, sequence, feature).
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal position information added to token embeddings."""

    def __init__(self, d_model: int, max_length: int = 5000) -> None:
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency[: d_model // 2])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.size(1) > self.encoding.size(1):
            raise ValueError("Sequence length exceeds max_length")
        return x + self.encoding[:, : x.size(1)].to(dtype=x.dtype)


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention computed in multiple representation heads."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_projection = nn.Linear(d_model, d_model)
        self.k_projection = nn.Linear(d_model, d_model)
        self.v_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        q = self._split_heads(self.q_projection(query))
        k = self._split_heads(self.k_projection(key))
        v = self._split_heads(self.v_projection(value))

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            # True means the attention connection is allowed.
            scores = scores.masked_fill(~mask.to(device=scores.device, dtype=torch.bool), -1e9)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        attended = weights @ v

        batch, _, length, _ = attended.shape
        joined = attended.transpose(1, 2).contiguous().view(batch, length, -1)
        return self.output_projection(joined)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, source_mask: Optional[Tensor]) -> Tensor:
        x = self.norm1(x + self.dropout(self.self_attention(x, x, x, source_mask)))
        return self.norm2(x + self.dropout(self.feed_forward(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        target_mask: Optional[Tensor],
        source_mask: Optional[Tensor],
    ) -> Tensor:
        x = self.norm1(x + self.dropout(self.self_attention(x, x, x, target_mask)))
        x = self.norm2(
            x + self.dropout(self.cross_attention(x, memory, memory, source_mask))
        )
        return self.norm3(x + self.dropout(self.feed_forward(x)))


class Transformer(nn.Module):
    """Sequence-to-sequence Transformer returning unnormalized token logits."""

    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_length: int = 5000,
        padding_index: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.padding_index = padding_index
        self.source_embedding = nn.Embedding(source_vocab_size, d_model, padding_index)
        self.target_embedding = nn.Embedding(target_vocab_size, d_model, padding_index)
        self.position = PositionalEncoding(d_model, max_length)
        encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder_layers = nn.ModuleList(
            copy.deepcopy(encoder_layer) for _ in range(num_layers)
        )
        self.decoder_layers = nn.ModuleList(
            copy.deepcopy(decoder_layer) for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(d_model, target_vocab_size)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def _source_mask(self, source_tokens: Tensor) -> Tensor:
        return (source_tokens != self.padding_index)[:, None, None, :]

    def _target_mask(self, target_tokens: Tensor) -> Tensor:
        length = target_tokens.size(1)
        causal = torch.tril(
            torch.ones(length, length, device=target_tokens.device, dtype=torch.bool)
        )
        non_padding = (target_tokens != self.padding_index)[:, None, None, :]
        return non_padding & causal[None, None, :, :]

    def encode(self, source_tokens: Tensor, source_mask: Tensor) -> Tensor:
        x = self.dropout(
            self.position(self.source_embedding(source_tokens) * math.sqrt(self.d_model))
        )
        for layer in self.encoder_layers:
            x = layer(x, source_mask)
        return x

    def decode(
        self,
        target_tokens: Tensor,
        memory: Tensor,
        target_mask: Tensor,
        source_mask: Tensor,
    ) -> Tensor:
        x = self.dropout(
            self.position(self.target_embedding(target_tokens) * math.sqrt(self.d_model))
        )
        for layer in self.decoder_layers:
            x = layer(x, memory, target_mask, source_mask)
        return x

    def forward(self, source_tokens: Tensor, target_tokens: Tensor) -> Tensor:
        source_mask = self._source_mask(source_tokens)
        target_mask = self._target_mask(target_tokens)
        memory = self.encode(source_tokens, source_mask)
        decoded = self.decode(target_tokens, memory, target_mask, source_mask)
        return self.output(decoded)


if __name__ == "__main__":
    # Small smoke test. Token id 0 is padding.
    model = Transformer(
        source_vocab_size=100,
        target_vocab_size=120,
        d_model=128,
        num_heads=8,
        num_layers=2,
        d_ff=512,
    )
    source = torch.randint(1, 100, (4, 10))
    target = torch.randint(1, 120, (4, 8))
    logits = model(source, target)
    print(f"output shape: {tuple(logits.shape)}")  # (4, 8, 120)


#测试一下是否好用