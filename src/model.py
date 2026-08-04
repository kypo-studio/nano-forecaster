"""Small encoder-only causal Transformer implemented without nn.Transformer."""
from __future__ import annotations

import math
import torch
from torch import nn


def causal_mask(length: int, device=None) -> torch.Tensor:
    """True entries are forbidden: token i cannot attend to any j > i."""
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)


class MultiHeadCausalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        batch, length, width = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        # Applying -infinity before softmax makes every forbidden future weight 0.
        scores = scores.masked_fill(causal_mask(length, x.device), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attended = self.dropout(weights) @ v
        attended = attended.transpose(1, 2).contiguous().reshape(batch, length, width)
        output = self.output(attended)
        return (output, weights) if return_weights else output


class SinusoidalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        positions = torch.arange(max_len).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        table = torch.zeros(max_len, d_model)
        table[:, 0::2] = torch.sin(positions * frequencies)
        table[:, 1::2] = torch.cos(positions * frequencies[:table[:, 1::2].shape[1]])
        self.register_buffer("table", table, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.table[:x.shape[1]]


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadCausalAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x)))
        return x + self.dropout(self.ffn(self.norm2(x)))


class NanoForecaster(nn.Module):
    def __init__(self, n_features: int, input_len: int, horizon: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2, d_ff: int = 128,
                 dropout: float = 0.1, positional_encoding: str = "sinusoidal"):
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        if positional_encoding == "sinusoidal":
            self.position = SinusoidalEncoding(d_model, input_len)
        elif positional_encoding == "learned":
            self.position = None
            self.position_embedding = nn.Parameter(torch.zeros(1, input_len, d_model))
            nn.init.normal_(self.position_embedding, std=0.02)
        else:
            raise ValueError("positional_encoding must be sinusoidal or learned")
        self.blocks = nn.ModuleList(
            EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.position(x) if self.position is not None else x + self.position_embedding[:, :x.shape[1]]
        for block in self.blocks:
            x = block(x)
        # Direct multi-horizon forecast from the final causal representation.
        return self.head(self.final_norm(x[:, -1]))


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

