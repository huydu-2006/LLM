# %%
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding ("Attention Is All You Need")."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout=0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    """Token embedding + positional encoding."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx=None,
        max_len: int = 5000,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_encode = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch, seq_len), int64
        x = self.token_embed(input_ids) * math.sqrt(self.d_model)
        return self.pos_encode(x)
