# %%
import math

import torch
import torch.nn as nn


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, is_causal: bool = False, dropout=0.1):
        super().__init__()
        self.is_causal: bool = is_causal
        self.dropout = nn.Dropout(dropout)
        self.attn_weights: torch.Tensor | None = None  # kept for visualization

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None
    ) -> torch.Tensor:
        d_k: int = key.size(-1)
        scores: torch.Tensor = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            d_k
        )
        if self.is_causal:
            scores = scores + torch.triu(
                torch.full_like(scores, float("-inf")), diagonal=1
            )
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))  # False = pad → -inf
        attn_weight: torch.Tensor = torch.softmax(scores, dim=-1)
        self.attn_weights = attn_weight.detach()  # (batch, heads, len_q, len_kv)
        return torch.matmul(self.dropout(attn_weight), value)
