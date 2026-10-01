# %%
import torch
import torch.nn as nn
from Modules import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, d_k: int, d_v: int, num_heads: int, is_causal: bool = False, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.W_q: nn.Linear = nn.Linear(d_model, num_heads * d_k, bias=False)
        self.W_k: nn.Linear = nn.Linear(d_model, num_heads * d_k, bias=False)
        self.W_v: nn.Linear = nn.Linear(d_model, num_heads * d_v, bias=False)
        self.W_o: nn.Linear = nn.Linear(num_heads * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(is_causal=is_causal)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None) -> torch.Tensor:
        # Q, K, V = (num_batches, len, d_model)
        num_batches, len_Q, len_KV = Q.size(0), Q.size(1), V.size(1)

        residual = Q
        Q = self.W_q(Q).view(num_batches, len_Q, self.num_heads, self.d_k)
        K = self.W_k(K).view(num_batches, len_KV, self.num_heads, self.d_k)
        V = self.W_v(V).view(num_batches, len_KV, self.num_heads, self.d_v)

        # Q, K, V = (num_batches, num_heads, len, d)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        Z = self.attention(Q, K, V, mask=mask)

        Z = Z.transpose(1, 2).contiguous().view(num_batches, len_Q, -1)

        X = self.dropout(self.W_o(Z))

        return self.layer_norm(X + residual)

class FeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_in)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        residual = X
        X = self.w_2(torch.relu(self.w_1(X)))
        X = self.dropout(X)
        return self.layer_norm(X + residual)
