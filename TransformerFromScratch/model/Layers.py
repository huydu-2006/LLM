# %%
import torch.nn as nn
from SubLayer import FeedForward, MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_hid, num_heads, d_k, d_v, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model, d_k, d_v, num_heads, dropout=dropout
        )
        self.fw = FeedForward(d_model, d_hid, dropout=dropout)

    def forward(self, enc_input: torch.Tensor, src_pad_mask=None) -> torch.Tensor:
        return self.fw(self.self_attn(enc_input, enc_input, enc_input, mask=src_pad_mask))  # enc_output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_hid, num_heads, d_k, d_v, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model, d_k, d_v, num_heads, is_causal=True, dropout=dropout
        )
        self.cross_attn = MultiHeadAttention(
            d_model, d_k, d_v, num_heads, dropout=dropout
        )
        self.fw = FeedForward(d_model, d_hid, dropout=dropout)

    def forward(
        self,
        dec_input: torch.Tensor,
        enc_output: torch.Tensor,
        tgt_pad_mask=None,
        src_pad_mask=None,
    ) -> torch.Tensor:

        x = self.self_attn(dec_input, dec_input, dec_input, mask=tgt_pad_mask)
        x = self.cross_attn(x, enc_output, enc_output, mask=src_pad_mask)
        return self.fw(x)
