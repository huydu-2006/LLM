# %%
import torch
import torch.nn as nn

from Embedding import TransformerEmbedding
from Layers import DecoderLayer, EncoderLayer


class Encoder(nn.Module):
    def __init__(self, num_stacks, d_model, d_hid, num_heads, d_k, d_v, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, d_hid, num_heads, d_k, d_v, dropout=dropout)
                for _ in range(num_stacks)
            ]
        )

    def forward(self, enc_input: torch.Tensor, src_pad_mask=None) -> torch.Tensor:
        for layer in self.layers:
            enc_input = layer(enc_input, src_pad_mask)
        return enc_input


class Decoder(nn.Module):
    def __init__(self, num_stacks, d_model, d_hid, num_heads, d_k, d_v, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, d_hid, num_heads, d_k, d_v, dropout=dropout)
                for _ in range(num_stacks)
            ]
        )

    def forward(
        self, dec_input, enc_output, tgt_pad_mask=None, src_pad_mask=None
    ) -> torch.Tensor:
        for layer in self.layers:
            dec_input = layer(dec_input, enc_output, tgt_pad_mask, src_pad_mask)
        return dec_input


class Transformer(nn.Module):
    """Full encoder-decoder Transformer: embeddings -> encoder -> decoder -> projection."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        num_stacks: int = 6,
        d_model: int = 512,
        d_hid: int = 2048,
        num_heads: int = 8,
        d_k: int = 64,
        d_v: int = 64,
        dropout=0.1,
        max_len: int = 5000,
        src_pad_id=None,
        tgt_pad_id=None,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id

        self.src_embed = TransformerEmbedding(
            src_vocab_size, d_model, padding_idx=src_pad_id, max_len=max_len, dropout=dropout
        )
        self.tgt_embed = TransformerEmbedding(
            tgt_vocab_size, d_model, padding_idx=tgt_pad_id, max_len=max_len, dropout=dropout
        )
        self.encoder = Encoder(num_stacks, d_model, d_hid, num_heads, d_k, d_v, dropout=dropout)
        self.decoder = Decoder(num_stacks, d_model, d_hid, num_heads, d_k, d_v, dropout=dropout)
        self.projection = nn.Linear(d_model, tgt_vocab_size)

        if tie_weights:
            # share the target embedding matrix with the output projection
            self.projection.weight = self.tgt_embed.token_embed.weight

    @staticmethod
    def make_pad_mask(input_ids: torch.Tensor, pad_id) -> torch.Tensor:
        """(batch, seq_len) -> (batch, 1, 1, seq_len); True = real token, False = pad."""
        return (input_ids != pad_id).unsqueeze(1).unsqueeze(2)

    def forward(self, src_input, tgt_input, src_pad_mask=None, tgt_pad_mask=None):
        # src_input, tgt_input: (batch, seq_len) token ids
        if src_pad_mask is None and self.src_pad_id is not None:
            src_pad_mask = self.make_pad_mask(src_input, self.src_pad_id)
        if tgt_pad_mask is None and self.tgt_pad_id is not None:
            tgt_pad_mask = self.make_pad_mask(tgt_input, self.tgt_pad_id)

        enc_output = self.encoder(self.src_embed(src_input), src_pad_mask)
        dec_output = self.decoder(
            self.tgt_embed(tgt_input), enc_output, tgt_pad_mask, src_pad_mask
        )
        return self.projection(dec_output)

    def get_attention_weights(self):
        """Return {module_name: (batch, heads, len_q, len_kv)} from the last forward pass."""
        weights = {}
        for name, module in self.named_modules():
            if getattr(module, "attn_weights", None) is not None:
                weights[name] = module.attn_weights
        return weights
