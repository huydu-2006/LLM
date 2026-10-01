# %%
"""Train thu mot Transformer nho tren cau tieng Viet co nghia, roi visualize
attention weight matrix cua moi layer / moi head.

Bai toan: dao nguoc thu tu tu trong cau (--task reverse) hoac copy (--task copy).
Cau demo duoc giu rieng (khong cho model hoc) de ket qua trung thuc.

Chay: python Visualize.py
      python Visualize.py --demo "tôi khá đọc sách" --steps 3000
Ket qua: attention_encoder.png, attention_decoder.png, attention_cross.png
"""

import argparse
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from Model import Transformer

PAD_TOKEN, BOS_TOKEN, EOS_TOKEN = "<pad>", "<bos>", "<eos>"
SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN)


# ------------------------------------------------------------------- corpus
SUBJECTS = ["tôi", "em", "anh", "chị", "bạn"]
ADVERBS = ["rất", "khá"]
VERBS = ["thích", "yêu", "học", "đọc", "viết", "chơi"]
NOUNS = ["toán", "văn", "nhạc", "sách", "bóng", "cờ"]


def build_corpus():
    """540 cau tieng Viet don gian tu mot vocab nho (tu lap lai nhieu)."""
    sentences = []
    for subj in SUBJECTS:
        for verb in VERBS:
            for noun in NOUNS:
                sentences.append(f"{subj} {verb} {noun}")
                for adv in ADVERBS:
                    sentences.append(f"{subj} {adv} {verb} {noun}")
    return sentences


def build_vocab(sentences):
    words = sorted({w for s in sentences for w in s.split()})
    itos = list(SPECIAL_TOKENS) + words
    stoi = {w: i for i, w in enumerate(itos)}
    return itos, stoi


def make_pair(text, task):
    """Cau goc -> (tgt input tokens, tgt output tokens)."""
    words = text.split()
    result = words if task == "copy" else words[::-1]
    return [BOS_TOKEN] + result, result + [EOS_TOKEN]


# ------------------------------------------------------------------ batching
def make_batch(sentences, stoi, task="reverse", device="cpu"):
    src_tokens = [s.split() for s in sentences]
    tgt_in_tokens, tgt_out_tokens = [], []
    for s in sentences:
        tgt_in, tgt_out = make_pair(s, task)
        tgt_in_tokens.append(tgt_in)
        tgt_out_tokens.append(tgt_out)

    def encode(token_lists):
        ids = [[stoi[tok] for tok in toks] for toks in token_lists]
        max_len = max(len(x) for x in ids)
        padded = [x + [stoi[PAD_TOKEN]] * (max_len - len(x)) for x in ids]
        return torch.tensor(padded, dtype=torch.long, device=device)

    return encode(src_tokens), encode(tgt_in_tokens), encode(tgt_out_tokens)


# ------------------------------------------------------------------ training
def train(model, samples, stoi, steps, task="reverse", batch_size=64, lr=1e-3):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=stoi[PAD_TOKEN])
    model.train()
    for step in range(1, steps + 1):
        batch = random.sample(samples, batch_size)
        src, tgt_in, tgt_out = make_batch(batch, stoi, task, device)
        logits = model(src, tgt_in)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 250 == 0 or step == steps:
            print(f"step {step:4d} | loss {loss.item():.4f}")


def predict(model, sentences, itos, stoi, task):
    """Tra ve list ket qua du doan (list token) cho tung cau."""
    src, tgt_in, _ = make_batch(sentences, stoi, task)
    model.eval()
    with torch.no_grad():
        pred = model(src, tgt_in).argmax(-1)
    results = []
    for j, sent in enumerate(sentences):
        expected = make_pair(sent, task)[1]
        got = [itos[t] for t in pred[j].tolist()[: len(expected)]]
        results.append(got)
    return results


# ----------------------------------------------------------------- visualize
def plot_group(weights, query_tokens, key_tokens, title, path):
    """Ve 1 figure: moi hang = 1 layer, moi cot = 1 head.

    weights: {layer_idx: (batch=1, heads, len_q, len_kv)}
    """
    layers = sorted(weights.keys())
    n_rows = len(layers)
    n_heads = weights[layers[0]].size(1)

    fig, axes = plt.subplots(
        n_rows, n_heads, figsize=(2.6 * n_heads, 2.6 * n_rows), squeeze=False
    )
    for r, layer in enumerate(layers):
        w = weights[layer][0]  # (heads, len_q, len_kv) cua cau dau tien trong batch
        for c in range(n_heads):
            ax = axes[r][c]
            ax.imshow(w[c].cpu(), cmap="Blues", vmin=0.0, vmax=w[c].max().item())
            ax.set_title(f"layer {layer} | head {c}", fontsize=9)
            ax.set_xticks(range(len(key_tokens)))
            ax.set_xticklabels(key_tokens, rotation=90, fontsize=8)
            ax.set_yticks(range(len(query_tokens)))
            ax.set_yticklabels(query_tokens, fontsize=8)
            if c == 0:
                ax.set_ylabel("query token", fontsize=9)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"da luu {path}")
    plt.close(fig)


def plot_attention(model, src, tgt_in, src_tokens, tgt_tokens):
    weights = model.get_attention_weights()
    enc_self, dec_self, dec_cross = {}, {}, {}
    for name, w in weights.items():
        layer_idx = name.split(".")[2]  # vd: "decoder.layers.0.cross_attn.attention"
        if name.startswith("encoder"):
            enc_self[layer_idx] = w
        elif "self_attn" in name:
            dec_self[layer_idx] = w
        elif "cross_attn" in name:
            dec_cross[layer_idx] = w

    plot_group(enc_self, src_tokens, src_tokens,
               "Encoder self-attention", "attention_encoder.png")
    plot_group(dec_self, tgt_tokens, tgt_tokens,
               "Decoder self-attention (causal)", "attention_decoder.png")
    plot_group(dec_cross, tgt_tokens, src_tokens,
               "Decoder cross-attention (query=tgt, key=src)", "attention_cross.png")


# ---------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2500, help="so buoc train")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--task", choices=["reverse", "copy"], default="reverse")
    parser.add_argument("--demo", type=str, default="anh rất thích nhạc",
                        help="cau de visualize (duoc giu rieng, khong train)")
    parser.add_argument("--holdout", type=int, default=15,
                        help="so cau giu rieng de kiem tra tong quat hoa")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = build_corpus()
    if args.demo not in samples:
        raise SystemExit(f"cau demo khong co trong corpus: {args.demo!r}")

    holdout = random.sample(samples, args.holdout)
    if args.demo not in holdout:
        holdout.append(args.demo)  # luon giu cau demo rieng
    train_samples = [s for s in samples if s not in holdout]
    itos, stoi = build_vocab(samples)

    model = Transformer(
        src_vocab_size=len(itos),
        tgt_vocab_size=len(itos),
        num_stacks=2,
        d_model=64,
        d_hid=128,
        num_heads=4,
        d_k=16,
        d_v=16,
        dropout=0.1,
        src_pad_id=stoi[PAD_TOKEN],
        tgt_pad_id=stoi[PAD_TOKEN],
    )
    print(f"corpus: {len(samples)} cau | train: {len(train_samples)} "
          f"| giu rieng: {len(holdout)} | vocab: {len(itos)} tu")
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    train(model, train_samples, stoi, steps=args.steps, task=args.task, lr=args.lr)

    # kiem tra tong quat hoa tren cac cau chua tung thay
    results = predict(model, holdout, itos, stoi, args.task)
    n_ok = sum(
        got == make_pair(sent, args.task)[1] for sent, got in zip(holdout, results)
    )
    print(f"test  : {n_ok}/{len(holdout)} cau chua tung thay duoc doan dung")

    # du doan + visualize cau demo (cung nam trong tap giu rieng)
    demo = args.demo
    src, tgt_in, _ = make_batch([demo], stoi, args.task)
    model.eval()
    with torch.no_grad():
        pred_ids = model(src, tgt_in).argmax(-1)[0].tolist()
    print(f"demo  : src = {demo}")
    print(f"        model doan: {' '.join(itos[i] for i in pred_ids)}")

    tgt_words = make_pair(demo, args.task)[0]  # [<bos>] + cac tu dich
    plot_attention(model, src, tgt_in, demo.split(), tgt_words)


if __name__ == "__main__":
    main()
