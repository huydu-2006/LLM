# %%
# Vi du NHO NHAT ve "embedding nhan String"
# Chay thu: python mini_demo.py
#
# Y tuong chi co 3 buoc, doc tu tren xuong:
#   CHU ---(tu dien)---> SO ---(bang tra)---> VECTOR

import torch
import torch.nn as nn

texts = ["hello", "hi"]

# --- Buoc 1: lam "tu dien" doi chu thanh so ---
chars = sorted(set("".join(texts)))             # ['e', 'h', 'i', 'l', 'o']
char2id = {c: i for i, c in enumerate(chars)}   # {'e': 0, 'h': 1, ...}
print("tu dien chu -> so:", char2id)

# --- Buoc 2: nn.Embedding chinh la "bang tra" so -> vector ---
d_model = 4                                     # moi chu duoc bieu dien bang 4 so
embed = nn.Embedding(num_embeddings=len(chars), embedding_dim=d_model)

# --- Buoc 3: dung ---
for t in texts:
    ids = [char2id[c] for c in t]               # 'hello' -> [1, 0, 3, 3, 4]
    x = embed(torch.tensor(ids))                # tra bang: (so chu, d_model)
    print(f"{t!r:8} -> ids {ids} -> vectors {tuple(x.shape)}")
