"""CPU/fp32 logit-lens projection from cached value_vectors.pkl.

Faithful to src/ffn_value_vectors/extract.py (same math: value_vectors @ lm_head.T,
topk, action-token decode), but:
  - runs on CPU fp32 (the GPU OOMs: 14GB model + 2.7GB vectors > 16GB),
  - loads only lm_head from safetensors (no 14GB model reload),
  - precomputes a per-vocab decode table (avoids ~10M tokenizer.decode calls).
Writes top_tokens_output.txt + top_tokens.pkl in the same format/location.
"""
import os, glob, json, pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoProcessor
from safetensors import safe_open
from ffn_value_vectors.action_tokenizer import ActionTokenizer

MODEL = "openvla/openvla-7b-finetuned-libero-10"
OUT = "/home/dongkyu/pkt_ws/mechanistic-steering-vlas/openvla/ffn_value_vectors/artifacts/ffn_value_projection"
CKPT = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--openvla--openvla-7b-finetuned-libero-10/snapshots/*"))[0]
TOPK = 30
torch.set_num_threads(16)

# 1) cached value vectors [N, 4096]
with open(os.path.join(OUT, "value_vectors.pkl"), "rb") as f:
    vv = pickle.load(f).float()
print("value_vectors:", tuple(vv.shape), vv.dtype)

# 2) lm_head weight [V, 4096] from safetensors (no full-model load)
idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))
key = "language_model.lm_head.weight"
shard = idx["weight_map"][key]
with safe_open(os.path.join(CKPT, shard), framework="pt") as f:
    W = f.get_tensor(key).float()
V = W.shape[0]
print("lm_head:", tuple(W.shape))

# 3) tokenizer + action-token decode table
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
vocab_size = proc.tokenizer.vocab_size
abins = 256
astart, aend = vocab_size - abins, vocab_size
atok = ActionTokenizer(vocab_size=vocab_size, bins=abins, min_action=-1.0, max_action=1.0)
print("vocab_size:", vocab_size, "action range:", astart, aend)

decoded = [None] * V
for i in range(V):
    if astart <= i < aend:
        av = atok.decode_token_ids_to_actions(np.array([i]))[0]
        decoded[i] = f"[action: {av:.3f}]"
    else:
        try:
            decoded[i] = repr(tok.decode([i]))
        except Exception:
            decoded[i] = f"<id{i}>"

# 4) batched projection + topk
BS = 8192
lines, top_ids_all, top_txt_all = [], [], []
n = vv.shape[0]
with torch.no_grad():
    for s in range(0, n, BS):
        logits = vv[s:s + BS] @ W.T
        top = torch.topk(logits, TOPK, dim=1).indices
        for r, row in enumerate(top.tolist()):
            toks = [decoded[t] for t in row]
            lines.append(f"[{s + r:04d}] {', '.join(toks)}")
            top_ids_all.append(row)
            top_txt_all.append(toks)
        if (s // BS) % 5 == 0:
            print(f"  projected {s + BS}/{n}", flush=True)

with open(os.path.join(OUT, "top_tokens_output.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
with open(os.path.join(OUT, "top_tokens.pkl"), "wb") as f:
    pickle.dump((torch.tensor(top_ids_all), top_txt_all), f)
print("WROTE", len(lines), "lines ->", os.path.join(OUT, "top_tokens_output.txt"))
