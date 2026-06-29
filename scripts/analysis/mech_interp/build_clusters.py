"""Verify shipped 'up' clusters against our projection, and build fast/slow clusters.

flat_idx (= line number in top_tokens_output.txt) maps to (layer, neuron) via
//11008, %11008  -- exactly what hooks.apply_gate_proj_hooks expects.
"""
import re, yaml

TXT = "/home/dongkyu/pkt_ws/mechanistic-steering-vlas/openvla/ffn_value_vectors/artifacts/ffn_value_projection/top_tokens_output.txt"
SHIPPED = "/home/dongkyu/pkt_ws/mechanistic-steering-vlas/openvla/libero_experiments/configs/interventions/dictionaries.yaml"

tok_re = re.compile(r"\[(\d+)\]\s*(.*)")
def parse_tokens(rest):
    # split on ", " but keep [action: x] and 'tok' items
    items = [x.strip() for x in rest.split(", ")]
    out = []
    for it in items:
        if it.startswith("[action"):
            out.append(("action", it))
        else:
            s = it.strip()
            if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
                s = s[1:-1]
            out.append(("tok", s))
    return out

# load all lines -> dict flat_idx -> list[(kind,str)]
rows = {}
with open(TXT) as f:
    for line in f:
        m = tok_re.match(line)
        if not m:
            continue
        idx = int(m.group(1))
        rows[idx] = parse_tokens(m.group(2))

print("parsed", len(rows), "rows")

# ---- (1) verify shipped up_10 / up_20 against our projection ----
shipped = yaml.safe_load(open(SHIPPED))
for name in ["up_10", "up_20"]:
    print(f"\n=== shipped {name}: our top-5 tokens for those flat indices ===")
    for flat in list(shipped[name].keys())[:6]:
        toks = rows.get(int(flat), [])
        top5 = [t[1] for t in toks[:5]]
        print(f"  flat {flat} (layer {int(flat)//11008}, neuron {int(flat)%11008}): {top5}")

# ---- (2) build fast / slow clusters ----
def norm(s):
    return s.lstrip("▁ ").lower()  # strip leading ▁ / space

def score(toks, prefixes, topk=10):
    cnt, first = 0, 99
    for r, (kind, s) in enumerate(toks[:topk]):
        if kind != "tok":
            continue
        n = norm(s)
        if any(n.startswith(p) for p in prefixes):
            cnt += 1
            first = min(first, r)
    return cnt, -first  # more matches better; earlier first-match better

CONCEPTS = {
    "fast": ["fast", "quick", "rapid", "swift"],
    "slow": ["slow"],
    "up":   ["up"],
}
built = {}
for concept, prefixes in CONCEPTS.items():
    scored = []
    for idx, toks in rows.items():
        c, nf = score(toks, prefixes)
        if c >= 1 and toks and toks[0][0] == "tok" and norm(toks[0][1]).startswith(tuple(prefixes)):
            scored.append((c, nf, idx))
    scored.sort(reverse=True)
    for size in (10, 20):
        chosen = [idx for _, _, idx in scored[:size]]
        built[f"{concept}_{size}"] = {int(i): [t[1] for t in rows[i][:10]] for i in chosen}
    print(f"\n=== {concept}: {len(scored)} candidate neurons (top-token startswith {prefixes}) ===")
    for c, nf, idx in scored[:10]:
        print(f"  flat {idx} (L{idx//11008}) match={c}: {[t[1] for t in rows[idx][:6]]}")

# write fast/slow clusters to a new dict file (keep up from shipped for fidelity)
out = {k: built[k] for k in ["fast_10", "fast_20", "slow_10", "slow_20"]}
OUTYAML = "/tmp/claude-1004/-home-dongkyu-pkt-ws-temporal-vla/80d5a2ad-9c6d-4d8d-a42d-f046988e25a8/scratchpad/fastslow_clusters.yaml"
with open(OUTYAML, "w") as f:
    yaml.safe_dump(out, f, allow_unicode=True, sort_keys=True)
print("\nWROTE", OUTYAML, "concepts:", list(out.keys()), {k: len(v) for k, v in out.items()})
