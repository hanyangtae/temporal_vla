"""scene 내 succ/fail 분리 '존재 증명' — 완전 held-out (stdout).

fit = train episode(60%)만, scene-중심화 gauss-16(AE) + raw 릿지 margin.
채점 = test episode만, scene별 AUROC (test 내 s>=2/f>=2 scene 한정), 5-seed.
길이단독도 같은 test 집합에서. usage: python - <slug> <ae.pt>
"""
import json, os, sys
import numpy as np

SLUG, AE = sys.argv[1], sys.argv[2]
BASE = os.path.expanduser("~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase/segB_v2/")
d = np.load(BASE + SLUG + ".npz", allow_pickle=True)
meta = json.loads(str(d["meta_json"]))
CB = {v: k for k, v in meta["phase_codebook"].items()}
X_raw = d["X"].astype(np.float32); P_all = d["P"].astype(np.float64)
pc, ep, sc, su = d["phase_code"], d["ep_id"], d["scene"], d["succ"]

import torch, sys as _s
_s.path.insert(0, "/tmp/kai_lab/repo/scripts/analysis/grid_phase")
import ae_cluster as ac
ck = torch.load(AE, map_location="cpu")
m_ = ac.BaseAE(ac.Encoder(1536, 16), ac.Decoder(16, 1536))
m_.load_state_dict(ck["state_dict"]); m_.eval()
mu_g = np.asarray(ck["scaler"]["mu"], np.float32); sd_g = float(ck["scaler"]["scalar_std"])
Z = []
with torch.no_grad():
    for i in range(0, len(X_raw), 8192):
        Z.append(m_.latent(torch.from_numpy((X_raw[i:i+8192] - mu_g) / sd_g)).numpy())
X16 = np.concatenate(Z).astype(np.float64)

def auroc(pos, neg):
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    s = np.concatenate([pos, neg]); r = s.argsort().argsort().astype(np.float64) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def gfit(Xc):
    mu = Xc.mean(0)
    C = np.cov(Xc.T) + 1e-3 * np.trace(np.cov(Xc.T)) / Xc.shape[1] * np.eye(Xc.shape[1])
    return mu, np.linalg.inv(C), np.linalg.slogdet(C)[1]

def gll(Xc, p):
    D = Xc - p[0]
    return -0.5 * ((D @ p[1]) * D).sum(1) - 0.5 * p[2]

ep_ids = np.unique(ep)
em = {e: (int(sc[ep == e][0]), int(su[ep == e][0])) for e in ep_ids}
for pcode in sorted(set(pc.tolist())):
    ph = CB[pcode]
    rows = {e: np.where((ep == e) & (pc == pcode))[0] for e in ep_ids}
    rows = {e: i for e, i in rows.items() if len(i) >= 4}
    elist = sorted(rows)
    ns = sum(em[e][1] for e in elist); nf = len(elist) - ns
    if ns < 10 or nf < 10:
        continue
    per_scene = {}
    for seed in range(5):
        rng = np.random.default_rng(seed)
        order = rng.permutation(elist)
        tr = set(order[: int(len(elist) * 0.6)].tolist())
        te = [e for e in elist if e not in tr]
        if sum(em[e][1] for e in tr) < 4 or sum(1 - em[e][1] for e in tr) < 4:
            continue
        cent = {}
        for s_ in set(em[e][0] for e in elist):
            g = [e for e in tr if em[e][0] == s_ and em[e][1] == 1] or \
                [e for e in elist if em[e][0] == s_ and em[e][1] == 1] or \
                [e for e in elist if em[e][0] == s_]
            cent[s_] = np.concatenate([X16[rows[e]] for e in g]).mean(0)
        dw = sorted(len(rows[e]) for e in tr if em[e][1] == 1)
        B = max(3, dw[len(dw) // 4])
        XC = lambda e: X16[rows[e]][:B] - cent[em[e][0]]
        Gs = gfit(np.concatenate([XC(e) for e in tr if em[e][1] == 1]))
        Gf = gfit(np.concatenate([XC(e) for e in tr if em[e][1] == 0]))
        by = {}
        for e in te:
            if len(rows[e]) < B:
                continue
            g_ = float((gll(XC(e), Gf) - gll(XC(e), Gs)).mean())
            by.setdefault(em[e][0], {0: [], 1: []})[em[e][1]].append((g_, len(rows[e])))
        for s_, v in by.items():
            if len(v[0]) >= 2 and len(v[1]) >= 2:
                a = auroc(np.array([x[0] for x in v[0]]), np.array([x[0] for x in v[1]]))
                l = auroc(np.array([float(x[1]) for x in v[0]]), np.array([float(x[1]) for x in v[1]]))
                per_scene.setdefault(s_, []).append((a, l, len(v[1]), len(v[0])))
    for s_, arr in sorted(per_scene.items()):
        med_a = float(np.median([x[0] for x in arr]))
        med_l = float(np.median([x[1] for x in arr]))
        n_ = max(x[2] + x[3] for x in arr)
        print(f"{SLUG}\t{ph}\ts{s_}\t{med_a:.2f}\t{med_l:.2f}\t{len(arr)}seed\tmaxn{n_}")
print(f"CLEAN_DONE {SLUG}")
