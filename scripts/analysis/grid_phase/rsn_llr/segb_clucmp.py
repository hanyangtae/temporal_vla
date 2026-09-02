"""채점 신호 정면 비교 (stdout): LLR / proprio-잔차 LLR / cluster 소속 / hybrid / 길이.

프로토콜 = segb_clean(혼합 scene·고정B·완전 held-out test·5-seed 중앙값).
cluster 할당 = PCA64w -> AE16(상우) -> active 최근접 중심.
usage: python - <slug> <kai_ae.pt> <sw_model.pt> <sw_pca.npz>
"""
import json
import os
import sys

import numpy as np

SLUG, KAI, SWM, SWP = sys.argv[1:5]
BASE = os.path.expanduser("~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase/segB_v2/")
d = np.load(BASE + SLUG + ".npz", allow_pickle=True)
meta = json.loads(str(d["meta_json"]))
CB = {v: k for k, v in meta["phase_codebook"].items()}
X_raw = d["X"].astype(np.float32)
P_all = d["P"].astype(np.float64)
pc, ep, sc, su = d["phase_code"], d["ep_id"], d["scene"], d["succ"]

import torch
sys.path.insert(0, "/tmp/kai_lab/repo/scripts/analysis/grid_phase")
import ae_cluster as ac

ck = torch.load(KAI, map_location="cpu")
mK = ac.BaseAE(ac.Encoder(1536, 16), ac.Decoder(16, 1536))
mK.load_state_dict(ck["state_dict"])
mK.eval()
muK = np.asarray(ck["scaler"]["mu"], np.float32)
sdK = float(ck["scaler"]["scalar_std"])
Z = []
with torch.no_grad():
    for i in range(0, len(X_raw), 8192):
        Z.append(mK.latent(torch.from_numpy((X_raw[i:i + 8192] - muK) / sdK)).numpy())
X16 = np.concatenate(Z).astype(np.float64)

# ── 상우 cluster 할당 ──
import torch.nn as nn

sw = torch.load(SWM, map_location="cpu")
pca = dict(np.load(SWP).items())
z64 = ((X_raw - pca["mu"].astype(np.float32)) @ pca["V"].astype(np.float32).T) \
    / pca["sqrt_lam"].astype(np.float32)
sd_ = sw["state_dict"]
head_w = [v for k, v in sd_.items() if k.endswith("head.weight")][0]
var_ = head_w.shape[0] == 32


class Enc(nn.Module):
    def __init__(self, din, dout, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.head = nn.Linear(hidden, dout)

    def forward(self, x):
        return self.head(self.net(x))


enc = Enc(z64.shape[1], 32 if var_ else 16)
enc.load_state_dict({k[4:]: v for k, v in sd_.items() if k.startswith("enc.")})
enc.eval()
with torch.no_grad():
    zz = []
    for i in range(0, len(z64), 8192):
        o = enc(torch.from_numpy(z64[i:i + 8192]))
        zz.append((o[:, :16] if var_ else o).numpy())
z16 = np.concatenate(zz)
cent = np.asarray(sw["kmeans_centers"], np.float32)
act = sw.get("active")
za = z16 if act is None else z16[:, np.asarray(act, bool)]
CL = np.empty(len(za), np.int64)
for i in range(0, len(za), 8192):
    D2 = ((za[i:i + 8192, None, :] - cent[None]) ** 2).sum(-1)
    CL[i:i + 8192] = D2.argmin(1)
NC = len(cent)
print(f"# {SLUG}: cluster 사용 {len(np.unique(CL))}/{NC}, variational={var_}")


def auroc(pos, neg):
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort().astype(np.float64) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def gfit(Xc):
    mu = Xc.mean(0)
    C = np.cov(Xc.T) + 1e-3 * np.trace(np.cov(Xc.T)) / Xc.shape[1] * np.eye(Xc.shape[1])
    return mu, np.linalg.inv(C), np.linalg.slogdet(C)[1]


def gll(Xc, p):
    D = Xc - p[0]
    return -0.5 * ((D @ p[1]) * D).sum(1) - 0.5 * p[2]


def ridge(P, X):
    lam = 1e-3 * len(P)
    return np.linalg.solve(P.T @ P + lam * np.eye(P.shape[1]), P.T @ X)


ep_ids = np.unique(ep)
em = {e: (int(sc[ep == e][0]), int(su[ep == e][0])) for e in ep_ids}
ARMS = ["llr", "llr_res", "clu", "hyb", "len"]
for pcode in sorted(set(pc.tolist())):
    ph = CB[pcode]
    rows = {e: np.where((ep == e) & (pc == pcode))[0] for e in ep_ids}
    rows = {e: i for e, i in rows.items() if len(i) >= 4}
    elist = sorted(rows)
    ns = sum(em[e][1] for e in elist)
    nf = len(elist) - ns
    if ns < 10 or nf < 10:
        continue
    pooled = {k: [] for k in ARMS}
    within = {k: {} for k in ARMS if k != "len"}
    for seed in range(5):
        rng = np.random.default_rng(seed)
        order = rng.permutation(elist)
        tr = set(order[: int(len(elist) * 0.6)].tolist())
        te = [e for e in elist if e not in tr]
        if sum(em[e][1] for e in tr) < 4 or sum(1 - em[e][1] for e in tr) < 4:
            continue
        centm, pstat = {}, {}
        for s_ in set(em[e][0] for e in elist):
            g = [e for e in tr if em[e][0] == s_ and em[e][1] == 1] or \
                [e for e in elist if em[e][0] == s_ and em[e][1] == 1] or \
                [e for e in elist if em[e][0] == s_]
            centm[s_] = np.concatenate([X16[rows[e]] for e in g]).mean(0)
            ga = [e for e in tr if em[e][0] == s_] or [e for e in elist if em[e][0] == s_]
            Pg = np.concatenate([P_all[rows[e]] for e in ga])
            pstat[s_] = (Pg.mean(0), Pg.std(0) + 1e-8)
        dw = sorted(len(rows[e]) for e in tr if em[e][1] == 1)
        B = max(3, dw[len(dw) // 4])

        def XC(e):
            return X16[rows[e]][:B] - centm[em[e][0]]

        def PC(e):
            mp, sp = pstat[em[e][0]]
            return (P_all[rows[e]][:B] - mp) / sp

        Xtr = np.concatenate([XC(e) for e in tr if len(rows[e]) >= B])
        Ptr = np.concatenate([PC(e) for e in tr if len(rows[e]) >= B])
        Wp = ridge(Ptr, Xtr)

        def RC(e):
            return XC(e) - PC(e) @ Wp

        Gs = gfit(np.concatenate([XC(e) for e in tr if em[e][1] == 1]))
        Gf = gfit(np.concatenate([XC(e) for e in tr if em[e][1] == 0]))
        Rs = gfit(np.concatenate([RC(e) for e in tr if em[e][1] == 1 and len(rows[e]) >= B]))
        Rf = gfit(np.concatenate([RC(e) for e in tr if em[e][1] == 0 and len(rows[e]) >= B]))
        cnt = np.ones((2, NC))
        for e in tr:
            for c in CL[rows[e]][:B]:
                cnt[em[e][1], c] += 1
        lo = np.log((cnt[0] / cnt[0].sum()) / (cnt[1] / cnt[1].sum()))

        def scores(e):
            sl = float((gll(XC(e), Gf) - gll(XC(e), Gs)).mean())
            sr = float((gll(RC(e), Rf) - gll(RC(e), Rs)).mean())
            scl = float(lo[CL[rows[e]][:B]].mean())
            return sl, sr, scl

        tr_s = np.array([scores(e) for e in tr if len(rows[e]) >= B])
        mu_t, sd_t = tr_s.mean(0), tr_s.std(0) + 1e-8
        by = {}
        vals = {k: {0: [], 1: []} for k in ARMS}
        for e in te:
            if len(rows[e]) < B:
                continue
            sl, sr, scl = scores(e)
            hy = (sl - mu_t[0]) / sd_t[0] + (scl - mu_t[2]) / sd_t[2]
            u = em[e][1]
            for k, v in [("llr", sl), ("llr_res", sr), ("clu", scl), ("hyb", hy),
                         ("len", float(len(rows[e])))]:
                vals[k][u].append(v)
            by.setdefault(em[e][0], {0: [], 1: []})[u].append((sl, sr, scl, hy))
        for k in vals:
            pooled[k].append(auroc(np.array(vals[k][0]), np.array(vals[k][1])))
        for s_, v in by.items():
            if len(v[0]) >= 2 and len(v[1]) >= 2:
                for ki, k in enumerate(["llr", "llr_res", "clu", "hyb"]):
                    within[k].setdefault(s_, []).append(
                        auroc(np.array([x[ki] for x in v[0]]),
                              np.array([x[ki] for x in v[1]])))

    def med(a):
        a = [x for x in a if not np.isnan(x)]
        return float(np.median(a)) if a else float("nan")

    w = {k: med([med(v) for v in within[k].values()]) for k in within}
    p_ = {k: med(pooled[k]) for k in ARMS}
    print(f"{SLUG}\t{ph}\tpooled " +
          " ".join(f"{k} {p_[k]:.2f}" for k in ARMS) +
          "\twithin " + " ".join(f"{k} {w[k]:.2f}" for k in w) + f"\t({ns}/{nf})")
print(f"CLUCMP_DONE {SLUG}")
