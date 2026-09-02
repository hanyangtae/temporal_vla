"""scene-전용 fit 재검증 — AE-16 가우시안 대조 (stdout). usage: python - <slug> <ae.pt>

혼합 scene(>=3s/3f)별 LOO: A16 = scene-전용 fit(그 scene 데이터만, AE-16 gauss)
vs B16 = task-pooled fit(AE-16 gauss). 이전 라운드(1536d 릿지 margin)와 동일 절차.
"""
import json, os, sys
import numpy as np

SLUG, AE = sys.argv[1], sys.argv[2]
BASE = os.path.expanduser("~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase/segB_v2/")
d = np.load(BASE + SLUG + ".npz", allow_pickle=True)
meta = json.loads(str(d["meta_json"]))
CB = {v: k for k, v in meta["phase_codebook"].items()}
X_raw = d["X"].astype(np.float32)
pc, ep, sc, su = d["phase_code"], d["ep_id"], d["scene"], d["succ"]

import sys as _sys
import torch
_sys.path.insert(0, "/tmp/kai_lab/repo/scripts/analysis/grid_phase")
import ae_cluster as ac
ckpt = torch.load(AE, map_location="cpu")
model = ac.BaseAE(ac.Encoder(1536, 16), ac.Decoder(16, 1536))
model.load_state_dict(ckpt["state_dict"]); model.eval()
mu_g = np.asarray(ckpt["scaler"]["mu"], np.float32); std_g = float(ckpt["scaler"]["scalar_std"])
Z = []
with torch.no_grad():
    for i in range(0, len(X_raw), 8192):
        Z.append(model.latent(torch.from_numpy((X_raw[i:i+8192] - mu_g) / std_g)).numpy())
X = np.concatenate(Z).astype(np.float64)

def auroc(pos, neg):
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    s = np.concatenate([pos, neg]); r = s.argsort().argsort().astype(np.float64) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def gfit(Xc):
    mu = Xc.mean(0)
    C = np.cov(Xc.T) + 1e-3 * np.trace(np.cov(Xc.T)) / Xc.shape[1] * np.eye(Xc.shape[1])
    return mu, np.linalg.inv(C), np.linalg.slogdet(C)[1]

def gll(Xc, prm):
    mu, Ci, ld = prm
    D = Xc - mu
    return -0.5 * ((D @ Ci) * D).sum(1) - 0.5 * ld

ep_ids = np.unique(ep)
ep_meta = {e: (int(sc[ep == e][0]), int(su[ep == e][0])) for e in ep_ids}
for pcode in sorted(set(pc.tolist())):
    phname = CB[pcode]
    rows = {}
    for e in ep_ids:
        idx = np.where((ep == e) & (pc == pcode))[0]
        if len(idx) >= 4:
            rows[e] = idx
    elist = sorted(rows)
    rng = np.random.default_rng(0)
    tr = set(rng.permutation(elist)[: int(len(elist) * 0.6)].tolist())
    if sum(ep_meta[e][1] for e in tr) < 4 or sum(1 - ep_meta[e][1] for e in tr) < 4:
        continue
    def cls_mat(eps_, u, B=None):
        xs = [X[rows[e]][:B] if B else X[rows[e]] for e in eps_ if ep_meta[e][1] == u]
        return np.concatenate(xs) if xs else np.zeros((0, 16))
    by_scene = {}
    for e in rows:
        by_scene.setdefault(ep_meta[e][0], []).append(e)
    # pooled fit (scene-중심화: scene 성공 평균 빼기)
    cent = {}
    for s_, eps_ in by_scene.items():
        g = [e for e in eps_ if e in tr and ep_meta[e][1] == 1] or [e for e in eps_ if ep_meta[e][1] == 1] or eps_
        cent[s_] = np.concatenate([X[rows[e]] for e in g]).mean(0)
    def XC(e, B):
        return X[rows[e]][:B] - cent[ep_meta[e][0]]
    dwp = sorted(len(rows[e]) for e in tr if ep_meta[e][1] == 1)
    Bp = max(3, dwp[len(dwp) // 4])
    Gs_p = gfit(np.concatenate([XC(e, Bp) for e in tr if ep_meta[e][1] == 1]))
    Gf_p = gfit(np.concatenate([XC(e, Bp) for e in tr if ep_meta[e][1] == 0]))
    for s_, eps_ in sorted(by_scene.items()):
        ns_ = sum(ep_meta[e][1] for e in eps_); nf_ = len(eps_) - ns_
        if ns_ < 3 or nf_ < 3:
            continue
        dw = sorted(len(rows[e]) for e in eps_ if ep_meta[e][1] == 1)
        B = max(3, dw[len(dw) // 4])
        vals = {k: {0: [], 1: []} for k in "AB"}
        for e in eps_:
            train = [x for x in eps_ if x != e]
            if sum(ep_meta[x][1] for x in train) < 2 or sum(1 - ep_meta[x][1] for x in train) < 2:
                continue
            u = ep_meta[e][1]
            Xe = X[rows[e]][:B]
            if len(X[rows[e]]) < B:
                continue
            mu_sc = np.concatenate([X[rows[x]] for x in train if ep_meta[x][1] == 1]).mean(0)
            try:
                Gs = gfit(np.concatenate([X[rows[x]][:B] for x in train if ep_meta[x][1] == 1]) - mu_sc)
                Gf = gfit(np.concatenate([X[rows[x]][:B] for x in train if ep_meta[x][1] == 0]) - mu_sc)
            except np.linalg.LinAlgError:
                continue
            m = float((gll(Xe - mu_sc, Gf) - gll(Xe - mu_sc, Gs)).mean())
            vals["A"][u].append(m)
            Xc_p = X[rows[e]][:Bp] - cent[ep_meta[e][0]]
            if len(Xc_p) >= 3:
                vals["B"][u].append(float((gll(Xc_p, Gf_p) - gll(Xc_p, Gs_p)).mean()))
        a = auroc(np.array(vals["A"][0]), np.array(vals["A"][1]))
        b = auroc(np.array(vals["B"][0]), np.array(vals["B"][1]))
        print(f"{SLUG}\t{phname}\ts{s_}\t{ns_}\t{nf_}\t{a:.2f}\t{b:.2f}")
print(f"SITU16_DONE {SLUG}")
