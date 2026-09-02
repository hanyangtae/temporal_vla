"""특징공간 비교 read 배터리 (stdout TSV). usage: python - <slug> [ae_encoder.pt]

공간: raw-1536 / PCA-16 (+ AE-16, encoder 주어지면). 혼합 scene 한정.
지표(공간별): condg_margin(릿지) / gauss_contrast(클래스별 조건부 가우시안 log-우도비;
16차원=full cov, raw=diag) / subspace_resid / len_only. 고정B·held-out ep·5-seed 중앙값
+ scene 내 순위(within) 병기.
"""
import json, os, sys
import numpy as np

SLUG = sys.argv[1]
AE = sys.argv[2] if len(sys.argv) > 2 else None
BASE = os.path.expanduser("~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase/segB_v2/")
d = np.load(BASE + SLUG + ".npz", allow_pickle=True)
meta = json.loads(str(d["meta_json"]))
CB = {v: k for k, v in meta["phase_codebook"].items()}
X_raw = d["X"].astype(np.float32); P_all = d["P"].astype(np.float64)
pc, ep, sc, su = d["phase_code"], d["ep_id"], d["scene"], d["succ"]

ep_ids0 = np.unique(ep)
ep_meta = {e: (int(sc[ep == e][0]), int(su[ep == e][0])) for e in ep_ids0}
sc_cnt = {}
for e in ep_ids0:
    s_, u = ep_meta[e]
    sc_cnt.setdefault(s_, [0, 0])[u] += 1
MIXED = {s_ for s_, v in sc_cnt.items() if v[0] >= 1 and v[1] >= 1}
ep_ids = np.array([e for e in ep_ids0 if ep_meta[e][0] in MIXED])

AE_Z = None
if AE:
    import torch
    enc = torch.load(AE, map_location="cpu")  # 계약 확정 후 구현
    raise SystemExit("AE 경로: 입력 계약 수신 후 구현")

def auroc(pos, neg):
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    s = np.concatenate([pos, neg]); r = s.argsort().argsort().astype(np.float64) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def ridge(P, X):
    lam = 1e-3 * len(P)
    return np.linalg.solve(P.T @ P + lam * np.eye(P.shape[1]), P.T @ X)

def gauss_fit(X, full):
    mu = X.mean(0)
    if full:
        C = np.cov(X.T) + 1e-3 * np.trace(np.cov(X.T)) / X.shape[1] * np.eye(X.shape[1])
        sign, logdet = np.linalg.slogdet(C)
        Ci = np.linalg.inv(C)
        return mu, Ci, logdet
    v = X.var(0) + 1e-6
    return mu, 1.0 / v, float(np.log(v).sum())

def gauss_ll(X, prm, full):
    mu, Ci, logdet = prm
    D = X - mu
    if full:
        return -0.5 * ((D @ Ci) * D).sum(1) - 0.5 * logdet
    return -0.5 * (D * D * Ci).sum(1) - 0.5 * logdet

for pcode in sorted(set(pc.tolist())):
    phname = CB[pcode]
    rows = []
    for e in ep_ids:
        idx = np.where((ep == e) & (pc == pcode))[0]
        if len(idx) >= 4:
            rows.append((e, ep_meta[e][0], ep_meta[e][1], idx))
    ns = sum(r[2] for r in rows); nf = len(rows) - ns
    if ns < 10 or nf < 10:
        continue
    out = {}
    within = {}
    for seed in range(5):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(rows))
        tr = set(order[: int(len(rows) * 0.6)].tolist())
        te = [i for i in range(len(rows)) if i not in tr]
        if sum(rows[i][2] for i in tr) < 4 or sum(1 - rows[i][2] for i in tr) < 4:
            continue
        stats = {}
        for s_ in set(r[1] for r in rows):
            g = [rows[i] for i in tr if rows[i][1] == s_] or [r for r in rows if r[1] == s_]
            gi_s = np.concatenate([r[3] for r in g if r[2] == 1]) if any(
                r[2] == 1 for r in g) else np.concatenate([r[3] for r in g])
            gi_a = np.concatenate([r[3] for r in g])
            stats[s_] = (X_raw[gi_s].mean(0), P_all[gi_a].mean(0), P_all[gi_a].std(0) + 1e-8)
        def prep(i):
            e, s_, u, idx = rows[i]
            mh, mp, sp = stats[s_]
            return u, X_raw[idx] - mh, (P_all[idx] - mp) / sp
        succ_dw = sorted(len(rows[i][3]) for i in tr if rows[i][2] == 1)
        B = max(3, succ_dw[len(succ_dw) // 4])
        # train 행렬 (고정B 창)
        def cls(u):
            Xs_, Ps_ = [], []
            for i in tr:
                if rows[i][2] == u:
                    _, Xi, Pi = prep(i)
                    Xs_.append(Xi[:B]); Ps_.append(Pi[:B])
            return np.concatenate(Xs_), np.concatenate(Ps_)
        Xs_tr, Ps_tr = cls(1); Xf_tr, Pf_tr = cls(0)
        # PCA-16 basis (train 전체)
        Xall_tr = np.concatenate([Xs_tr, Xf_tr])
        mu0 = Xall_tr.mean(0)
        _, S_, Vt = np.linalg.svd(Xall_tr - mu0, full_matrices=False)
        V16 = Vt[:16]
        spaces = {"raw": (lambda Z: Z, True if False else False),
                  "pca16": (lambda Z: (Z - mu0) @ V16.T, True)}
        for sp_name, (proj, full_cov) in spaces.items():
            Zs, Zf = proj(Xs_tr), proj(Xf_tr)
            Ws_, Wf_ = ridge(Ps_tr, Zs), ridge(Pf_tr, Zf)
            gs, gf = gauss_fit(Zs, full_cov), gauss_fit(Zf, full_cov)
            mu_s = Zs.mean(0)
            _, Ss_, Vs_ = np.linalg.svd(Zs - mu_s, full_matrices=False)
            k = min(int(np.searchsorted(np.cumsum(Ss_**2) / (Ss_**2).sum(), 0.9)) + 1, len(Ss_) - 1)
            Vsub = Vs_[:k]
            vals = {m: {0: [], 1: []} for m in ["margin", "gauss", "subresid", "len"]}
            for i in te:
                u, Xi, Pi = prep(i)
                if len(Xi) < B:
                    continue
                Zb, Pb = proj(Xi[:B]), Pi[:B]
                m_ = ((Zb - Pb @ Ws_) ** 2).sum(1) - ((Zb - Pb @ Wf_) ** 2).sum(1)
                g_ = gauss_ll(Zb, gf, full_cov) - gauss_ll(Zb, gs, full_cov)
                r_ = Zb - mu_s
                sr = (np.linalg.norm(r_ - r_ @ Vsub.T @ Vsub, axis=1)**2
                      / (np.linalg.norm(r_, axis=1)**2 + 1e-8))
                for mname, v in [("margin", m_.mean()), ("gauss", g_.mean()),
                                 ("subresid", sr.mean()), ("len", float(len(Xi)))]:
                    vals[mname][u].append(float(v))
                key = (sp_name, rows[i][1])
                w = within.setdefault(key, {m: {0: [], 1: []} for m in ["margin", "gauss"]})
                w["margin"][u].append(float(m_.mean()))
                w["gauss"][u].append(float(g_.mean()))
            for mname in vals:
                out.setdefault((sp_name, mname), []).append(
                    auroc(np.array(vals[mname][0]), np.array(vals[mname][1])))
    def med(a):
        a = [x for x in a if not np.isnan(x)]
        return float(np.median(a)) if a else float("nan")
    for (sp_name, mname), arr in sorted(out.items()):
        print(f"{SLUG}\t{phname}\t{sp_name}\t{mname}\t{med(arr):.3f}\t{ns}\t{nf}")
    # scene 내 순위 (혼합 scene, 클래스 3+씩)
    for sp_name in ["raw", "pca16"]:
        for mname in ["margin", "gauss"]:
            ws = []
            for (spn, s_), w in within.items():
                if spn != sp_name:
                    continue
                if len(w[mname][0]) >= 3 and len(w[mname][1]) >= 3:
                    ws.append(auroc(np.array(w[mname][0]), np.array(w[mname][1])))
            if ws:
                print(f"{SLUG}\t{phname}\t{sp_name}\t{mname}_within\t{float(np.median(ws)):.3f}\t{ns}\t{nf}")
print(f"SPACE_DONE {SLUG}")
