"""situation-전용 fit 검증 (stdout TSV): 혼합 scene(>=3s/3f)별 leave-one-episode-out.

비교 3열: A=scene-전용 fit condg margin / B=pooled fit(전 scene, 해당 scene 제외 안 함=스펙
그대로) margin / C=scene-전용 성공-부분공간 잔차. 전부 fixB(그 scene train 성공 dwell 25pct).
"""
import json, os, sys
import numpy as np

BASE = os.path.expanduser("~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase/segB_v2/")
SLUGS = ["OpenDrawer_left", "OpenDrawer_right", "DishwasherRack_out", "OvenRack_out",
         "PPCC_candle", "PPCC_bread", "PPCC_marshmallow", "PPCC_jug", "CoffeeSetupMug"]

def auroc(pos, neg):
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    s = np.concatenate([pos, neg]); r = s.argsort().argsort().astype(np.float64) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def ridge(P, X):
    lam = 1e-3 * len(P)
    return np.linalg.solve(P.T @ P + lam * np.eye(P.shape[1]), P.T @ X)

print("slug\tphase\tscene\tn_s\tn_f\tA_situfit\tB_pooledfit\tC_subresid\tlen_only\tB_budget")
for SLUG in SLUGS:
    d = np.load(BASE + SLUG + ".npz", allow_pickle=True)
    meta = json.loads(str(d["meta_json"]))
    CB = {v: k for k, v in meta["phase_codebook"].items()}
    X_all = d["X"].astype(np.float32); P_all = d["P"].astype(np.float64)
    pc, ep, sc, su = d["phase_code"], d["ep_id"], d["scene"], d["succ"]
    ep_ids = np.unique(ep)
    ep_meta = {e: (int(sc[ep == e][0]), int(su[ep == e][0])) for e in ep_ids}
    for pcode in sorted(set(pc.tolist())):
        phname = CB[pcode]
        # 에피소드별 index (phase 한정)
        rows = {}
        for e in ep_ids:
            idx = np.where((ep == e) & (pc == pcode))[0]
            if len(idx) >= 4:
                rows[e] = idx
        # pooled fit (전 episode 60%, seed 0 — 스펙: scene fit-노출 허용)
        rng = np.random.default_rng(0)
        elist = sorted(rows)
        tr = set(rng.permutation(elist)[: int(len(elist) * 0.6)].tolist())
        if sum(ep_meta[e][1] for e in tr) < 4 or sum(1 - ep_meta[e][1] for e in tr) < 4:
            continue
        stats = {}
        for s_ in set(v[0] for v in ep_meta.values()):
            g = [e for e in tr if ep_meta[e][0] == s_] or [e for e in rows if ep_meta[e][0] == s_]
            if not g:
                continue
            gi_s = np.concatenate([rows[e] for e in g if ep_meta[e][1] == 1]) if any(
                ep_meta[e][1] == 1 for e in g) else np.concatenate([rows[e] for e in g])
            gi_a = np.concatenate([rows[e] for e in g])
            stats[s_] = (X_all[gi_s].mean(0), P_all[gi_a].mean(0), P_all[gi_a].std(0) + 1e-8)
        def prep(e):
            s_, u = ep_meta[e]
            mh, mp, sp = stats[s_]
            return u, X_all[rows[e]] - mh, (P_all[rows[e]] - mp) / sp
        def fitW(eps_, u):
            Xs_, Ps_ = [], []
            for e in eps_:
                if ep_meta[e][1] == u:
                    _, Xi, Pi = prep(e)
                    Xs_.append(Xi); Ps_.append(Pi)
            return ridge(np.concatenate(Ps_), np.concatenate(Xs_))
        Wsp, Wfp = fitW(tr, 1), fitW(tr, 0)
        # 혼합 scene 별
        by_scene = {}
        for e in rows:
            by_scene.setdefault(ep_meta[e][0], []).append(e)
        for s_, eps_ in sorted(by_scene.items()):
            ns_ = sum(ep_meta[e][1] for e in eps_); nf_ = len(eps_) - ns_
            if ns_ < 3 or nf_ < 3:
                continue
            dw = sorted(len(rows[e]) for e in eps_ if ep_meta[e][1] == 1)
            B = max(3, dw[len(dw) // 4])
            vals = {k: {0: [], 1: []} for k in "ABCL"}
            for e in eps_:
                train = [x for x in eps_ if x != e]
                if sum(ep_meta[x][1] for x in train) < 2 or sum(1 - ep_meta[x][1] for x in train) < 2:
                    continue
                u, Xi, Pi = prep(e)
                if len(Xi) < B:
                    continue
                Xb, Pb = Xi[:B], Pi[:B]
                Ws_, Wf_ = fitW(train, 1), fitW(train, 0)
                vals["A"][u].append(float((((Xb - Pb @ Ws_) ** 2).sum(1)
                                           - ((Xb - Pb @ Wf_) ** 2).sum(1)).mean()))
                vals["B"][u].append(float((((Xb - Pb @ Wsp) ** 2).sum(1)
                                           - ((Xb - Pb @ Wfp) ** 2).sum(1)).mean()))
                Xtr = np.concatenate([prep(x)[1][:B] for x in train if ep_meta[x][1] == 1])
                mu = Xtr.mean(0)
                _, S, V = np.linalg.svd(Xtr - mu, full_matrices=False)
                k = min(int(np.searchsorted(np.cumsum(S**2) / (S**2).sum(), 0.9)) + 1, len(S) - 1)
                r_ = Xb - mu
                vals["C"][u].append(float((np.linalg.norm(r_ - r_ @ V[:k].T @ V[:k], axis=1)**2
                                           / (np.linalg.norm(r_, axis=1)**2 + 1e-8)).mean()))
                vals["L"][u].append(float(len(Xi)))
            a = auroc(np.array(vals["A"][0]), np.array(vals["A"][1]))
            b = auroc(np.array(vals["B"][0]), np.array(vals["B"][1]))
            c = auroc(np.array(vals["C"][0]), np.array(vals["C"][1]))
            l = auroc(np.array(vals["L"][0]), np.array(vals["L"][1]))
            print(f"{SLUG}\t{phname}\ts{s_}\t{ns_}\t{nf_}\t{a:.2f}\t{b:.2f}\t{c:.2f}\t{l:.2f}\t{B}")
print("SITUFIT_DONE")
