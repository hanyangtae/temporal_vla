"""grid v2 분리도 전수 배터리 (stdout TSV). usage: python - <slug>

행: task phase metric value n_s n_f note
전부 길이-공정: 고정B(train 성공 dwell 25pct) 초반 창, held-out episode, 5-seed 중앙값.
지표: condg_margin / meandiff_proj / dist_succ / subspace_resid / disp_ratio / len_only
+ per-scene(situation) 내 condg margin 분포.
"""
import json, os, sys
import numpy as np

SLUG = sys.argv[1]
BASE = os.path.expanduser("~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase/segB_v2/")
d = np.load(BASE + SLUG + ".npz", allow_pickle=True)
meta = json.loads(str(d["meta_json"]))
CB = {v: k for k, v in meta["phase_codebook"].items()}
X_all = d["X"].astype(np.float32); P_all = d["P"].astype(np.float64)
pc, ep, sc, su = d["phase_code"], d["ep_id"], d["scene"], d["succ"]

def auroc(pos, neg):
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort().astype(np.float64) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def ridge(P, X, lam_scale=1e-3):
    lam = lam_scale * len(P)
    return np.linalg.solve(P.T @ P + lam * np.eye(P.shape[1]), P.T @ X)

ep_ids = np.unique(ep)
ep_meta = {e: (int(sc[ep == e][0]), int(su[ep == e][0])) for e in ep_ids}

for pcode in sorted(set(pc.tolist())):
    phname = CB[pcode]
    rows = []
    for e in ep_ids:
        idx = np.where((ep == e) & (pc == pcode))[0]
        if len(idx) >= 4:
            rows.append((e, ep_meta[e][0], ep_meta[e][1], idx))
    ns = sum(r[2] for r in rows); nf = len(rows) - ns
    if ns < 10 or nf < 10:
        print(f"{SLUG}\t{phname}\tSKIP\tnan\t{ns}\t{nf}\tep부족")
        continue
    res = {k: [] for k in ["condg_margin", "meandiff_proj", "dist_succ",
                            "subspace_resid", "len_only", "ep_margin"]}
    disp_r, eff_rank = [], []
    scene_within = {}
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
            gi_s = np.concatenate([r[3] for r in g if r[2] == 1]) if any(r[2] == 1 for r in g) \
                else np.concatenate([r[3] for r in g])
            gi_a = np.concatenate([r[3] for r in g])
            stats[s_] = (X_all[gi_s].mean(0), P_all[gi_a].mean(0), P_all[gi_a].std(0) + 1e-8)
        def prep(i):
            e, s_, u, idx = rows[i]
            mh, mp, sp = stats[s_]
            return u, X_all[idx] - mh, (P_all[idx] - mp) / sp
        succ_dw = sorted(len(rows[i][3]) for i in tr if rows[i][2] == 1)
        B = max(3, succ_dw[len(succ_dw) // 4])
        # fit (train, 고정B 창 record 로만 fit — 길이 누출 차단)
        def cls_data(u):
            Xs_, Ps_ = [], []
            for i in tr:
                if rows[i][2] == u:
                    _, Xi, Pi = prep(i)
                    Xs_.append(Xi[:B]); Ps_.append(Pi[:B])
            return np.concatenate(Xs_), np.concatenate(Ps_)
        Xs_tr, Ps_tr = cls_data(1); Xf_tr, Pf_tr = cls_data(0)
        Ws, Wf = ridge(Ps_tr, Xs_tr), ridge(Pf_tr, Xf_tr)
        mu_s, mu_f = Xs_tr.mean(0), Xf_tr.mean(0)
        dmu = mu_f - mu_s; dmu /= (np.linalg.norm(dmu) + 1e-8)
        Xc = Xs_tr - mu_s
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        k = int(np.searchsorted(np.cumsum(S**2) / (S**2).sum(), 0.9)) + 1
        V = _[ :k]
        cs, cf = np.cov(Xs_tr.T), np.cov(Xf_tr.T)
        disp_r.append(float(np.trace(cf) / (np.trace(cs) + 1e-8)))
        Sf = np.linalg.svd(Xf_tr - mu_f, compute_uv=False)
        eff_rank.append(float((Sf**2).sum()**2 / (Sf**4).sum()))
        vals = {k2: {0: [], 1: []} for k2 in res}
        for i in te:
            u, Xi, Pi = prep(i)
            if len(Xi) < B:
                continue
            Xb, Pb = Xi[:B], Pi[:B]
            m = ((Xb - Pb @ Ws) ** 2).sum(1) - ((Xb - Pb @ Wf) ** 2).sum(1)
            m_full = ((Xi - Pi @ Ws) ** 2).sum(1) - ((Xi - Pi @ Wf) ** 2).sum(1)
            vals["condg_margin"][u].append(float(m.mean()))
            vals["ep_margin"][u].append(float(m_full.mean()))
            vals["meandiff_proj"][u].append(float(((Xb - mu_s) @ dmu).mean()))
            vals["dist_succ"][u].append(float(np.linalg.norm(Xb - mu_s, axis=1).mean()))
            r_ = Xb - mu_s
            vals["subspace_resid"][u].append(float((np.linalg.norm(r_ - r_ @ V.T @ V, axis=1)**2
                                                    / (np.linalg.norm(r_, axis=1)**2 + 1e-8)).mean()))
            vals["len_only"][u].append(float(len(Xi)))
        for k2 in res:
            res[k2].append(auroc(np.array(vals[k2][0]), np.array(vals[k2][1])))
        # per-scene: held-out ep 를 scene 별로 (seed pool)
        for i in te:
            u, Xi, Pi = prep(i)
            if len(Xi) < B:
                continue
            m = float((((Xi[:B] - Pi[:B] @ Ws) ** 2).sum(1)
                       - ((Xi[:B] - Pi[:B] @ Wf) ** 2).sum(1)).mean())
            scene_within.setdefault(rows[i][1], {0: [], 1: []})[u].append(m)
    def med(a):
        a = [x for x in a if not np.isnan(x)]
        return float(np.median(a)) if a else float("nan")
    B_note = f"B~{B}" if 'B' in dir() else ""
    for k2 in ["condg_margin", "ep_margin", "meandiff_proj", "dist_succ", "subspace_resid", "len_only"]:
        print(f"{SLUG}\t{phname}\t{k2}\t{med(res[k2]):.3f}\t{ns}\t{nf}\t{B_note}")
    print(f"{SLUG}\t{phname}\tdisp_ratio\t{med(disp_r):.2f}\t{ns}\t{nf}\teff_rank_f={med(eff_rank):.1f}")
    ps = []
    for s_, v in sorted(scene_within.items()):
        if len(v[0]) >= 3 and len(v[1]) >= 3:
            ps.append((s_, auroc(np.array(v[0]), np.array(v[1])), len(v[1]), len(v[0])))
    if ps:
        arr = [x[1] for x in ps]
        print(f"{SLUG}\t{phname}\tscene_margin\t{float(np.median(arr)):.3f}\t{ns}\t{nf}\t"
              f"scenes={len(ps)} min={min(arr):.2f} max={max(arr):.2f} "
              f"detail={';'.join(f's{a}:{b:.2f}({c}/{d})' for a,b,c,d in ps)}")
print(f"BATTERY_DONE {SLUG}")
