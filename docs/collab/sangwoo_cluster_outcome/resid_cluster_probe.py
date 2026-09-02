"""condg식 물리 잔차화 후에도 3분류(fail-only/shared/succ)·phase 구조가 남는가.

Arms (동일 파이프라인, 입력만 다름 — AE는 공정성 위해 양쪽 다 생략):
  raw   : L12-D3 DiT pooled 1536 → PCA64+whiten(train) → kmeans K=48(train)
  resid : 물리상태 16d(gripper_qpos, base pos/rot, eef pos/quat rel) ridge 회귀
          잔차(train fit) → 동일 PCA→kmeans
지표: phase 다수결 purity·NMI, outcome 3분류(Fisher+BH 관문, t<45 대조 포함), phys R².
"""
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import fisher_exact

ROOT = "/home/iw/task_classification/datasets_local/phase_cls_pq3"
CELLS = ["pq3_drawer_left", "pq3_drawer_right", "pq3_ppcc_beer", "pq3_ppcc_bread", "pq3_ppcc_pizza_cutter"]
LAYER_IDX = 5   # capture_layers [0,2,4,8,10,12,15] -> 12
DEN_IDX = 3
K = 48
SEEDS = [0, 1, 2]
PHASE_NAMES = ["reach-obj", "grasp", "transport", "place", "insert", "terminal", "wrong-grasp",
               "reach-handle", "grasp-handle", "pull", "open-done", "disengage", "push-back"]

idx = pd.read_parquet(f"{ROOT}/index.parquet")
xs, es = [], []
for c in CELLS:
    sh = f"phase_event_pq3__{c}"
    dit = np.load(f"{ROOT}/shards/{sh}.dit.npy", mmap_mode="r")
    xs.append(np.asarray(dit[:, LAYER_IDX, DEN_IDX, :], np.float32))
    es.append(np.load(f"{ROOT}/shards/{sh}.extra.npy").astype(np.float32))
    n = len(xs[-1])
    assert (idx[idx.shard == sh].shape[0] == n), (c, n)
# index.parquet 행 순서가 shard 순서와 일치하는지: shard별 row 열로 정렬해 재배열
idx = idx.sort_values(["shard", "row"])
order = {f"phase_event_pq3__{c}": i for i, c in enumerate(CELLS)}
idx["shard_ord"] = idx["shard"].map(order)
idx = idx.sort_values(["shard_ord", "row"]).reset_index(drop=True)
X = np.concatenate(xs)
PHYS = np.concatenate(es)[:, 12:28]   # action_vector 제외, 물리상태만
assert len(X) == len(idx) == 12041

sp = json.load(open(f"{ROOT}/splits/cell_union.json"))
ep2sp = {}
for s in ["train", "val", "test"]:
    for e in sp[s]:
        ep2sp[e] = s
idx["split"] = idx["episode_id"].map(ep2sp)
print("split counts:", idx["split"].value_counts().to_dict())
tr = (idx["split"] == "train").to_numpy()

# --- ridge 잔차화 (train fit) ---
mu_p, sd_p = PHYS[tr].mean(0), PHYS[tr].std(0) + 1e-8
P = (PHYS - mu_p) / sd_p
P1 = np.hstack([P, np.ones((len(P), 1), np.float32)])
lam = 1e-2
A = P1[tr].T @ P1[tr] + lam * np.eye(P1.shape[1], dtype=np.float32)
B = P1[tr].T @ X[tr]
W = np.linalg.solve(A, B)
pred = P1 @ W
ss_res = ((X - pred) ** 2).sum(0)
ss_tot = ((X - X[tr].mean(0)) ** 2).sum(0)
r2_all = 1 - ss_res.sum() / ss_tot.sum()
te = ~tr
r2_ho = 1 - ((X[te] - pred[te]) ** 2).sum() / ((X[te] - X[tr].mean(0)) ** 2).sum()
print(f"phys->feature ridge R2: pooled {r2_all:.3f} | heldout(val+test) {r2_ho:.3f}")
XR = X - pred


def pca_whiten(x, trm, n=64):
    m = x[trm].mean(0)
    xc = (x - m).astype(np.float64)
    cov = xc[trm].T @ xc[trm] / (trm.sum() - 1)
    lam_, V = np.linalg.eigh(cov)
    lam_, V = lam_[::-1][:n].clip(1e-8), V[:, ::-1][:, :n]
    return (xc @ V) / np.sqrt(lam_)


def outcome_table(cl, idx, K, t_max=None):
    d = idx.copy()
    d["cluster"] = cl
    if t_max is not None:
        d = d[d["t"] < t_max]
    ep = d.groupby(["episode_id", "episode_success"], observed=True)["cluster"].agg(set).reset_index()
    n_s = int((ep.episode_success).sum()); n_f = len(ep) - n_s
    rows = []
    for k in range(K):
        has = ep["cluster"].map(lambda s: k in s)
        a = int((has & ep.episode_success).sum()); b = n_s - a
        c = int((has & ~ep.episode_success).sum()); dd = n_f - c
        if a + c == 0:
            continue
        orr = ((a + .5) * (dd + .5)) / ((b + .5) * (c + .5))
        p = fisher_exact([[a, b], [c, dd]])[1]
        rows.append((k, a, c, a / n_s, c / n_f, np.log2(orr), p))
    t = pd.DataFrame(rows, columns=["cluster", "n_ep_succ", "n_ep_fail", "prev_s", "prev_f", "log2_or", "p"])
    # BH
    t = t.sort_values("p").reset_index(drop=True)
    m = len(t)
    q = t["p"] * m / (np.arange(m) + 1)
    t["q"] = np.minimum.accumulate(q[::-1])[::-1]
    t["side"] = "mid"
    t.loc[(t.log2_or >= 1) & (t.q < .05), "side"] = "succ"
    t.loc[(t.log2_or <= -1) & (t.q < .05), "side"] = "fail"
    t["exclusive"] = ""
    t.loc[(t.n_ep_succ == 0), "exclusive"] = "failure"
    t.loc[(t.n_ep_fail == 0), "exclusive"] = "success"
    return t.sort_values("cluster")


ph = idx["phase_id"].to_numpy()
res = {}
for name, feat in [("raw", X), ("resid", XR)]:
    Z = pca_whiten(feat, tr)
    for seed in SEEDS:
        cl = KMeans(K, n_init=10, random_state=seed).fit(Z[tr]).predict(Z)
        # phase purity (train 다수결)
        pur_n = pur_d = 0
        maj = np.full(K, -1)
        for k in range(K):
            sel = ph[tr & (cl == k)]
            if len(sel):
                bc = np.bincount(sel, minlength=13)
                maj[k] = bc.argmax(); pur_n += bc.max(); pur_d += len(sel)
        nmi = normalized_mutual_info_score(ph[te], cl[te])
        t_full = outcome_table(cl, idx, K)
        t_ctrl = outcome_table(cl, idx, K, t_max=45)
        fx = set(t_full[t_full.exclusive == "failure"].cluster)
        fx_surv = fx & set(t_ctrl[t_ctrl.exclusive == "failure"].cluster)
        sides = t_full["side"].value_counts().to_dict()
        res[(name, seed)] = dict(purity=pur_n / pur_d, nmi=nmi, sides=sides,
                                 n_fail_excl=len(fx), n_fail_excl_t45=len(fx_surv))
        if seed == 0:
            # phase별 3분류 공존 확인
            t_full["phase"] = [PHASE_NAMES[maj[k]] if maj[k] >= 0 else "-" for k in t_full.cluster]
            co = t_full.groupby("phase")["side"].agg(lambda s: dict(pd.Series(s).value_counts()))
            print(f"\n=== {name} s0: phase별 side 구성 ===")
            print(co.to_string())
            t_full.to_csv(f"/tmp/resid_probe_{name}_s0.csv", index=False)

print("\n=== 요약 (arm, seed) ===")
for k, v in res.items():
    print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
