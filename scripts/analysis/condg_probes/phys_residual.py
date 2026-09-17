"""활성화 = 물리 상태의 재부호화인가? — 잔차 판별 (stdout 전용).

pkl 의 states(eef_pos_rel/quat_rel/gripper) + 속도(인접 record 차)로 활성화를 릿지
회귀하고, (a) 물리 설명력 R², (b) 경로길이/관반경이 잔차에서 남는가,
(c) succ/fail 분리(mean/disp AUROC)가 잔차에서 남는가를 잰다. DiT L12 vs VL 대조.
usage: python - <slug=OpenDrawer_left> <phases=reach-to-handle,grasp-handle>
"""
import glob
import os
import pickle
import sys

import numpy as np

SLUG = sys.argv[1]
PHASES = sys.argv[2].split(",")
NQ = 4
GRID = os.path.expanduser("~/datasets/temporal_vla_store/groot/n15/grid/")
TASKMAP = {"OpenDrawer_left": "OpenDrawer/left",
           "DishwasherRack_out": "DishwasherRack/out",
           "PPCC_candle": "PPCC/candle"}
pat = GRID + f"*/*/{TASKMAP[SLUG]}/s*/n[0-4]/base/rollout.pkl"
pkls = sorted(glob.glob(pat))
print(f"# {SLUG}: pkl {len(pkls)}개")

EPS = []      # (scene, succ, H[n,1536], V[n,2048], P[n,phys], phases[n])
for p in pkls:
    d = pickle.load(open(p, "rb"))
    cap = [int(x) for x in d["capture_layers"]]
    li = cap.index(12)
    H = np.stack([np.asarray(h[li, 3], dtype=np.float32).mean(0)
                  for h in d["hidden_states"]])
    V = np.stack([np.asarray(v, dtype=np.float32) for v in d["vl_hidden_states"]])
    st = []
    for s in d["states"]:
        st.append(np.concatenate([np.asarray(s["observation.state.eef_pos_rel"]),
                                  np.asarray(s["observation.state.eef_quat_rel"]),
                                  np.asarray(s["observation.state.gripper_qpos"])]))
    P0 = np.stack(st).astype(np.float64)              # [n,9]
    vel = np.vstack([np.zeros((1, 9)), np.diff(P0, axis=0)])
    P = np.hstack([P0, vel])                          # [n,18]
    EPS.append((int(d["scene_idx"]), int(d["episode_success"]), H, V, P,
                list(d["feature_phases"])))

D_H, D_V = EPS[0][2].shape[1], EPS[0][3].shape[1]


def auroc(pos, neg):
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort().astype(np.float64) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


for phname in PHASES:
    # phase record 수집 (scene-중심화: 성공 phase 평균)
    def gather(feat_idx):
        rows, scenes, succs, epid = [], [], [], []
        for k, (sc, su, H, V, P, phs) in enumerate(EPS):
            idx = [i for i, p in enumerate(phs) if p == phname]
            if len(idx) < NQ:
                continue
            F = (H if feat_idx == 0 else V)[idx]
            rows.append((sc, su, F, P[idx], k))
        return rows

    for name, fi in [("DiT_L12", 0), ("VL", 1)]:
        rows = gather(fi)
        if not rows:
            continue
        # 길이 공정화: episode 별 phase 첫 B record 만 (B = 성공 dwell 25퍼센타일)
        sdw = sorted(len(r[2]) for r in rows if r[1] == 1)
        B = max(3, sdw[len(sdw) // 4])
        rows = [(sc, su, F[:B], Pp[:B], k) for (sc, su, F, Pp, k) in rows]
        # scene-중심화
        Xc, Pc, su_r, ep_r = [], [], [], []
        for sc in set(r[0] for r in rows):
            grp = [r for r in rows if r[0] == sc]
            ref = np.concatenate([r[2] for r in grp if r[1] == 1]) if any(
                r[1] == 1 for r in grp) else np.concatenate([r[2] for r in grp])
            mu = ref.mean(0)
            pref = np.concatenate([r[3] for r in grp])
            pmu, psd = pref.mean(0), pref.std(0) + 1e-8
            for r in grp:
                Xc.append(r[2] - mu)
                Pc.append((r[3] - pmu) / psd)
                su_r.append(np.full(len(r[2]), r[1]))
                ep_r.append(np.full(len(r[2]), r[4]))
        X = np.concatenate(Xc).astype(np.float64)
        P = np.concatenate(Pc)
        su_r = np.concatenate(su_r).astype(bool)
        ep_r = np.concatenate(ep_r)
        # 릿지 회귀 (X ~ P)
        lam = 1e-3 * len(P)
        W = np.linalg.solve(P.T @ P + lam * np.eye(P.shape[1]), P.T @ X)
        R = X - P @ W
        r2 = 1 - (R ** 2).sum() / (X ** 2).sum()
        # 경로/관 (성공만, episode별 사분위)
        def path_stats(Z):
            rowsq = [[] for _ in range(NQ)]
            for e in set(ep_r[su_r]):
                ix = np.where((ep_r == e) & su_r)[0]
                for j, i in enumerate(ix):
                    rowsq[min(NQ - 1, j * NQ // len(ix))].append(i)
            mus = [Z[np.array(r)].mean(0) for r in rowsq]
            pl = sum(np.linalg.norm(mus[i + 1] - mus[i]) for i in range(NQ - 1))
            rad = np.median([np.median(np.linalg.norm(Z[np.array(r)] - m, axis=1))
                             for r, m in zip(rowsq, mus)])
            return pl, rad
        # 분리 (mean-dir AUROC + disp AUROC)
        def seps(Z):
            Zs, Zf = Z[su_r], Z[~su_r]
            if len(Zf) < 20:
                return float("nan"), float("nan")
            v = Zf.mean(0) - Zs.mean(0)
            v /= np.linalg.norm(v) + 1e-12
            m_a = auroc(Zf @ v, Zs @ v)
            mu_s = Zs.mean(0)
            d_a = auroc(np.linalg.norm(Zf - mu_s, axis=1),
                        np.linalg.norm(Zs - mu_s, axis=1))
            return m_a, d_a
        pl0, rad0 = path_stats(X)
        pl1, rad1 = path_stats(R)
        m0, d0 = seps(X)
        m1, d1 = seps(R)
        print(f"{phname:<16}{name:<8} R2_phys={r2:.2f}  "
              f"경로/관 raw={pl0/rad0:.2f}→잔차={pl1/rad1:.2f}  "
              f"meanAUROC {m0:.2f}→{m1:.2f}  dispAUROC {d0:.2f}→{d1:.2f}  "
              f"(rec s/f={int(su_r.sum())}/{int((~su_r).sum())})")
print("PHYS_DONE")
