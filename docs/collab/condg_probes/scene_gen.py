"""조건부 margin의 scene 일반화 판별 (stdout 전용, LOSO).

변형 B: scene 중심화 없이 pooled fit → held-out scene 적용 (통짜 연산자)
변형 C: train scene들은 per-scene 중심화로 W 공유 fit, held-out scene 은 오프셋을
        그 episode 의 첫 K record 평균으로 자가 추정 (unseen 근사)
지표: held-out scene episode 의 고정B margin AUROC + 길이단독 (5개 scene fold 중앙값)
usage: python - <slug> <phase> [K=3]
"""
import glob
import os
import pickle
import sys

import numpy as np

SLUG, PHNAME = sys.argv[1], sys.argv[2]
K_SELF = int(sys.argv[3]) if len(sys.argv) > 3 else 3
GRID = os.path.expanduser("~/datasets/temporal_vla_store/groot/n15/grid/")
TASKMAP = {"OpenDrawer_left": "OpenDrawer/left", "OpenDrawer_right": "OpenDrawer/right",
           "DishwasherRack_out": "DishwasherRack/out", "PPCC_candle": "PPCC/candle"}
pkls = sorted(glob.glob(GRID + f"*/*/{TASKMAP[SLUG]}/s*/n[0-4]/base/rollout.pkl"))

EPS = []
seen_cells = set()
for p in pkls:
    d = pickle.load(open(p, "rb"))
    cell = (int(d["scene_idx"]), int(d["noise_idx"]))
    if cell in seen_cells:
        continue
    seen_cells.add(cell)
    cap = [int(x) for x in d["capture_layers"]]
    H = np.stack([np.asarray(h[cap.index(12), 3], dtype=np.float32).mean(0)
                  for h in d["hidden_states"]]).astype(np.float64)
    st = [np.concatenate([np.asarray(s["observation.state.eef_pos_rel"]),
                          np.asarray(s["observation.state.eef_quat_rel"]),
                          np.asarray(s["observation.state.gripper_qpos"])])
          for s in d["states"]]
    P0 = np.stack(st).astype(np.float64)
    P = np.hstack([P0, np.vstack([np.zeros((1, 9)), np.diff(P0, axis=0)])])
    idx = [i for i, ph in enumerate(d["feature_phases"]) if ph == PHNAME]
    if len(idx) >= 4:
        EPS.append((int(d["scene_idx"]), int(d["episode_success"]), H[idx], P[idx]))
print(f"# {SLUG}/{PHNAME}: ep {len(EPS)} (s/f={sum(e[1] for e in EPS)}"
      f"/{sum(1-e[1] for e in EPS)})")


def auroc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort().astype(np.float64) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def ridge(Xl, Pl):
    X = np.concatenate(Xl)
    P = np.concatenate(Pl)
    lam = 1e-3 * len(P)
    return np.linalg.solve(P.T @ P + lam * np.eye(P.shape[1]), P.T @ X)


sdw = sorted(len(e[2]) for e in EPS if e[1] == 1)
B = max(3, sdw[len(sdw) // 4])
scenes = sorted(set(e[0] for e in EPS))
res = {"B_pool": [], "C_self": [], "len": []}
n_folds = 0
for ho in scenes:                            # 유효 fold 최대 5개
    tr = [e for e in EPS if e[0] != ho]
    te = [e for e in EPS if e[0] == ho]
    if n_folds >= 5:
        break
    if sum(e[1] for e in te) < 2 or sum(1 - e[1] for e in te) < 2:
        continue
    n_folds += 1
    # 공통: phys 스케일은 train 전체
    Pall = np.concatenate([e[3] for e in tr])
    pmu, psd = Pall.mean(0), Pall.std(0) + 1e-8
    zp = lambda P: (P - pmu) / psd

    # ── B: 무중심화 pooled fit ──
    gmu = np.concatenate([e[2] for e in tr]).mean(0)
    WsB = ridge([e[2] - gmu for e in tr if e[1] == 1], [zp(e[3]) for e in tr if e[1] == 1])
    WfB = ridge([e[2] - gmu for e in tr if e[1] == 0], [zp(e[3]) for e in tr if e[1] == 0])
    # ── C: per-scene 중심화 공유 W ──
    cen = {}
    for sc in set(e[0] for e in tr):
        g = [e for e in tr if e[0] == sc]
        ref = [e[2] for e in g if e[1] == 1] or [e[2] for e in g]
        cen[sc] = np.concatenate(ref).mean(0)
    WsC = ridge([e[2] - cen[e[0]] for e in tr if e[1] == 1],
                [zp(e[3]) for e in tr if e[1] == 1])
    WfC = ridge([e[2] - cen[e[0]] for e in tr if e[1] == 0],
                [zp(e[3]) for e in tr if e[1] == 0])

    for tag, Ws, Wf, center in [
            ("B_pool", WsB, WfB, lambda e: gmu),
            ("C_self", WsC, WfC, lambda e: e[2][:K_SELF].mean(0))]:
        m = {0: [], 1: []}
        for e in te:
            if len(e[2]) < B:
                continue
            X = (e[2] - center(e))[:B]
            P = zp(e[3])[:B]
            mm = ((X - P @ Ws) ** 2).sum(1) - ((X - P @ Wf) ** 2).sum(1)
            m[e[1]].append(float(np.mean(mm)))
        res[tag].append(auroc(np.array(m[0]), np.array(m[1])))
    dw = {0: [], 1: []}
    for e in te:
        dw[e[1]].append(float(len(e[2])))
    res["len"].append(auroc(np.array(dw[0]), np.array(dw[1])))

fmt = lambda v: ("n=0" if not v else f"{np.nanmedian(v):.2f}({np.nanmin(v):.2f}-{np.nanmax(v):.2f}, n={len(v)})")
print(f"held-out-SCENE 고정B={B} margin AUROC (fold 중앙값(범위)):")
print(f"  B 통짜(무중심화): {fmt(res['B_pool'])}")
print(f"  C 공유W+자가오프셋(K={K_SELF}): {fmt(res['C_self'])}")
print(f"  길이단독: {fmt(res['len'])}")
