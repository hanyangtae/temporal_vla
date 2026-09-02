"""조건부 대조 margin의 held-out 판별력 (stdout 전용).

성공/실패 각각 상태→활성화 릿지 회귀(ĥ_s, ĥ_f)를 train episode 로 fit,
held-out episode 에서 margin m = ‖h−ĥ_s(s)‖² − ‖h−ĥ_f(s)‖² 의 AUROC.
record 단위 / episode-mean / 초반 절반 한정 3종. usage: python - <slug> <phases>
"""
import glob
import os
import pickle
import sys

import numpy as np

SLUG = sys.argv[1]
PHASES = sys.argv[2].split(",")
GRID = os.path.expanduser("~/datasets/temporal_vla_store/groot/n15/grid/")
TASKMAP = {"OpenDrawer_left": "OpenDrawer/left", "OpenDrawer_right": "OpenDrawer/right",
           "DishwasherRack_out": "DishwasherRack/out", "OvenRack_out": "OvenRack/out",
           "PPCC_candle": "PPCC/candle", "PPCC_bread": "PPCC/bread",
           "PPCC_marshmallow": "PPCC/marshmallow", "PPCC_jug": "PPCC/jug",
           "CoffeeSetupMug": "CoffeeSetupMug"}
pkls = sorted(glob.glob(GRID + f"*/*/{TASKMAP[SLUG]}/s*/n[0-4]/base/rollout.pkl"))
print(f"# {SLUG}: pkl {len(pkls)}개")

EPS = []
seen_cells = set()
for p in pkls:
    d = pickle.load(open(p, "rb"))
    cell = (int(d["scene_idx"]), int(d["noise_idx"]))
    if cell in seen_cells:      # 다중 머신 수집 task 의 중복 셀 제거
        continue
    seen_cells.add(cell)
    cap = [int(x) for x in d["capture_layers"]]
    H = np.stack([np.asarray(h[cap.index(12), 3], dtype=np.float32).mean(0)
                  for h in d["hidden_states"]])
    st = [np.concatenate([np.asarray(s["observation.state.eef_pos_rel"]),
                          np.asarray(s["observation.state.eef_quat_rel"]),
                          np.asarray(s["observation.state.gripper_qpos"])])
          for s in d["states"]]
    P0 = np.stack(st).astype(np.float64)
    P = np.hstack([P0, np.vstack([np.zeros((1, 9)), np.diff(P0, axis=0)])])
    EPS.append((int(d["scene_idx"]), int(d["episode_success"]), H.astype(np.float64),
                P, list(d["feature_phases"])))


def auroc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort().astype(np.float64) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


rng = np.random.default_rng(0)
for phname in PHASES:
    if not any(phname in r[4] for r in EPS):
        print(f"{phname}: phase 없음(codebook)")
        continue
    rows = []
    for k, (sc, su, H, P, phs) in enumerate(EPS):
        idx = [i for i, ph in enumerate(phs) if ph == phname]
        if len(idx) >= 4:
            rows.append((sc, su, H[idx], P[idx], k))
    ns, nf = sum(r[1] for r in rows), sum(1 - r[1] for r in rows)
    if ns < 7 or nf < 7:
        print(f"{phname}: ep 부족 s/f={ns}/{nf}")
        continue
    # scene-중심화 파라미터는 train 에서만 (성공 기준)
    order = rng.permutation(len(rows))
    tr = set(order[: int(len(rows) * 0.6)])
    stats = {}
    for sc in set(r[0] for r in rows):
        g = [rows[i] for i in tr if rows[i][0] == sc] or [r for r in rows if r[0] == sc]
        ref_h = np.concatenate([r[2] for r in g if r[1] == 1]) if any(
            r[1] == 1 for r in g) else np.concatenate([r[2] for r in g])
        ref_p = np.concatenate([r[3] for r in g])
        stats[sc] = (ref_h.mean(0), ref_p.mean(0), ref_p.std(0) + 1e-8)

    def prep(i):
        sc, su, H, P, _ = rows[i]
        mh, mp, sp = stats[sc]
        return su, H - mh, (P - mp) / sp

    def ridge(X_list, P_list):
        X = np.concatenate(X_list)
        P = np.concatenate(P_list)
        lam = 1e-3 * len(P)
        return np.linalg.solve(P.T @ P + lam * np.eye(P.shape[1]), P.T @ X)

    Ws = ridge(*zip(*[(prep(i)[1], prep(i)[2]) for i in tr if rows[i][1] == 1]))
    Wf = ridge(*zip(*[(prep(i)[1], prep(i)[2]) for i in tr if rows[i][1] == 0]))

    # 길이 공정화: 고정 예산 B = train 성공 dwell 의 25퍼센타일 (전 episode 동일 초반 창)
    succ_dw = sorted(len(rows[i][2]) for i in tr if rows[i][1] == 1)
    B = max(3, succ_dw[len(succ_dw) // 4])
    ep_m = {0: [], 1: []}
    fixB_m = {0: [], 1: []}
    dwell = {0: [], 1: []}
    for i in range(len(rows)):
        if i in tr:
            continue
        su, X, P = prep(i)
        m = ((X - P @ Ws) ** 2).sum(1) - ((X - P @ Wf) ** 2).sum(1)
        ep_m[su].append(float(np.mean(m)))
        if len(m) >= B:
            fixB_m[su].append(float(np.mean(m[:B])))
        dwell[su].append(len(m))
    print(f"{phname:<16} held-out AUROC(fail>succ): episode "
          f"{auroc(np.array(ep_m[0]), np.array(ep_m[1])):.2f}  "
          f"고정B={B} {auroc(np.array(fixB_m[0]), np.array(fixB_m[1])):.2f}"
          f"(n={len(fixB_m[0])}/{len(fixB_m[1])})  "
          f"길이단독 {auroc(np.array(dwell[0], float), np.array(dwell[1], float)):.2f}  "
          f"(held-out s/f={len(ep_m[1])}/{len(ep_m[0])})")
print("MARGIN_DONE")
