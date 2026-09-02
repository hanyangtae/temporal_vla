"""setm_gt v4r판: 대상 scene 실패만 v4r pkl로 교체, 타 scene은 segA 유지.

usage: python - <slug> <target_scene:int>
수식·계약 = v4sb판 동일 (v = 타 scene scene-내 mean-diff 평균, s = 하이브리드).
산출: instr_setm_v4r_gt/<slug>/<phase>/dit_L12/conceptors.npz
"""
import glob
import json
import os
import pickle
import sys

import numpy as np

SLUG, TGT = sys.argv[1], int(sys.argv[2])
A = os.path.expanduser("~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase_v4/")

# ── 타 scene 통계 (segA, GT phase) ──
d = np.load(A + f"segA/{SLUG}.npz", allow_pickle=True)
meta = json.loads(str(d["meta_json"]))
cap = [int(x) for x in meta["capture_layers"]]
seg = meta["segment_names"].index("all")
X = d["X"][:, cap.index(12), 3, seg, :].astype(np.float32)
pc, sc, su = d["phase_code"], d["scene"], d["succ"]
CB = meta["phase_codebook"]

# ── 대상 scene v4r pkl (실패만 사용) ──
pkls = sorted(glob.glob(A + f"v4r_collect/*/ps_base/{SLUG}_s{TGT}_*/{SLUG}/ps_base/"
                        f"diag_grid/diag/{SLUG}/s*/n*/ps_base/rollout.pkl"))
tgt_fail = {}   # phase명 → [record 벡터]
n_ep = n_fail = 0
li = None
for p in pkls:
    r = pickle.load(open(p, "rb"))
    n_ep += 1
    if int(r["episode_success"]) == 1:
        continue
    n_fail += 1
    if li is None:
        li = [int(x) for x in r["capture_layers"]].index(12)
    for h, ph in zip(r["hidden_states"], r["gt_phases"]):
        v = np.asarray(h[li, 3], dtype=np.float32).mean(0)
        tgt_fail.setdefault(ph, []).append(v)
print(f"# {SLUG} s{TGT}: v4r ep {n_ep} (실패 {n_fail}), 실패 phase 분포: "
      + " ".join(f"{k}:{len(v)}" for k, v in sorted(tgt_fail.items())))

outroot = A + f"instr_setm_v4r_gt/{SLUG}"
n_out = 0
for ph, pcode in sorted(CB.items()):
    deltas = []
    for s_ in sorted(set(sc.tolist())):
        if s_ == TGT:
            continue
        ms = np.where((sc == s_) & (pc == pcode) & (su == 1))[0]
        mf = np.where((sc == s_) & (pc == pcode) & (su == 0))[0]
        if len(ms) < 20 or len(mf) < 20:
            continue
        deltas.append(X[ms].mean(0) - X[mf].mean(0))
    if len(deltas) < 2 or len(tgt_fail.get(ph, [])) < 10:
        continue
    delta = np.mean(deltas, 0)
    v = (delta / (np.linalg.norm(delta) + 1e-12)).astype(np.float64)
    mu_f_t = np.mean(tgt_fail[ph], 0).astype(np.float64)
    s_val = float(mu_f_t @ v) + float(np.mean([dd @ v for dd in deltas]))
    d_out = os.path.join(outroot, ph, "dit_L12")
    os.makedirs(d_out, exist_ok=True)
    np.savez_compressed(os.path.join(d_out, "conceptors.npz"),
                        alpha0_v_steer=v.astype(np.float32), alpha0_s=np.float32(s_val))
    with open(os.path.join(d_out, "metadata.json"), "w") as f:
        json.dump({"op": "setpoint", "variant": "instr_setm_v4r_gt", "phase": ph,
                   "target_scene": TGT, "target_src": "v4r_collect(재수집 정본)",
                   "pool": "other-scenes(segA)+target-fail(v4r)",
                   "n_scene_delta": len(deltas),
                   "n_target_fail_rec": len(tgt_fail[ph]),
                   "phase_label_source": "gt", "layer": 12}, f, ensure_ascii=False, indent=1)
    n_out += 1
print(f"[setm_gt v4r] {SLUG} s{TGT}: {n_out} phase → instr_setm_v4r_gt/{SLUG}")
print(f"V4R_SETM_DONE {SLUG}")
