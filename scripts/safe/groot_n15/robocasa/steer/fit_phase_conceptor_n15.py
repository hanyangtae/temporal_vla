"""N1.5 phase-event 대조 conceptor fit (Rung 3 steering target 생성).

방향 1: 관측 분리가 confound(scene/길이)로 약하다는 걸 확인했으니, **관측을 더 파지 않고
steering 연산자를 fit해 인과(ΔSR)로 판정**한다(사용자 논리 = 인과가 arbiter).

N1.6판 fit_conceptor_steering.py 는 [L,D]/[L,T,D] token layout·task_id 그룹핑이라 N1.5
phase-event([L=7,K=4,D=1536] denoise + feature_phases + seed별 cell)에 안 맞음. 이 어댑터는:
  - **단일 scene(cell dir, seed 구분)** 내에서만 fit → scene confound 제거([[dit-succfail-apparent-separation-confound]]).
  - denoise축(K=4) mean-pool → 주입점(transformer_blocks[ℓ] residual, D=1536) / VL(vlln, D=2048)과 동일 공간.
  - group = {global(전 record), 각 phase(reach/transport/insert-settle)} × layer(DiT capture 7 + VL).
  - C_steer = C_success ∧ ¬C_failure (src.conceptor.contrastive_conceptor), alpha grid sweep.

산출 NPZ 는 serve 소비 계약(steering_hooks.load_steering_matrix)과 동일: 키 `alpha{a}_C_steer`
/`_C_success`/`_C_failure`. 사용: serve `--steering-npz <.../conceptors.npz> --steering-pathway
{dit,vl} [--steering-layer <blk>] --steering-alpha <a> --steering-beta <b> --steering-key C_steer`.
DiT layer_tag `dit_L<blk>` 의 <blk> 를 그대로 --steering-layer 로 넘긴다(transformer_blocks 인덱스).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/safe/groot_n15/robocasa/analyze"))

from phase_separation import load_rollout, phase_records  # noqa: E402
from src.conceptor import (  # noqa: E402
    and_conceptor,
    as_float32,
    compute_conceptor,
    conceptor_overlap,
    conceptor_quota,
    not_conceptor,
)

DEFAULT_ALPHAS = [0.1, 0.3, 1.0, 3.0, 10.0]
OVERLAP_BAND = (0.85, 0.95)  # COAST A.10.2 Stage2


def carve_5phase(pkl_path, phases: list[str], w: int) -> list[str]:
    """5-phase carving (사용자 지시): event 직전 W record 를 pre-grasp/pre-place 로 재라벨.

    strict(drop-aware) 수집 pkl 의 grasp_steps(반복 가능)/event_steps['place:obj'] 사용.
    - pre-grasp: 각 grasp record-idx g 직전 [g-w, g) 중 reach-to-object 인 record.
    - pre-place: place record-idx p 직전 [p-w, p) 중 transport 인 record.
    → reach / pre-grasp / transport / pre-place / insert-settle 5-phase.
    """
    import pickle

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    out = list(phases)
    n = len(out)
    grasp_steps = list(d.get("grasp_steps") or [])
    if not grasp_steps and d.get("event_steps", {}).get("grasp:obj") is not None:
        grasp_steps = [d["event_steps"]["grasp:obj"]]
    for g in grasp_steps:
        for r in range(max(0, int(g) - w), min(int(g), n)):
            if out[r] == "reach-to-object":
                out[r] = "pre-grasp"
    p_step = d.get("event_steps", {}).get("place:obj")
    if p_step is not None:
        for r in range(max(0, int(p_step) - w), min(int(p_step), n)):
            if out[r] == "transport":
                out[r] = "pre-place"
    return out


def gather_class_records(rolls, phase, layer_key, success):
    """success(0/1) rollout 들의 phase 소속 record 벡터 [N, D] 스택. phase='global'이면 전 record.

    wrong-grasp record 는 phase bin 에선 라벨 분리로 자동 제외(fit group 에 wrong-grasp 없음),
    global 은 사용자 결정대로 전 record 포함(phase 무구분 조건이므로).
    """
    out = []
    for r in rolls:
        if r["success"] != success:
            continue
        if phase == "global":
            if layer_key == "VL":
                if r["vl"] is not None:
                    out.extend(list(r["vl"]))
            else:
                out.extend(list(r["dit"][:, layer_key, :]))
        else:
            out.extend(phase_records(r, phase, layer_key))
    return np.asarray(out, dtype=np.float64) if out else np.empty((0, 0))


def select_alpha(sweep, band):
    lo, hi = band
    in_band = sorted((r["alpha"] for r in sweep if lo <= r["overlap"] <= hi))
    if in_band:
        return in_band[0], "band"
    return min(sweep, key=lambda r: min(abs(r["overlap"] - lo), abs(r["overlap"] - hi)))["alpha"], "closest"


def fit_one(Xs, Xf, alphas, band):
    """succ/fail 표본 → alpha별 C_steer + sweep. 반환 (fits{alpha:dict}, meta)."""
    sweep, cache = [], {}
    for a in alphas:
        Cs = compute_conceptor(Xs, a)
        Cf = compute_conceptor(Xf, a)
        cache[a] = (Cs, Cf)
        sweep.append({"alpha": float(a), "overlap": conceptor_overlap(Cs, Cf),
                      "quota_success": conceptor_quota(Cs), "quota_failure": conceptor_quota(Cf)})
    sel_alpha, sel_mode = select_alpha(sweep, band)
    fits = {}
    for a in {sel_alpha, alphas[1] if len(alphas) > 1 else alphas[0]}:  # 선택 alpha + 안전 default 저장
        Cs, Cf = cache[a]
        Csteer = and_conceptor(Cs, not_conceptor(Cf))
        fits[a] = {"C_steer": Csteer, "C_success": Cs, "C_failure": Cf,
                   "quota_steer": conceptor_quota(Csteer)}
    return fits, {"alpha_sweep": sweep, "selected_alpha": float(sel_alpha), "selection_mode": sel_mode,
                  "n_success": int(Xs.shape[0]), "n_failure": int(Xf.shape[0]), "feature_dim": int(Xs.shape[1])}


def save_npz(out_dir, fits, meta):
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for a, g in fits.items():
        arrays[f"alpha{a:g}_C_steer"] = as_float32(g["C_steer"])
        arrays[f"alpha{a:g}_C_success"] = as_float32(g["C_success"])
        arrays[f"alpha{a:g}_C_failure"] = as_float32(g["C_failure"])
    np.savez_compressed(out_dir / "conceptors.npz", **arrays)
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description="N1.5 phase-event 대조 conceptor fit")
    ap.add_argument("--run-dir", default="outputs/eval/robocasa/groot_n15/phase_event_aligned_4cell/raw_rollouts")
    ap.add_argument("--cell", required=True, help="예: PickPlaceCounterToCabinet/ppcc_bread")
    ap.add_argument("--groups", default="global,transport,reach-to-object")
    ap.add_argument("--alphas", default=None)
    ap.add_argument("--min-per-class", type=int, default=3)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--carve-window", type=int, default=0,
                    help=">0 이면 5-phase carving: event 직전 W record 를 pre-grasp/pre-place 로 재라벨")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cell_dir = run_dir / args.cell
    pkls = sorted(cell_dir.glob("*.pkl"))
    if not pkls:
        raise SystemExit(f"no pkl under {cell_dir}")
    rolls = [load_rollout(p) for p in pkls]
    if args.carve_window > 0:
        for r, p in zip(rolls, pkls):
            r["phases"] = carve_5phase(p, r["phases"], args.carve_window)
    cap = rolls[0]["capture_layers"]
    has_vl = rolls[0]["vl"] is not None
    layer_keys = list(range(rolls[0]["dit"].shape[1])) + (["VL"] if has_vl else [])
    alphas = [float(x) for x in args.alphas.split(",")] if args.alphas else DEFAULT_ALPHAS
    groups = args.groups.split(",")
    cell_id = args.cell.split("/")[-1]
    out_root = Path(args.out_dir) if args.out_dir else run_dir.parent / "analysis" / "conceptor_steering_n15" / cell_id
    n_s = sum(r["success"] for r in rolls); n_f = len(rolls) - n_s
    print(f"[cell {cell_id}] {len(rolls)} rollouts succ={n_s} fail={n_f}  layers={cap}+VL={has_vl}")

    summary = {}
    for group in groups:
        for lk in layer_keys:
            tag = "vl" if lk == "VL" else f"dit_L{cap[lk]}"
            Xs = gather_class_records(rolls, group, lk, 1)
            Xf = gather_class_records(rolls, group, lk, 0)
            if Xs.size == 0 or Xf.size == 0 or Xs.shape[0] < args.min_per_class or Xf.shape[0] < args.min_per_class:
                continue
            fits, meta = fit_one(Xs, Xf, alphas, OVERLAP_BAND)
            meta.update({"cell": cell_id, "group": group, "layer_tag": tag,
                         "steering_layer": None if lk == "VL" else int(cap[lk]),
                         "pathway": "vl" if lk == "VL" else "dit"})
            save_npz(out_root / group / tag, fits, meta)
            summary.setdefault(group, {})[tag] = {"sel_alpha": meta["selected_alpha"],
                                                  "n_s": meta["n_success"], "n_f": meta["n_failure"]}
            print(f"  [{group}/{tag}] Ns={Xs.shape[0]} Nf={Xf.shape[0]} sel_alpha={meta['selected_alpha']:g} "
                  f"overlap@sel={[s['overlap'] for s in meta['alpha_sweep'] if s['alpha']==meta['selected_alpha']][0]:.3f}")
    (out_root / "fit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[done] -> {out_root}")


if __name__ == "__main__":
    main()
