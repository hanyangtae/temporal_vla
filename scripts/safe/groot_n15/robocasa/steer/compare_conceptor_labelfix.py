"""판정 교정(라벨 flip)이 대조 conceptor를 얼마나 바꾸는지 정량 비교.

배경: apple 성공 판정의 중심거리 임계값(0.07) 오판정으로 fit 데이터(ep0-59)의
succ/fail 라벨 일부가 뒤집힘 (예: s100050은 fail 5판 중 3판이 실제 성공).
이 스크립트는 같은 fit 데이터에서 (a) 원 라벨, (b) 교정 라벨로 conceptor를 각각
fit하고 연산자 수준에서 비교한다. 실행 중 파이프라인 파일은 수정하지 않음(신규 파일).

sanity 게이트: 원-라벨 재fit C_steer가 배포 NPZ(final_ps60)와 cosF ≈ 1.0 이어야
비교 자체를 신뢰 (재구현 오차 검증).

사용 (repo root에서, torch 있는 python):
  python scripts/safe/groot_n15/robocasa/steer/compare_conceptor_labelfix.py \
      --cell PickPlaceCounterToStove/ppcs_apple_s100050 --override 28,45,55 \
      [--alpha 0.3] [--deployed-root outputs/.../analysis/final_ps60]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/safe/groot_n15/robocasa/analyze"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_phase_conceptor_n15 import gather_class_records, fit_one  # noqa: E402
from phase_separation import load_rollout  # noqa: E402

GROUPS = ["global", "reach-to-object", "grasp", "transport", "place", "insert-settle"]
STEER_LAYERS = [4, 8, 12]


def frob_cos(A, B):
    na, nb = np.linalg.norm(A), np.linalg.norm(B)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.sum(A * B) / (na * nb))


def spec(C):
    ev = np.linalg.eigvalsh(C)
    return float(ev[-1]), int((ev > 0.5).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="outputs/eval/robocasa/groot_n15/phase_event_6p/raw_rollouts")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--override", required=True, help="fail→succ로 교정할 ep 번호들 (예: 28,45,55)")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--deployed-root",
                    default="outputs/eval/robocasa/groot_n15/phase_event_6p/analysis/final_ps60")
    ap.add_argument("--min-per-class", type=int, default=3)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    cell_dir = Path(args.run_dir) / args.cell
    cell_id = args.cell.split("/")[-1]
    pkls = sorted(cell_dir.glob("*.pkl"))
    if not pkls:
        raise SystemExit(f"no pkl under {cell_dir}")
    flip_eps = {int(x) for x in args.override.split(",")}

    rolls = [load_rollout(p) for p in pkls]
    eps = [int(p.stem.split("--")[1][2:]) for p in pkls]
    cap = rolls[0]["capture_layers"]
    lidx = {L: cap.index(L) for L in STEER_LAYERS}

    n_flip = sum(1 for e, r in zip(eps, rolls) if e in flip_eps and r["success"] == 0)
    print(f"[{cell_id}] rollouts={len(rolls)} 원 succ={sum(r['success'] for r in rolls)} "
          f"flip 대상(fail→succ)={n_flip}/{len(flip_eps)}", flush=True)

    def fit_variant(corrected: bool):
        out = {}
        # gather_class_records는 r['success']를 읽음 → 임시 오버라이드
        saved = [r["success"] for r in rolls]
        if corrected:
            for e, r in zip(eps, rolls):
                if e in flip_eps:
                    r["success"] = 1
        try:
            for g in GROUPS:
                for L in STEER_LAYERS:
                    Xs = gather_class_records(rolls, g, lidx[L], 1)
                    Xf = gather_class_records(rolls, g, lidx[L], 0)
                    if Xs.shape[0] < args.min_per_class or Xf.shape[0] < args.min_per_class:
                        continue
                    fits, meta = fit_one(Xs, Xf, [args.alpha], (0.85, 0.95))
                    out[(g, L)] = (fits[args.alpha], meta)
        finally:
            for r, s in zip(rolls, saved):
                r["success"] = s
        return out

    old = fit_variant(False)
    new = fit_variant(True)

    # sanity 게이트: 배포 NPZ와 원-라벨 재fit 대조 (global/L4)
    dep_path = Path(args.deployed_root) / cell_id / "global" / "dit_L4" / "conceptors.npz"
    if dep_path.exists() and ("global", 4) in old:
        dep = np.load(dep_path)
        key = f"alpha{args.alpha:g}_C_steer"
        if key in dep:
            c = frob_cos(dep[key], old[("global", 4)][0]["C_steer"])
            print(f"[sanity] 배포NPZ vs 원라벨 재fit (global/L4) cosF={c:.6f}", flush=True)
            if c < 0.999:
                print("[sanity] WARNING: 재현 불일치 — 비교 해석 주의", flush=True)
        else:
            print(f"[sanity] 배포NPZ에 {key} 없음 (keys={list(dep.keys())[:4]})", flush=True)
    else:
        print(f"[sanity] 배포NPZ 없음: {dep_path}", flush=True)

    print(f"\n{'group':16s} {'L':>3s} {'cos(Csteer)':>11s} {'relΔF':>7s} "
          f"{'λmax o→n':>13s} {'nλ>.5 o→n':>10s} {'cos(Cfail)':>10s} {'Nf o→n':>9s}")
    res = {}
    for k in sorted(old.keys(), key=lambda x: (GROUPS.index(x[0]), x[1])):
        if k not in new:
            print(f"{k[0]:16s} {k[1]:3d}  (교정 후 표본 부족)")
            continue
        fo, mo = old[k]
        fn, mn = new[k]
        cs = frob_cos(fo["C_steer"], fn["C_steer"])
        rel = float(np.linalg.norm(fo["C_steer"] - fn["C_steer"]) / max(np.linalg.norm(fo["C_steer"]), 1e-12))
        lo, no5 = spec(fo["C_steer"])
        ln, nn5 = spec(fn["C_steer"])
        cf = frob_cos(fo["C_failure"], fn["C_failure"])
        print(f"{k[0]:16s} {k[1]:3d} {cs:11.4f} {rel:7.3f} {lo:6.3f}→{ln:5.3f} "
              f"{no5:4d}→{nn5:3d} {cf:10.4f} {mo['n_failure']:4d}→{mn['n_failure']:3d}")
        res[f"{k[0]}/L{k[1]}"] = {"cos_Csteer": cs, "relF": rel, "lmax_old": lo, "lmax_new": ln,
                                  "n_gt05_old": no5, "n_gt05_new": nn5, "cos_Cfail": cf,
                                  "Nf_old": mo["n_failure"], "Nf_new": mn["n_failure"],
                                  "Ns_old": mo["n_success"], "Ns_new": mn["n_success"]}
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(res, indent=1))
        print(f"\nsaved {args.out_json}")


if __name__ == "__main__":
    main()
