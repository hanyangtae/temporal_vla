#!/usr/bin/env python3
"""pq3 gated 성립 게이트 (계획서 v9 §arm gated fit 게이트, Codex R2#2 / Gate2 높음#6).

task-pooled fit manifest 를 입력으로:
  ① phase × scene × label 기여 표 (episode 수·record 수·dwell)
  ② phase 별 양 클래스 episode 하한 (task 합산 ≥ min-class)
  ③ phase 별 record 하한 (per-step 분할 반영 — record 수는 step 당 동일)
  ④ leave-one-scene-out C_steer 유사도 (phase 별 min over scenes; conceptor_overlap)
을 산출해 task 별 pass 판정 json 을 쓴다. build_pq3_queue 는 이 report 의 pass 없이
gated arm 을 생성하지 않는다. 승준(anaconda python)에서 실행 — full-token pkl 필요.

exit: 0 (report 생성 — pass 여부는 json 내). LOO fit 은 phase×scene 수만큼 conceptor
재계산이 필요하므로 α 는 대표 1개(--alpha, 기본 10)로 수행 (기하 진단 목적).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.conceptor import (  # noqa: E402
    and_conceptor,
    compute_conceptor,
    conceptor_overlap,
    not_conceptor,
)
from fit_phase_conceptor_n15 import (  # noqa: E402
    FULLTOKEN_MODE,
    gather_class_records_eps,
    load_rollout_fulltoken,
)


def load_manifest_rolls(manifest: Path, token_pool: str):
    """fit 로더와 동일한 계약 검사(kind/axes/layers/K/T/D 전 pkl 일치 — R2 중간#1)."""
    import pickle

    rolls, contract = [], None
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pkl, label, scene = line.split("\t")[:3]
        if not scene.strip().lstrip("-").isdigit():
            raise SystemExit(f"{manifest}: scene 열 비숫자 행 (fail-closed): {line[:60]}")
        with open(pkl, "rb") as f:
            d = pickle.load(f)
        if d.get("capture_token_mode") != FULLTOKEN_MODE:
            raise SystemExit(f"{pkl}: full-token pkl 아님 (gated 게이트는 pq3 수집분 전용)")
        r = load_rollout_fulltoken(d, pkl, token_pool)
        sig = (d.get("feature_kind"), tuple(d.get("feature_axes") or []),
               tuple(r["capture_layers"] or []),
               int(r["dit_k"].shape[2]), int(r["token_count"] or -1),
               int(r["dit_k"].shape[3]))
        if contract is None:
            contract = sig
        elif sig != contract:
            raise SystemExit(f"{pkl}: 계약 혼입 (kind,axes,layers,K,T,D)={sig} != {contract}")
        r["success"] = int(label)
        r["scene"] = scene
        rolls.append(r)
    if not rolls:
        raise SystemExit(f"manifest 비어 있음: {manifest}")
    return rolls


def steer_conceptor(rolls, phase, lk, k, alpha):
    rolls_k = [{**r, "dit": r["dit_k"][:, :, k, :]} for r in rolls]
    Xs, ns = gather_class_records_eps(rolls_k, phase, lk, 1)
    Xf, nf = gather_class_records_eps(rolls_k, phase, lk, 0)
    if Xs.size == 0 or Xf.size == 0 or Xs.shape[0] < 2 or Xf.shape[0] < 2:
        return None, ns, nf
    Cs = compute_conceptor(Xs, alpha)
    Cf = compute_conceptor(Xf, alpha)
    return and_conceptor(Cs, not_conceptor(Cf)), ns, nf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", required=True,
                    help="task=manifest 쌍 콤마 목록 (예: OpenDrawer=/path/fit_drawer.tsv,PickPlaceCounterToCabinet=/path/fit_ppcc.tsv)")
    ap.add_argument("--phases", required=True,
                    help="task=phase1+phase2 형식 (Gate D 에서 확정한 병합 후 phase 집합)")
    ap.add_argument("--layer", type=int, required=True, help="Stage1 gated 선택 layer (물리 번호)")
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--token-pool", default="mean")
    ap.add_argument("--min-class-eps", type=int, default=3)
    ap.add_argument("--record-floor", type=int, default=0,
                    help="phase 당 클래스별 record 하한 (0=미적용 — Gate D 에서 dwell 보고 확정)")
    ap.add_argument("--loo-floor", type=float, default=0.0,
                    help="LOO 유사도 하한 (0=보고만 — Gate D 에서 확정)")
    ap.add_argument("--enforce", action="store_true",
                    help="판정용 실행 (R2 치명#2): record/loo floor >0 필수, 전 scene LOO "
                         "성공 필수 (report-only 기본과 구분 — queue 는 enforce 산출만 소비)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.enforce and (args.record_floor <= 0 or args.loo_floor <= 0):
        raise SystemExit("--enforce 는 --record-floor·--loo-floor 명시(>0) 필수")

    phase_map = {}
    for part in args.phases.split(","):
        task, plist = part.split("=")
        phase_map[task.strip()] = [p.strip() for p in plist.split("+") if p.strip()]

    report = {}
    for part in args.tasks.split(","):
        task, manifest = part.split("=")
        task = task.strip()
        rolls = load_manifest_rolls(Path(manifest.strip()), args.token_pool)
        cap = rolls[0]["capture_layers"]
        if args.layer not in list(cap):
            raise SystemExit(f"{task}: layer {args.layer} 가 capture_layers {cap} 에 없음")
        lk = list(cap).index(args.layer)
        K = rolls[0]["dit_k"].shape[2]
        scenes = sorted({r["scene"] for r in rolls})
        phases = phase_map.get(task)
        if not phases:
            raise SystemExit(f"--phases 에 task {task} 누락")

        table, checks, loo = {}, [], {}
        for ph in phases:
            row = {"per_scene": {}, "eps_s": 0, "eps_f": 0, "records_s": 0, "records_f": 0}
            for sc in scenes:
                sub = [r for r in rolls if r["scene"] == sc]
                rolls_0 = [{**r, "dit": r["dit_k"][:, :, 0, :]} for r in sub]
                Xs, ns = gather_class_records_eps(rolls_0, ph, lk, 1)
                Xf, nf = gather_class_records_eps(rolls_0, ph, lk, 0)
                row["per_scene"][sc] = {
                    "eps_s": ns, "eps_f": nf,
                    "recs_s": int(Xs.shape[0]) if Xs.size else 0,
                    "recs_f": int(Xf.shape[0]) if Xf.size else 0,
                }
                row["eps_s"] += ns; row["eps_f"] += nf
                row["records_s"] += row["per_scene"][sc]["recs_s"]
                row["records_f"] += row["per_scene"][sc]["recs_f"]
            class_ok = row["eps_s"] >= args.min_class_eps and row["eps_f"] >= args.min_class_eps
            record_ok = (args.record_floor <= 0 or
                         min(row["records_s"], row["records_f"]) >= args.record_floor)
            checks.append({"phase": ph, "class_ok": class_ok, "record_ok": record_ok})

            # LOO(scene) 유사도 — step 평균 (per-step 배치와 동일 객체 기준)
            sims = []
            for sc in scenes:
                keep = [r for r in rolls if r["scene"] != sc]
                per_step_sims = []
                for k in range(K):
                    C_full, _, _ = steer_conceptor(rolls, ph, lk, k, args.alpha)
                    C_loo, _, _ = steer_conceptor(keep, ph, lk, k, args.alpha)
                    if C_full is None or C_loo is None:
                        per_step_sims = []
                        break
                    per_step_sims.append(conceptor_overlap(C_full, C_loo))
                if per_step_sims:
                    sims.append({"scene": sc, "sim": float(np.mean(per_step_sims))})
            loo[ph] = {"per_scene": sims,
                       "min": min((s["sim"] for s in sims), default=None)}
            table[ph] = row

        if args.enforce:
            # 판정용: 모든 scene 의 LOO fit 이 성공하고 min ≥ floor 여야 함
            # (일부 scene LOO 실패가 sims 에서 빠져 통과하는 구멍 봉쇄 — R2 치명#2)
            loo_ok = all(
                len(loo[ph]["per_scene"]) == len(scenes)
                and loo[ph]["min"] is not None
                and loo[ph]["min"] >= args.loo_floor
                for ph in phases
            )
        else:
            loo_ok = all(
                (loo[ph]["min"] is None and args.loo_floor <= 0)
                or (loo[ph]["min"] is not None and loo[ph]["min"] >= args.loo_floor)
                for ph in phases
            )
        task_pass = all(c["class_ok"] and c["record_ok"] for c in checks) and loo_ok
        import hashlib as _hl
        manifest_sha = _hl.sha256(Path(manifest.strip()).read_bytes()).hexdigest()[:16]
        report[task] = {
            "pass": bool(task_pass),
            "enforce": bool(args.enforce),
            "layer": args.layer, "alpha": args.alpha, "phases": phases,
            "manifest": manifest.strip(), "manifest_sha": manifest_sha,
            "token_pool": args.token_pool, "num_denoise_steps": int(K),
            "scenes": scenes,
            "min_class_eps": args.min_class_eps, "record_floor": args.record_floor,
            "loo_floor": args.loo_floor,
            "table": table, "checks": checks, "loo": loo,
            "note": "pass=False 인 task 는 H2/H3 전체 N/A (부분 cell 탈락 금지 — Codex R2#2). "
                    "queue 는 enforce=True report 만 소비할 것",
        }
        print(f"[gated-gate] {task}: pass={task_pass} checks={checks} "
              f"loo_min={{ {', '.join(f'{p}:{loo[p]['min']}' for p in phases)} }}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[gated-gate] -> {args.out}")


if __name__ == "__main__":
    main()
