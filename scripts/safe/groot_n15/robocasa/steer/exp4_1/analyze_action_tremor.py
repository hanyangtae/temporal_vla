#!/usr/bin/env python3
"""개입(steering)이 액션 궤적에 준 떨림(jitter) 을 base 대비 정량화 — exp4-1 진단.

collector(`http_feature_collect.py`)가 사이드카에 남긴 `action_kinematics`(record 해상도
speed/jerk + 개입 전후 split)를 읽어, **개입 이후(post) 구간**에서 각 steering arm 의 떨림을
base arm(`noise_resample`: t0 에 reseed 만·steering 없음)과 **같은 episode 로 paired 비교**한다.

  Δjerk_post = jerk_post(steered) − jerk_post(noise_resample)      (>0 = steering 이 떨림 추가)
  ratio      = jerk_post(steered) / jerk_post(noise_resample)

noise_resample 을 base 로 쓰므로 Δ 는 "단순 재샘플(noise)을 넘어 steering 방향이 유발한 떨림"을
분리한다. arm 자체의 pre→post 증가(jerk_post/jerk_pre)도 함께 낸다. GPU·serve 무접촉(사이드카만).

사용: analyze_action_tremor.py --eval-root <exp4_1/eval> [--pool eval|fit] [--out <json>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

BASE_ARM = "noise_resample"
STEER_ARMS = [
    "setM_permanent", "setM_gated", "setM_future_only", "setM_gated_future_only",
    "conceptor_permanent", "conceptor_gated",
    # placebo 는 뒤로 미뤄져 있음 — 생기면 자동 포함
    "setM_permanent_placebo", "setM_gated_placebo",
    "setM_future_only_placebo", "setM_gated_future_only_placebo",
]

# ★핵심 비교: future_only(future 세그먼트만·mask[0,1,0])가 full(전 세그먼트·mask[1,1,1])보다
# 더 부드럽게(낮은 jerk) 개입하는가. (future_only arm, full counterpart) 쌍 — 같은 episode paired.
FUTURE_ONLY_PAIRS = [
    ("setM_future_only", "setM_permanent"),
    ("setM_gated_future_only", "setM_gated"),
    # conceptor_future_only 는 아직 arm 미존재 (사용자 검토 중) — 생기면 추가
    ("conceptor_future_only", "conceptor_permanent"),
    ("conceptor_gated_future_only", "conceptor_gated"),
]


def _load_sidecars(arm_dir: Path, pool: str) -> dict:
    """episode_idx → action_kinematics (있는 것만)."""
    out = {}
    for j in (arm_dir / pool / "raw_rollouts").rglob("*--ep*--succ*.json"):
        try:
            d = json.loads(j.read_text())
        except Exception:
            continue
        k = d.get("action_kinematics")
        if k is None:
            continue
        out[int(d["episode_idx"])] = {
            "kin": k,
            "success": int(d.get("episode_success", 0)),
        }
    return out


def _post(kin: dict) -> float | None:
    p = kin.get("post_intervention")
    return None if p is None else p.get("jerk_mean")


def _pre(kin: dict) -> float | None:
    p = kin.get("pre_intervention")
    return None if p is None else p.get("jerk_mean")


def _post_speed(kin: dict) -> float | None:
    p = kin.get("post_intervention")
    return None if p is None else p.get("speed_mean")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--pool", default="eval", choices=["eval", "fit"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cells = sorted(p.name for p in args.eval_root.iterdir()
                   if p.is_dir() and (p / BASE_ARM / args.pool).exists())
    report = {"pool": args.pool, "cells": {}}
    print(f"{'cell':16} {'arm':30} {'n':>3} {'base_post':>9} {'arm_post':>9} "
          f"{'Δjerk':>8} {'ratio':>6} {'post/pre':>8}")
    print("-" * 100)
    for cell in cells:
        cdir = args.eval_root / cell
        base = _load_sidecars(cdir / BASE_ARM, args.pool)
        if not base:
            continue
        cell_rep = {}
        for arm in STEER_ARMS:
            adir = cdir / arm
            if not (adir / args.pool).exists():
                continue
            steered = _load_sidecars(adir, args.pool)
            eps = sorted(set(base) & set(steered))
            rows = []
            for ep in eps:
                bp = _post(base[ep]["kin"])
                sp = _post(steered[ep]["kin"])
                if bp is None or sp is None or bp <= 0:
                    continue
                pre = _pre(steered[ep]["kin"])
                rows.append({
                    "ep": ep, "base_post": bp, "arm_post": sp,
                    "delta": sp - bp, "ratio": sp / bp,
                    "post_over_pre": (sp / pre if pre and pre > 0 else None),
                    "arm_post_speed": _post_speed(steered[ep]["kin"]),
                })
            if not rows:
                continue
            n = len(rows)
            bp_m = median(r["base_post"] for r in rows)
            ap_m = median(r["arm_post"] for r in rows)
            dj = median(r["delta"] for r in rows)
            rt = median(r["ratio"] for r in rows)
            pp = [r["post_over_pre"] for r in rows if r["post_over_pre"] is not None]
            pp_m = median(pp) if pp else float("nan")
            cell_rep[arm] = {
                "n": n, "base_post_jerk": bp_m, "arm_post_jerk": ap_m,
                "delta_jerk_median": dj, "ratio_median": rt,
                "post_over_pre_median": pp_m,
                "per_episode": rows,
            }
            print(f"{cell:16} {arm:30} {n:3d} {bp_m:9.4f} {ap_m:9.4f} "
                  f"{dj:+8.4f} {rt:6.2f} {pp_m:8.2f}")
        report["cells"][cell] = cell_rep

    # ── ★future_only vs full 개입 매끄러움 대조 (사용자 핵심 질문) ──────────────────
    # Δjerk = jerk_post(future_only) − jerk_post(full).  < 0 → future_only 가 더 부드러움.
    print("\n" + "=" * 78)
    print("future_only vs full — 개입 후 jerk 비교 (Δ<0 = future_only 가 더 부드러움)")
    print("=" * 78)
    print(f"{'cell':16} {'future_only vs full':34} {'n':>3} {'fo_jerk':>8} {'full_jerk':>9} "
          f"{'Δjerk':>8} {'ratio':>6}")
    print("-" * 90)
    fo_report = {}
    for cell in cells:
        cdir = args.eval_root / cell
        for fo_arm, full_arm in FUTURE_ONLY_PAIRS:
            fo = _load_sidecars(cdir / fo_arm, args.pool)
            full = _load_sidecars(cdir / full_arm, args.pool)
            eps = sorted(set(fo) & set(full))
            rows = []
            for ep in eps:
                jf = _post(fo[ep]["kin"])
                jl = _post(full[ep]["kin"])
                if jf is None or jl is None or jl <= 0:
                    continue
                rows.append({"ep": ep, "fo_jerk": jf, "full_jerk": jl,
                             "delta": jf - jl, "ratio": jf / jl})
            if not rows:
                continue
            n = len(rows)
            fj = median(r["fo_jerk"] for r in rows)
            lj = median(r["full_jerk"] for r in rows)
            dj = median(r["delta"] for r in rows)
            rt = median(r["ratio"] for r in rows)
            fo_report[f"{cell}:{fo_arm}_vs_{full_arm}"] = {
                "n": n, "fo_jerk": fj, "full_jerk": lj,
                "delta_jerk_median": dj, "ratio_median": rt, "per_episode": rows,
            }
            smoother = "  ← 더 부드러움" if dj < 0 else ""
            print(f"{cell:16} {fo_arm+' vs '+full_arm:34} {n:3d} {fj:8.4f} {lj:9.4f} "
                  f"{dj:+8.4f} {rt:6.2f}{smoother}")
    report["future_only_vs_full"] = fo_report

    if args.out:
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
