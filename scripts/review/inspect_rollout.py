#!/usr/bin/env python3
"""수집 산출물(rollout pkl) 무결성·정합·bias 점검.

1261줄 수집 스크립트를 위에서 아래로 읽는 대신 **산출물로 거꾸로 검증**한다.
체크리스트 해설: docs/review/S1_verify_collect.md

    P=/home/dongkyu/miniconda3/envs/lerobot_safe/bin/python   # torch 필요
    $P scripts/review/inspect_rollout.py <pkl|디렉토리> [--limit N] [--summary]

점검 항목
  A. activation 캡처 — hidden_states/vl_hidden_states 존재·shape 일관·NaN/Inf·분산 0
  B. 분류 저장       — cell_id·scenario_seed·instruction·success 가 경로/파일명/pkl 에서 일치
  C. 축 정합         — records 수 vs actions vs states vs feature_phases vs action_vectors
  D. 결정성          — inference_seed 가 episode_idx 로부터 결정적으로 유도되는가
  E. bias 신호       — succ/fail 길이 분포(길이 confound), instruction 쏠림, phase 도달률
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

LEN_KEYS = ("hidden_states", "vl_hidden_states", "actions", "states", "feature_phases")


def find_pkls(target: Path) -> list[Path]:
    return [target] if target.is_file() else sorted(target.rglob("*.pkl"))


def parse_name(p: Path) -> dict:
    m = re.search(r"ep(\d+)--succ([01])", p.name)
    return {"name_ep": int(m.group(1)), "name_succ": int(m.group(2))} if m else {}


def arr(x):
    a = np.asarray(x.detach().cpu() if hasattr(x, "detach") else x)
    # ★ float16 그대로 std/제곱을 내면 오버플로로 inf 가 나온다 (실제 값은 유한).
    # activation 은 fp16 로 저장되고 absmax ~950 이라 제곱이 fp16 범위를 넘는다.
    # 실 파이프라인(src/conceptor/core.py float64, fit_*.py float32 캐스팅)은
    # 이미 캐스팅 후 계산하므로 안전하다 — 이 버그는 본 점검 도구에만 있었다.
    return a.astype(np.float32) if a.dtype == np.float16 else a


def stats(x) -> dict:
    a = arr(x)
    f = np.isfinite(a) if a.dtype.kind == "f" else np.ones_like(a, bool)
    return {"shape": tuple(a.shape), "dtype": str(a.dtype),
            "nan": int(np.isnan(a).sum()) if a.dtype.kind == "f" else 0,
            "inf": int((~f & ~np.isnan(a)).sum()) if a.dtype.kind == "f" else 0,
            "std": float(a[f].std()) if f.any() else 0.0,
            "absmax": float(np.abs(a[f]).max()) if f.any() else 0.0}


def inspect_one(p: Path, verbose: bool = True) -> dict:
    out: dict = {"path": str(p), "problems": []}
    out.update(parse_name(p))
    try:
        d = pickle.load(p.open("rb"))
    except Exception as e:  # noqa: BLE001 — 손상 탐지가 목적
        out["problems"].append(f"언피클 실패: {type(e).__name__}")
        return out
    if not isinstance(d, dict):
        out["problems"].append(f"최상위가 dict 아님: {type(d).__name__}")
        return out

    g = d.get
    out.update({k: g(k) for k in ("cell_id", "scenario_seed", "seed", "episode_idx",
                                  "episode_success", "canonical_instruction", "robocasa_task",
                                  "inference_seed", "n_action_steps", "feature_kind",
                                  "vl_feature_kind", "layer_count", "phase_scheme")})

    # ---- C. 축 정합
    lens = {k: len(g(k)) for k in LEN_KEYS if isinstance(g(k), list)}
    av = g("action_vectors")
    if av is not None:
        lens["action_vectors"] = int(arr(av).shape[0])
    out["lens"] = lens
    if len(set(lens.values())) > 1:
        out["problems"].append(f"record 축 길이 불일치: {lens}")
    n = max(lens.values()) if lens else 0
    out["n_records"] = n
    tl = g("phase_timeline")
    if isinstance(tl, list) and n:
        out["phase_timeline_len"] = len(tl)
        if len(tl) not in (n, n + 1):
            out["problems"].append(f"phase_timeline({len(tl)}) 이 record({n}) / record+1 어느 쪽도 아님")
    if isinstance(g("feature_phases"), list):
        out["phases"] = sorted(set(g("feature_phases")))

    # ---- A. activation
    for key in ("hidden_states", "vl_hidden_states"):
        v = g(key)
        if not isinstance(v, list) or not v:
            continue
        s0 = stats(v[0])
        out[key] = s0
        shapes = {tuple(arr(x).shape) for x in v}
        if len(shapes) > 1:
            out["problems"].append(f"{key} shape 이 record 간 불일치: {sorted(shapes)[:3]}")
        bad_nan = sum(1 for x in v if np.isnan(arr(x)).any())
        bad_inf = sum(1 for x in v if not np.isfinite(arr(x)).all())
        if bad_nan or bad_inf - bad_nan:
            out["problems"].append(f"{key}: NaN {bad_nan} record / Inf {bad_inf - bad_nan} record")
        zero_var = sum(1 for x in v if arr(x).std() == 0.0)
        if zero_var:
            out["problems"].append(f"{key}: 분산 0 record {zero_var}개 (캡처 실패 의심)")
    cl = g("capture_layers")
    if cl is not None:
        out["capture_layers"] = list(cl)
        hs = g("hidden_states")
        if isinstance(hs, list) and hs:
            sh = arr(hs[0]).shape
            if sh and sh[0] != len(cl):
                out["problems"].append(f"hidden_state 첫 축({sh[0]}) != capture_layers({len(cl)})")

    # ---- B. 분류 저장
    if out.get("name_succ") is not None and out.get("episode_success") is not None \
            and out["name_succ"] != out["episode_success"]:
        out["problems"].append(
            f"성공 라벨 불일치: 파일명 succ{out['name_succ']} vs pkl {out['episode_success']}")
    if out.get("name_ep") is not None and out.get("episode_idx") is not None \
            and out["name_ep"] != out["episode_idx"]:
        out["problems"].append(f"episode 불일치: 파일명 ep{out['name_ep']} vs pkl {out['episode_idx']}")
    if out.get("cell_id") and out["cell_id"] != p.parent.name:
        out["problems"].append(f"cell 불일치: 경로 {p.parent.name} vs pkl {out['cell_id']}")
    if out.get("robocasa_task") and out["robocasa_task"] != p.parent.parent.name:
        out["problems"].append(f"task 불일치: 경로 {p.parent.parent.name} vs pkl {out['robocasa_task']}")
    cid = out.get("cell_id") or ""
    m = re.search(r"_s(\d+)$", cid)
    if m and out.get("scenario_seed") is not None and int(m.group(1)) != out["scenario_seed"]:
        out["problems"].append(f"scene seed 불일치: cell_id {m.group(1)} vs scenario_seed {out['scenario_seed']}")
    if out.get("canonical_instruction") and g("task_description") \
            and out["canonical_instruction"].strip() != str(g("task_description")).strip():
        out["problems"].append("canonical_instruction != task_description (모델에 간 문장과 라벨이 다름)")

    if verbose:
        print(f"\n── {p.parent.name}/{p.name}")
        for k in ("cell_id", "scenario_seed", "episode_idx", "episode_success", "inference_seed",
                  "n_action_steps", "n_records", "phase_timeline_len", "feature_kind",
                  "vl_feature_kind", "capture_layers", "canonical_instruction"):
            if out.get(k) is not None:
                print(f"   {k:20s} {out[k]}")
        print(f"   {'lens':20s} {lens}")
        for k in ("hidden_states", "vl_hidden_states"):
            if k in out:
                s = out[k]
                print(f"   {k:20s} per-record {s['shape']} {s['dtype']} "
                      f"std={s['std']:.4g} absmax={s['absmax']:.4g}")
        if out.get("phases"):
            print(f"   {'phases':20s} {out['phases']}")
        for pr in out["problems"]:
            print(f"   ⚠ {pr}")
        if not out["problems"]:
            print("   ✓ 문제 없음")
    return out


def summarize(res: list[dict]) -> None:
    print("\n" + "=" * 70)
    print(f"요약 — {len(res)} rollout")
    print("=" * 70)
    probs = [r for r in res if r["problems"]]
    print(f"문제 있는 rollout: {len(probs)} / {len(res)}")
    for k, v in Counter(pr.split(":")[0].split("(")[0].strip()
                        for r in res for pr in r["problems"]).most_common():
        print(f"   {v:4d}  {k}")

    # D. 결정성 — inference_seed 유도 규칙
    pairs = [(r["episode_idx"], r["inference_seed"]) for r in res
             if r.get("episode_idx") is not None and r.get("inference_seed") is not None]
    if pairs:
        ratios = {s / e for e, s in pairs if e}
        print(f"\n결정성 — inference_seed / episode_idx = {sorted(ratios)[:3]}"
              f"{' (일관)' if len(ratios) == 1 else ' ⚠ 불일관'}")

    # E. 길이 confound
    by = defaultdict(list)
    for r in res:
        if r.get("episode_success") is not None and r.get("n_records"):
            by[int(r["episode_success"])].append(r["n_records"])
    if len(by) == 2:
        print("\n길이 confound — record 수 분포")
        for s in (1, 0):
            v = np.array(by[s])
            print(f"   succ={s}  n={v.size:3d}  mean={v.mean():6.2f}  min={v.min():3d}  max={v.max():3d}")
        a, b = np.array(by[1]), np.array(by[0])
        lo, hi = max(a.min(), b.min()), min(a.max(), b.max())
        if lo > hi:
            print("   ⚠ 범위가 전혀 겹치지 않음 — 길이만으로 라벨이 완전 결정된다(AUROC 1.0)")
        else:
            n_ov = int(((a >= lo) & (a <= hi)).sum() + ((b >= lo) & (b <= hi)).sum())
            print(f"   겹침 구간 [{lo}, {hi}] 안의 rollout {n_ov}/{len(res)} "
                  f"— 길이 통제 분석은 이 구간에서만 유효")
    elif by:
        s = next(iter(by))
        print(f"\n⚠ 한 클래스만 존재 (succ={s}, n={len(by[s])}) — 대조 불성립")

    # instruction 쏠림
    ins = defaultdict(list)
    for r in res:
        if r.get("canonical_instruction") is not None and r.get("episode_success") is not None:
            ins[r["canonical_instruction"]].append(int(r["episode_success"]))
    if len(ins) > 1:
        print("\ninstruction 쏠림 — instruction 별 SR (편차 크면 VL 분리는 아티팩트 위험)")
        for k, v in sorted(ins.items(), key=lambda kv: -len(kv[1])):
            print(f"   SR {np.mean(v):.2f}  n={len(v):3d}  {str(k)[:60]}")

    # phase 도달률 (선택 효과)
    ph = Counter()
    for r in res:
        for x in r.get("phases", []):
            ph[x] += 1
    if ph:
        print(f"\nphase 도달률 (전체 {len(res)} rollout 중 그 phase 가 등장한 수)")
        for k, v in ph.most_common():
            print(f"   {v:4d}  {k}")
        print("   → 후반 phase 도달률이 낮으면 phase 별 분석에 선택 효과가 낀다")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    pkls = find_pkls(Path(a.target))
    if not pkls:
        print("pkl 없음", file=sys.stderr)
        return 1
    if a.limit:
        pkls = pkls[:a.limit]
    print(f"검사 대상 {len(pkls)}개")
    summarize([inspect_one(p, verbose=not a.summary) for p in pkls])
    return 0


if __name__ == "__main__":
    sys.exit(main())
