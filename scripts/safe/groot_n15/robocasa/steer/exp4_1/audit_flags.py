#!/usr/bin/env python3
"""exp4-1: latch 적용 무결성 감사 (24a §3, R5 무음 미적용 방지).

steering arm rollout 사이드카의 phase_gated_flags 를 t0 manifest 의 K(t0_record)와 대조:
  sum(flags) == n_inferences − K  &&  first_true == K
(K ≥ n_inferences 면 전부 False). 불일치 rollout 은 quarantine/ 로 이동(사이드카+mp4)하고
목록을 출력 — 집계 전 필수 실행.

사용:
  python audit_flags.py --t0-manifest <t0_manifest.tsv> --arm-root <arm rollout 루트> \
      [--arm A0] [--quarantine]
  A0 arm 은 --steer-from-record 미사용이라 flags 전부 False + steer_from_record null 검사.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def load_t0(path: Path) -> dict:
    lines = path.read_text().splitlines()
    header = lines[0].split("\t")
    out = {}
    for ln in lines[1:]:
        if not ln.strip():
            continue
        r = dict(zip(header, ln.split("\t")))
        out[(r["cell"], int(r["episode_idx"]), int(r["inference_seed"]))] = r["t0_record"]
    return out


def audit_one(sidecar: Path, t0map: dict, arm: str) -> tuple[bool, str]:
    d = json.loads(sidecar.read_text())
    cell = d.get("cell_id") or sidecar.parent.name
    key = (cell, int(d.get("episode_idx", -1)), int(d.get("inference_seed", -1)))
    flags = d.get("phase_gated_flags")
    n_inf = int(d.get("n_inferences", -1))
    if flags is None or len(flags) != n_inf:
        return False, f"flags 길이 {None if flags is None else len(flags)} != n_inferences {n_inf}"
    if arm == "A0":
        if d.get("steer_from_record") is not None:
            return False, "A0 인데 steer_from_record 설정됨"
        if any(flags):
            return False, "A0 인데 gated flag True 존재"
        return True, "ok(A0)"
    k_str = t0map.get(key)
    if k_str is None:
        return False, f"t0 manifest 에 없는 episode: {key}"
    if k_str == "NA":
        return False, f"미주석(t0=NA) episode 가 steering arm 에 존재: {key}"
    k = int(k_str)
    if d.get("steer_from_record") != k:
        return False, f"steer_from_record {d.get('steer_from_record')} != manifest K {k}"
    if d.get("steer_phase_mode") == "current":
        # gated arm: K 이후 flag[i] == (그 시점 phase 가 serve 에 등록됨) — 미등록 phase 는
        # identity 폴백이 정상. 불변식: K 전 전부 False + K 이후는 등록 phase 집합과 일치.
        if any(flags[:k]):
            return False, f"gated: K={k} 이전에 True 존재"
        reg = set((d.get("serve_steering") or {}).get("phases") or [])
        ph = d.get("feature_phases") or []
        if len(ph) == n_inf and reg:
            bad = [i for i in range(k, n_inf) if flags[i] != (ph[i] in reg)]
            if bad:
                return False, f"gated: flag≠(phase∈등록집합) {len(bad)}건 (첫 idx {bad[0]})"
        elif not any(flags[k:]) and n_inf > k:
            return False, "gated: K 이후 전부 False (등록 phase 미통과 의심)"
        return True, "ok(gated)"
    want_true = max(0, n_inf - k)
    if sum(flags) != want_true:
        return False, f"sum(flags)={sum(flags)} != n_inf−K={want_true}"
    if want_true > 0:
        first = flags.index(True)
        if first != k:
            return False, f"first_true={first} != K={k}"
    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0-manifest", type=Path, required=True)
    ap.add_argument("--arm-root", type=Path, required=True)
    ap.add_argument("--arm", default="steer", help="'A0' 면 무개입 검사 모드")
    ap.add_argument("--quarantine", action="store_true", help="불일치 rollout 격리 이동")
    args = ap.parse_args()
    t0map = load_t0(args.t0_manifest)
    bad, ok = [], 0
    sidecars = sorted(args.arm_root.rglob("*--succ*.json"))
    for sc in sidecars:
        good, why = audit_one(sc, t0map, args.arm)
        if good:
            ok += 1
            continue
        bad.append((sc, why))
        print(f"[BAD] {sc.name}: {why}")
        if args.quarantine:
            qdir = args.arm_root / "quarantine"
            qdir.mkdir(exist_ok=True)
            for ext in (".json", ".mp4"):
                p = sc.with_suffix(ext)
                if p.exists():
                    shutil.move(str(p), qdir / p.name)
    print(f"[audit] ok={ok} bad={len(bad)} / {len(sidecars)} ({args.arm_root})")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
