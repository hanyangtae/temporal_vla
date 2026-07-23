#!/usr/bin/env python3
"""exp4-1 연산자 NPZ 검수표 — 배포 전 게이트 확인 (rollout 쓰기 전 필수).

확인 항목 (24a §4 + 07-23 교정):
  · 처치: layer, 세그먼트 갭, 이동/갭 중앙 ≤ MOVE_GAP_RATIO_MAX
  · 위약: max_s |cos_seg| ≤ PL_COS_MAX (준직교), dose-match 스케일(seg_mask)과 클립 여부
  · LOO: ep{E} 디렉토리 수 = fit-풀 대상 수
  · NPZ 실키 검증 (v_seg 단위벡터·bounds 끝 == T·mask 길이)

사용: python inspect_npz.py --npz-root <npz 루트> [--cell <cell>] [--json <out>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TREAT = ("setM_permanent", "setM_future_only")
PLACEBO = ("setM_permanent_placebo", "setM_future_only_placebo")
LOO_DIRS = ("setM_permanent_loo", "setM_permanent_placebo_loo",
            "setM_future_only_loo", "setM_future_only_placebo_loo")
GATED = ("setM_gated", "setM_gated_placebo",
         "setM_gated_future_only", "setM_gated_future_only_placebo")
MOVE_GAP_RATIO_MAX = 3.0
PL_COS_MAX = 0.3


def load_one(d: Path) -> dict | None:
    """<...>/dit_L*/ 디렉토리 하나 → NPZ 실키 + metadata 요약."""
    npz_p = next(d.glob("conceptors.npz"), None)
    meta_p = d / "metadata.json"
    if npz_p is None or not meta_p.exists():
        return None
    z = np.load(npz_p)
    meta = json.loads(meta_p.read_text())
    out = {"dir": str(d), "layer": meta.get("layer"), "operator": meta.get("operator")}
    if "alpha0_v_seg" in z.files:
        v = z["alpha0_v_seg"].astype(np.float64)
        s = z["alpha0_s_tok"].astype(np.float64).reshape(-1)
        b = z["alpha0_seg_bounds"].astype(int)
        mk = z["alpha0_seg_mask"].astype(np.float64).reshape(-1)
        nrm = np.linalg.norm(v, axis=1)
        out.update({
            "npz_ok": bool(np.all((nrm > 0.99) & (nrm < 1.01)) and int(b[-1, 1]) == len(s)
                           and len(mk) == v.shape[0]),
            "T": int(len(s)), "S": int(v.shape[0]), "mask": [round(float(x), 3) for x in mk],
        })
    else:
        out["npz_ok"] = False
        out["missing_keys"] = z.files
    sd = meta.get("seg_diag") or {}
    if sd:
        out["gaps"] = {g["segment"]: round(g["gap_fail_minus_succ"], 1)
                       for g in sd.get("segments", [])}
        out["move_over_gap"] = (round(sd.get("move_over_gap_median", float("nan")), 2),
                                round(sd.get("move_over_gap_max", float("nan")), 2))
        out["move_gate_ok"] = sd.get("move_over_gap_median", 9e9) <= MOVE_GAP_RATIO_MAX
    for k in ("cos_seg_vs_treat", "cos_seg_max", "dose_match_scale", "scale_clipped",
              "perm_id", "seg_dose", "seg_dose_treat"):
        if k in meta:
            out[k] = meta[k]
    if "pl_pool" in meta:
        out["pl_pool"] = meta["pl_pool"]
    if "cos_seg_vs_treat" in out and "cos_seg_max" not in out:
        out["cos_seg_max"] = max(abs(c) for c in out["cos_seg_vs_treat"])
    out["ortho_ok"] = (out.get("cos_seg_max") is None
                       or out["cos_seg_max"] <= PL_COS_MAX + 1e-9)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-root", type=Path, required=True)
    ap.add_argument("--cell", action="append", default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    cells = args.cell or sorted(p.name for p in args.npz_root.iterdir() if p.is_dir())
    report, bad = {}, []
    for cell in cells:
        croot = args.npz_root / cell
        entry = {}
        for arm in TREAT + PLACEBO:
            for d in sorted((croot / arm).glob("steer/dit_L*")):
                r = load_one(d)
                if r:
                    entry[arm] = r
        for arm in LOO_DIRS:
            eps = sorted((croot / arm).glob("ep*/dit_L*"))
            if not eps:
                continue
            samples = [load_one(d) for d in eps]
            samples = [s for s in samples if s]
            entry[arm] = {
                "n_loo": len(samples),
                "layers": sorted({s["layer"] for s in samples}),
                "npz_ok_all": all(s["npz_ok"] for s in samples),
                "cos_seg_max_worst": (max((s.get("cos_seg_max") or 0) for s in samples)
                                      if any("cos_seg_max" in s for s in samples) else None),
                "scale_range": ([round(min(min(s["dose_match_scale"]) for s in samples), 2),
                                 round(max(max(s["dose_match_scale"]) for s in samples), 2)]
                                if all("dose_match_scale" in s for s in samples) else None),
            }
        for arm in GATED:
            phs = sorted(p.name for p in croot.glob(f"{arm}/*") if p.is_dir())
            if not phs:
                continue
            det = {}
            for ph in phs:
                d = next((croot / arm / ph).glob("dit_L*"), None)
                if d:
                    r = load_one(d)
                    if r:
                        det[ph] = {k: r.get(k) for k in
                                   ("cos_seg_max", "dose_match_scale", "mask",
                                    "scale_clipped", "npz_ok", "ortho_ok")}
            entry[arm] = {"phases": phs, "n_phases": len(phs), "detail": det}
        report[cell] = entry

        print(f"\n=== {cell} ===")
        for arm in TREAT + PLACEBO:
            r = entry.get(arm)
            if not r:
                print(f"  {arm:28s} (없음)")
                continue
            line = (f"  {arm:28s} L{r['layer']} npz={'ok' if r['npz_ok'] else 'BAD'} "
                    f"mask={r.get('mask')} 이동/갭={r.get('move_over_gap')}")
            if "cos_seg_vs_treat" in r:
                line += (f" cos_seg={[round(c, 2) for c in r['cos_seg_vs_treat']]}"
                         f"{'' if r['ortho_ok'] else ' ✗준직교'}")
            if r.get("scale_clipped"):
                line += " ⚠scale-clip"
            fb = (r.get("pl_pool") or {}).get("fallback")
            if fb:
                line += f" FALLBACK={fb}"
            print(line)
            if not r["npz_ok"] or not r.get("ortho_ok", True) or not r.get("move_gate_ok", True):
                bad.append(f"{cell}/{arm}")
        for arm in LOO_DIRS:
            r = entry.get(arm)
            if r:
                print(f"  {arm:28s} n={r['n_loo']} layers={r['layers']} "
                      f"npz={'ok' if r['npz_ok_all'] else 'BAD'} "
                      f"cos_max={r['cos_seg_max_worst']} scale={r['scale_range']}")
                if not r["npz_ok_all"] or (r["cos_seg_max_worst"] or 0) > PL_COS_MAX + 1e-9:
                    bad.append(f"{cell}/{arm}")
        for arm in GATED:
            r = entry.get(arm)
            if r:
                worst = [f"{ph}:{d.get('cos_seg_max'):.2f}" for ph, d in r["detail"].items()
                         if d.get("cos_seg_max") is not None and not d.get("ortho_ok", True)]
                print(f"  {arm:28s} phases={r['n_phases']} {r['phases']}"
                      + (f" ✗준직교 {worst}" if worst else ""))
                if worst:
                    bad.append(f"{cell}/{arm}")

    print(f"\n[검수] 문제 {len(bad)}건" + (f": {bad}" if bad else " — 전 항목 통과"))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"[wrote] {args.json}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
