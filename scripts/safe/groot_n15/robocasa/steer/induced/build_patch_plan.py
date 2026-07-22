"""exp4-2 Track I P0 arm plan 빌더 — baseline 앵커 + donor meta → arm_plan.tsv.

변형 (24b §2.2): B2(DiT L15, passB succ donor 재사용, 타 phase), B4(donor 통계 매칭
noise, make_noise_npz.py 산출), B3(타 task donor — 수집 후 동일 스키마), B1(VL donor —
extract_vl_donor_npz 산출, pathway=vl).

창 규약: W ∈ {3,6} records, start_record = target ep 의 phase 앵커 (B2/B3=transport 초입,
B1=reach 초입). donor_start = donor meta_json feature_phases 의 앵커 (B2=insert-settle
시작 — 타 phase 주입) + assert (donor phase 오정렬 방어, 24b §6).

  python build_patch_plan.py --baseline-dir <.../p0/baseline/raw_rollouts/<TASK>/<CELL>> \
      --donor-glob '<.../patchceil/*/donors/ep*_L15.npz>' --noise-dir <.../p0/noise> \
      --out <.../p0/arm_plan.tsv> [--n-targets 4] [--windows 3,6]

arm_plan.tsv 열: variant  pathway  ep_idx  inference_seed  npz  start_record  donor_start  patch_len  tag
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

# 6p proximity 라벨 어휘: reach-to-object/grasp/place/insert-settle ("transport" 없음).
# 24b 의 "transport 초입" ≈ 6p 의 "place" 시작 (파지 후 이동 구간).
TARGET_PHASE = {"b1": "reach-to-object", "b2": "place", "b3": "place", "b4": "place"}
DONOR_PHASE_B2 = "insert-settle"


def _phase_anchor(phases: list[str], phase: str) -> int | None:
    for i, p in enumerate(phases):
        if p == phase:
            return i
    return None


def _npz_meta(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return json.loads(bytes(z["meta_json"]).decode()) if "meta_json" in z else {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-dir", required=True)
    ap.add_argument("--donor-glob", required=True, help="B2 donor NPZ glob (passB ep*_L15.npz)")
    ap.add_argument("--noise-dir", default=None, help="B4 noise NPZ 디렉토리 (없으면 B4 생략)")
    ap.add_argument("--n-targets", type=int, default=4, help="변형당 target 성공 ep 수")
    ap.add_argument("--windows", default="3,6")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = Path(args.baseline_dir)
    targets = []  # (ep_idx, inf, anchors{phase:record})
    for p in sorted(base.glob("task*--ep*--succ1.json")):
        d = json.loads(p.read_text())
        m = re.search(r"--ep(\d+)--", p.name)
        if m is None:
            continue
        phases = d.get("phase_timeline") or d.get("feature_phases") or []
        targets.append((int(m.group(1)), int(d.get("inference_seed") or 0), phases))
    if not targets:
        print("ABORT: baseline 성공 사이드카 없음", file=sys.stderr)
        return 2

    donors = sorted(glob.glob(args.donor_glob))
    if not donors:
        print(f"ABORT: donor 없음 ({args.donor_glob})", file=sys.stderr)
        return 2
    windows = [int(w) for w in args.windows.split(",")]

    rows = []

    def add(variant, pathway, ep_idx, inf, npz, t0, d0, plen):
        tag = f"{variant}_ep{ep_idx}_t{t0}_w{plen}"
        rows.append((variant, pathway, ep_idx, inf, npz, t0, d0, plen, tag))

    # B2: passB succ donor, target transport 초입 ← donor insert-settle 시작 (타 phase)
    for w in windows:
        for i, (ep_idx, inf, phases) in enumerate(targets[: args.n_targets]):
            t0 = _phase_anchor(phases, TARGET_PHASE["b2"])
            if t0 is None:
                print(f"WARN: ep{ep_idx} transport 앵커 없음 — skip")
                continue
            dn = donors[i % len(donors)]
            meta = _npz_meta(dn)
            d_phases = meta.get("feature_phases") or []
            d0 = _phase_anchor(d_phases, DONOR_PHASE_B2)
            if d0 is None:
                print(f"WARN: donor {Path(dn).name} 에 {DONOR_PHASE_B2} 없음 — skip")
                continue
            n_rec = int(meta.get("n_records") or 0)
            assert d0 + w <= n_rec, f"donor {dn} 창 초과 (d0={d0} w={w} R={n_rec})"
            add(f"b2_w{w}", "dit", ep_idx, inf, dn, t0, d0, w)

    # B4: noise NPZ (donor 통계 매칭) — 같은 창 규약
    if args.noise_dir:
        noises = sorted(Path(args.noise_dir).glob("*.npz"))
        for w in windows:
            for i, (ep_idx, inf, phases) in enumerate(targets[: args.n_targets]):
                t0 = _phase_anchor(phases, TARGET_PHASE["b4"])
                if t0 is None or not noises:
                    continue
                nz = noises[i % len(noises)]
                meta = _npz_meta(str(nz))
                d_phases = meta.get("feature_phases") or []
                d0 = _phase_anchor(d_phases, DONOR_PHASE_B2) or 0
                scale = meta.get("scale")
                add(f"b4_s{scale}_w{w}", "dit", ep_idx, inf, str(nz), t0, d0, w)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("variant\tpathway\tep_idx\tinference_seed\tnpz\tstart_record\tdonor_start\tpatch_len\ttag\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"wrote {out}: {len(rows)} rows "
          f"({len({r[0] for r in rows})} variants × targets)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
