"""exp4-2 P0 캘리브레이션 grid 생성 — baseline 사이드카 앵커 → spec json + grid.tsv.

Phase A baseline(`--no-features` 사이드카)에서 성공 episode 와 grasp 앵커(record)를 읽어,
4모드(C1/G1/P1/P2) × config × episode 행을 만든다. 모든 확률량은 spec_seed 로 결정
(spec_seed = sha256(mode|config|ep) 파생 — tsv 재생성이 곧 재현).

  python build_perturb_grid.py --baseline-dir <.../p0/baseline/raw_rollouts/<TASK>/<CELL>> \
      --out-dir <.../p0> [--eps-per-config 4] [--n-cal 12]

출력: <out>/specs/<config>/ep{idx}.json, <out>/grid.tsv
grid.tsv 열: mode  config  ep_idx  inference_seed  spec(컨테이너 경로 아님 — 러너가 변환)  tag
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# config 격자 (24b 개정 메뉴): C1 scale × WAM σ / G1 σxyz / P1 δ / P2 F×dur
C1_SCALES = [0.5, 1.0, 2.0]
G1_SIGMAS = [0.05, 0.10, 0.15]
P1_MAGS = [0.03, 0.08, 0.15]
P2_GRID = [(5.0, 2), (15.0, 2), (40.0, 2), (5.0, 5), (15.0, 5), (40.0, 5)]


def spec_seed(mode: str, config: str, ep: int) -> int:
    h = hashlib.sha256(f"{mode}|{config}|{ep}".encode()).hexdigest()
    return int(h[:8], 16) % (2**31)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-dir", required=True,
                    help="Phase A 사이드카 디렉토리 (task*--ep*--succ*.json)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-cal", type=int, default=12, help="캘리브레이션 성공 ep 수")
    ap.add_argument("--eps-per-config", type=int, default=4,
                    help="config 당 1라운드 ep 수 (톱업은 값 키워 재생성 — 결정적 재현)")
    ap.add_argument("--modes", default="C1,G1,P1,P2")
    ap.add_argument("--sham-eps", type=int, default=1,
                    help="모드당 sham 행 ep 수 (P0 중 배선 드리프트 감시)")
    args = ap.parse_args()

    base = Path(args.baseline_dir)
    eps = []  # (ep_idx, inf_seed, grasp_record)
    for p in sorted(base.glob("task*--ep*--succ1.json")):
        d = json.loads(p.read_text())
        m = re.search(r"--ep(\d+)--", p.name)
        if m is None:
            continue
        ep_idx = int(m.group(1))
        grasp = (d.get("event_steps") or {}).get("grasp:obj")
        eps.append((ep_idx, int(d.get("inference_seed") or 0), grasp))
    if len(eps) < args.n_cal:
        print(f"ABORT: 성공 baseline {len(eps)}개 < --n-cal {args.n_cal} — Phase A 부족",
              file=sys.stderr)
        return 2
    eps = eps[: args.n_cal]
    no_grasp = [e for e in eps if e[2] is None]
    if no_grasp:
        print(f"WARN: grasp 앵커 없는 성공 ep {[e[0] for e in no_grasp]} — P1/P2 에서 제외")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    out = Path(args.out_dir)
    specs_dir = out / "specs"
    rows = []

    def add(mode: str, config: str, ep_idx: int, inf: int, spec: dict, sham: bool = False):
        tag = f"{config}{'_sham' if sham else ''}_ep{ep_idx}"
        spec = {**spec, "spec_seed": spec_seed(mode, config + ("_sham" if sham else ""), ep_idx),
                "sham": sham, "tag": tag}
        sp = specs_dir / config / f"ep{ep_idx}{'_sham' if sham else ''}.json"
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(spec, indent=1))
        rows.append((mode, config, ep_idx, inf, str(sp), tag))

    def pick(config_i: int):
        """config 별 ep 배분 — 라운드로빈 오프셋으로 config 간 ep 다양화."""
        n = len(eps)
        return [eps[(config_i + j) % n] for j in range(args.eps_per_config)]

    ci = 0
    for mode in modes:
        if mode == "C1":
            for s in C1_SCALES:
                config = f"c1_s{int(s * 100):03d}"
                for ep_idx, inf, _ in pick(ci):
                    add("C1_camera", config, ep_idx, inf, {"mode": "C1_camera", "scale": s})
                ci += 1
        elif mode == "G1":
            for s in G1_SIGMAS:
                config = f"g1_x{int(s * 100):03d}"
                for ep_idx, inf, _ in pick(ci):
                    add("G1_gripper_init", config, ep_idx, inf,
                        {"mode": "G1_gripper_init", "sigma_xyz_m": s})
                ci += 1
        elif mode == "P1":
            for mag in P1_MAGS:
                config = f"p1_d{int(mag * 100):03d}"
                for ep_idx, inf, grasp in pick(ci):
                    if grasp is None:
                        continue
                    add("P1_displace", config, ep_idx, inf,
                        {"mode": "P1_displace", "magnitude": mag,
                         "trigger_record": max(0, int(grasp) - 2)})
                ci += 1
        elif mode == "P2":
            for mag, dur in P2_GRID:
                config = f"p2_f{int(mag):03d}d{dur}"
                for ep_idx, inf, grasp in pick(ci):
                    if grasp is None:
                        continue
                    add("P2_force", config, ep_idx, inf,
                        {"mode": "P2_force", "magnitude": mag, "duration_records": dur,
                         "trigger_record": int(grasp) + 1})
                ci += 1
        else:
            print(f"ABORT: unknown mode {mode}", file=sys.stderr)
            return 2

    # sham 감시 행 (모드 대표 config 1개 × sham-eps)
    sham_reps = {"C1": ("C1_camera", "c1_s100", {"mode": "C1_camera", "scale": 1.0}),
                 "G1": ("G1_gripper_init", "g1_x010", {"mode": "G1_gripper_init", "sigma_xyz_m": 0.10}),
                 "P1": ("P1_displace", "p1_d008", {"mode": "P1_displace", "magnitude": 0.08}),
                 "P2": ("P2_force", "p2_f015d2", {"mode": "P2_force", "magnitude": 15.0,
                                                  "duration_records": 2})}
    for m in modes:
        mode_name, config, spec = sham_reps[m]
        for ep_idx, inf, grasp in eps[: args.sham_eps]:
            if m in ("P1", "P2"):
                if grasp is None:
                    continue
                spec = {**spec, "trigger_record": max(0, int(grasp) - 2) if m == "P1"
                        else int(grasp) + 1}
            add(mode_name, config, ep_idx, inf, spec, sham=True)

    out.mkdir(parents=True, exist_ok=True)
    grid = out / "grid.tsv"
    with grid.open("w") as f:
        f.write("mode\tconfig\tep_idx\tinference_seed\tspec\ttag\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    n_cfg = len({r[1] for r in rows})
    print(f"wrote {grid}: {len(rows)} rows, {n_cfg} configs, cal_eps={[e[0] for e in eps]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
