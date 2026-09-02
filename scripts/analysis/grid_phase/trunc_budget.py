#!/usr/bin/env python
"""길이 절제 모드별 **학습 데이터 포인트(record) 수** 집계 (docs/steering/43 후속 A).

질문: 절제(none / rollout / phase-gt) 사이의 detector 성능 차이가 **학습 데이터량**
차이로 설명되는가? 절제는 학습·보정 시퀀스를 잘라내므로 record 수가 줄어든다.
줄어든 양이 크면 "절제가 confound 를 제거했다"와 "표본이 줄어 학습이 약해졌다"가
섞인다 — 그 크기를 먼저 재는 것이 이 스크립트다.

## 규약 (재계산 금지)

- **scene split 은 sim_detail.json 의 `scene_split` 을 그대로 쓴다** (seed0 재현).
- **cap 도 sim_detail.json 의 `config.truncate_caps` 를 그대로 쓴다** — rollout_W /
  phase_caps 를 여기서 다시 추정하지 않는다 (구현 drift 방지).
- 절제 적용은 `failure_detector_sim.truncate_episode` 를 **import 해서** 쓴다.
  MIN_TRUNC_LEN 미만이 되어 버려지는 판까지 그쪽 규약과 정확히 일치시킨다.
- 집계 대상은 **train split** (학습 데이터량 질문). calib 은 CP 밴드용이라 참고로만
  같이 낸다.

## 사용
    ~/anaconda3/bin/python scripts/analysis/grid_phase/trunc_budget.py \
        --shard-dir ~/datasets/.../analysis/grid_phase/segA \
        --sim-root outputs/analysis/grid_phase/detector_trunc \
        --out outputs/analysis/grid_phase/detector_trunc/trunc_budget.tsv
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from failure_detector_sim import (  # noqa: E402
    load_shard_episodes, truncate_episode,
)

MODES = ("none", "rollout", "phase-gt")


def load_sim(sim_root: Path, mode: str) -> dict:
    p = sim_root / mode / "sim_detail.json"
    if not p.exists():
        raise SystemExit(f"sim_detail.json 없음: {p}")
    d = json.loads(p.read_text(encoding="utf-8"))
    if d["config"]["truncate_train"] != mode:
        raise SystemExit(
            f"{p}: truncate_train={d['config']['truncate_train']} != 디렉터리 {mode}")
    return d


def budget_for(eps, mode: str, W, caps) -> tuple[int, int, int]:
    """(record 수, 남은 판 수, 버려진 판 수) — 절제 후."""
    n_rec = n_ep = n_drop = 0
    for e in eps:
        te = truncate_episode(e, mode, W, caps)
        if te is None:
            n_drop += 1
            continue
        n_rec += te.T
        n_ep += 1
    return n_rec, n_ep, n_drop


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--sim-root", required=True,
                    help="detector_trunc 디렉터리 (하위에 none/rollout/phase-gt)")
    ap.add_argument("--out", required=True, help="출력 TSV 경로")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--denoise", type=int, default=-1)
    ap.add_argument("--seg", default="all")
    ap.add_argument("--part", default="train", choices=["train", "calib"])
    args = ap.parse_args()

    shard_dir = Path(args.shard_dir)
    sim_root = Path(args.sim_root)
    sims = {m: load_sim(sim_root, m) for m in MODES}

    # scene_split / caps 는 mode 간 동일해야 한다 (같은 seed) — 어긋나면 fail-loud.
    base = sims["none"]
    for m in MODES[1:]:
        if sims[m]["scene_split"] != base["scene_split"]:
            raise SystemExit(f"scene_split 이 mode 간 다르다 (none vs {m})")
        if sims[m]["config"]["truncate_caps"] != base["config"]["truncate_caps"]:
            raise SystemExit(f"truncate_caps 가 mode 간 다르다 (none vs {m})")

    tasks = sorted(base["scene_split"])
    rows: list[dict] = []
    for slug in tasks:
        p = shard_dir / f"{slug}.npz"
        if not p.exists():
            raise SystemExit(f"shard 없음: {p}")
        eps, _spec = load_shard_episodes(p, args.layer, args.denoise, args.seg)
        scenes = set(int(s) for s in base["scene_split"][slug][args.part])
        part_eps = [e for e in eps if e.scene in scenes]
        capcfg = base["config"]["truncate_caps"][slug]
        W = capcfg["rollout_W"]
        caps = {int(k): int(v) for k, v in (capcfg["phase_caps"] or {}).items()}
        row = {"task": slug, "part": args.part, "n_ep": len(part_eps),
               "n_ep_fail": sum(1 for e in part_eps if e.y == 1),
               "W": W, "phase_caps": ",".join(f"{k}:{v}" for k, v in sorted(caps.items()))}
        for m in MODES:
            rec, nep, ndrop = budget_for(part_eps, m, W, caps)
            row[f"rec_{m}"] = rec
            row[f"ep_{m}"] = nep
            row[f"drop_{m}"] = ndrop
        base_rec = row["rec_none"]
        for m in MODES:
            row[f"pct_{m}"] = round(100.0 * row[f"rec_{m}"] / base_rec, 1) if base_rec else None
        rows.append(row)
        print(f"[{slug}] {args.part} ep={row['n_ep']} "
              f"none={row['rec_none']} rollout={row['rec_rollout']}"
              f"({row['pct_rollout']}%) phase-gt={row['rec_phase-gt']}"
              f"({row['pct_phase-gt']}%)", flush=True)

    tot = {"task": "__total__", "part": args.part,
           "n_ep": sum(r["n_ep"] for r in rows),
           "n_ep_fail": sum(r["n_ep_fail"] for r in rows), "W": "", "phase_caps": ""}
    for m in MODES:
        tot[f"rec_{m}"] = sum(r[f"rec_{m}"] for r in rows)
        tot[f"ep_{m}"] = sum(r[f"ep_{m}"] for r in rows)
        tot[f"drop_{m}"] = sum(r[f"drop_{m}"] for r in rows)
    for m in MODES:
        tot[f"pct_{m}"] = round(100.0 * tot[f"rec_{m}"] / tot["rec_none"], 1)
    rows.append(tot)

    cols = (["task", "part", "n_ep", "n_ep_fail", "W", "phase_caps"]
            + [f"rec_{m}" for m in MODES] + [f"pct_{m}" for m in MODES]
            + [f"ep_{m}" for m in MODES] + [f"drop_{m}" for m in MODES])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(cols)]
    for r in rows:
        lines.append("\t".join(str(r.get(c, "")) for c in cols))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n[budget] part={args.part} — 학습 record 수 (절제 모드별)")
    hdr = f"{'task':22s} {'ep':>4s} {'none':>7s} {'rollout':>9s} {'phase-gt':>10s}"
    print(hdr)
    for r in rows:
        print(f"{r['task'][:22]:22s} {r['n_ep']:4d} {r['rec_none']:7d} "
              f"{r['rec_rollout']:6d}({r['pct_rollout']:4.0f}%) "
              f"{r['rec_phase-gt']:7d}({r['pct_phase-gt']:4.0f}%)")
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
