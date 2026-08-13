#!/usr/bin/env python3
"""online-gated eval 사이드카 → per_episode.tsv (+ 요약).

캡처-OFF eval 의 판정 원천은 `raw_rollouts/<TASK>/<cell>/task*--ep*--succ{0,1}.json`
사이드카다 (`http_feature_collect.py --no-features`). 여기서 판정(success)과 개입 감사
필드(trigger_step / phase_at_trigger / phase_gated_flags / serve_steering)를 뽑아
한 arm 의 표를 만든다 — 로그 파싱 금지, 판정은 항상 산출물에서.

열:
  ep  scene_idx  env_seed  inference_seed  success  steps  n_inferences
  trigger_step  phase_at_trigger  n_gated  gated_mode  arm  slug  beta  op  serve_gpu  run_tag

사용:
    python scripts/steer/online_gated/collect_results.py \
        --job-dir outputs/eval/robocasa/groot_n15/online_gated/OvenRack_out/online \
        --scenes-tsv <logs>/scenes_OvenRack_out.tsv --arm online --slug OvenRack_out \
        --expect 20
`--expect` 를 주면 행 수가 모자랄 때 exit 13 (러너의 미완 판정).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

STEM_RE = re.compile(r"task(?P<tid>\d+)--ep(?P<ep>\d+)--succ(?P<succ>[01])$")
COLUMNS = ("ep", "scene_idx", "env_seed", "inference_seed", "success", "steps",
           "n_inferences", "trigger_step", "phase_at_trigger", "n_gated", "gated_mode",
           "arm", "slug", "beta", "op", "serve_gpu", "run_tag")


def cell(v) -> str:
    if v is None:
        return "NA"
    if isinstance(v, bool):
        return str(int(v))
    return str(v).replace("\t", " ")


def load_scenes(path: Path | None) -> dict[int, int]:
    """env_seed → scene_idx."""
    if path is None:
        return {}
    out = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        p = ln.split("\t")
        out[int(p[1])] = int(p[0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", type=Path, required=True, help="<out>/<slug>/<arm>")
    ap.add_argument("--scenes-tsv", type=Path, default=None)
    ap.add_argument("--arm", default="NA")
    ap.add_argument("--slug", default="NA")
    ap.add_argument("--expect", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="기본 <job-dir>/per_episode.tsv")
    args = ap.parse_args()

    seed2scene = load_scenes(args.scenes_tsv)
    rows = []
    for jp in sorted((args.job_dir / "raw_rollouts").rglob("task*--ep*--succ*.json")):
        m = STEM_RE.match(jp.stem)
        if m is None:
            continue
        sc = json.loads(jp.read_text(encoding="utf-8"))
        steer = sc.get("serve_steering") or {}
        flags = sc.get("phase_gated_flags") or []
        env_seed = sc.get("scenario_seed", sc.get("seed"))
        rows.append({
            "ep": int(m.group("ep")),
            "scene_idx": seed2scene.get(int(env_seed)) if env_seed is not None else None,
            "env_seed": env_seed,
            "inference_seed": sc.get("inference_seed"),
            # 스템의 succ 와 본문 판정이 어긋나면 사이드카가 손상된 것 → fail-loud
            "success": int(sc.get("episode_success", m.group("succ"))),
            "steps": sc.get("steps"),
            "n_inferences": sc.get("n_inferences"),
            "trigger_step": sc.get("trigger_step"),
            "phase_at_trigger": sc.get("phase_at_trigger"),
            "n_gated": sum(1 for f in flags if f),
            "gated_mode": sc.get("gated_steering_mode", "off"),
            "arm": args.arm,
            "slug": args.slug,
            "beta": steer.get("beta"),
            "op": steer.get("op"),
            "serve_gpu": sc.get("serve_gpu"),
            "run_tag": sc.get("run_tag"),
        })
        if int(sc.get("episode_success", -1)) not in (-1, int(m.group("succ"))):
            raise SystemExit(f"{jp}: 스템 succ 와 episode_success 불일치")
    rows.sort(key=lambda r: r["ep"])

    out_path = args.out or (args.job_dir / "per_episode.tsv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = ["\t".join(COLUMNS)]
    body += ["\t".join(cell(r[c]) for c in COLUMNS) for r in rows]
    out_path.write_text("\n".join(body) + "\n", encoding="utf-8")

    n = len(rows)
    s = sum(r["success"] for r in rows)
    fired = [r for r in rows if r["trigger_step"] not in (None, "NA")]
    sr = f"{s / n:.3f}" if n else "NA"
    print(f"[results] {args.slug}/{args.arm}: n={n} succ={s} SR={sr} "
          f"fired={len(fired)}/{n} → {out_path}", flush=True)
    if args.expect and n < args.expect:
        print(f"[results] INCOMPLETE — {n} < expect {args.expect}", file=sys.stderr)
        return 13
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
