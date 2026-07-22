"""INSIGHT VLM primitive-segmentation pilot runner.

Glues the three pieces of this package together:

    rollout_adapter.load_episode   (our pkl+mp4 -> EpisodeData)
        -> segmentation.segment_episode  (INSIGHT-ported VLM labeling)
            -> segments.json + annotated.mp4 per rollout
                -> aggregate metrics (summary.json / summary.tsv)

The VLM backend is chosen by ``vlm_client.get_vlm_client``: a real Gemini client
when ``GEMINI_API_KEY`` (or ``~/.config/temporal_vla/gemini_api_key``) is present,
otherwise a deterministic MockVLMClient (``--mock`` forces mock). Mock mode
exercises the whole pipeline (pkl load, mp4 decode, panel split, keyframe encode,
prompt build, response parse, frame->step map, annotated video) WITHOUT an API
key — its primitive labels are placeholders, not real segmentations.

Run (from repo root) with the env that has numpy/torch/cv2/google-genai:
    ~/miniconda3/envs/lerobot_safe/bin/python \
        scripts/analysis/insight_seg/run_pilot.py \
        --root outputs/eval/robocasa/groot_n15/coast4_instruction_pathway_50ep/raw_rollouts \
        --task coffee_setup_mug:2:2 --task close_fridge:2:1 \
        --out outputs/analysis/insight_seg/pilot01

PROVENANCE: orchestration only; segmentation logic is vendored from
github.com/insight-vla/insight (Apache-2.0) in segmentation.py / prompts.py.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import traceback
from pathlib import Path

import numpy as np

# Make sibling modules importable whether run as a script or a module.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import rollout_adapter as ra          # noqa: E402
import lerobot_adapter as la          # noqa: E402
import segmentation as seg            # noqa: E402
import vlm_client as vc               # noqa: E402


def _parse_task_spec(spec: str) -> tuple[str, int, int]:
    """Parse ``substr:nsucc:nfail`` (nsucc/nfail optional, default 2/1)."""
    parts = spec.split(":")
    substr = parts[0]
    nsucc = int(parts[1]) if len(parts) > 1 and parts[1] else 2
    nfail = int(parts[2]) if len(parts) > 2 and parts[2] else 1
    return substr, nsucc, nfail


def _parse_episodes(spec: str, all_eps: list[int]) -> list[int]:
    """Parse ``auto:N`` (first N) or a comma list ``0,5,10``."""
    spec = spec.strip()
    if spec.startswith("auto:"):
        n = int(spec.split(":", 1)[1])
        return all_eps[:n]
    return [int(x) for x in spec.split(",") if x.strip() != "" and int(x) in all_eps]


def _segment_lengths(segments: list[dict]) -> list[int]:
    return [int(s["end_step"]) - int(s["start_step"]) for s in segments]


def _coverage_ok(segments: list[dict], n_steps: int) -> bool:
    """Contiguous, gap-free, full coverage [0, n_steps]."""
    if not segments:
        return False
    if int(segments[0]["start_step"]) != 0:
        return False
    if int(segments[-1]["end_step"]) != n_steps:
        return False
    for a, b in zip(segments, segments[1:]):
        if int(a["end_step"]) != int(b["start_step"]):
            return False
    return True


def run_one(ep: dict, ep_id: str, vlm, cfg_kwargs: dict, out_dir: Path) -> dict:
    """Segment one (already-loaded) episode; write artifacts; return a metrics row."""
    n_steps = int(ep["n_steps"])

    cfg = seg.SegConfig(has_gripper=bool(ep["has_gripper"]), **cfg_kwargs)
    segments = seg.segment_episode(ep, vlm, cfg)

    task_name = ep["meta"].get("task_name", "task")
    ep_out = out_dir / task_name / ep_id
    ep_out.mkdir(parents=True, exist_ok=True)

    record = {
        "id": ep_id,
        "source": ep["meta"].get("source", "rollout"),
        "task_name": task_name,
        "task": ep["task"],
        "success": int(ep["success"]),
        "n_steps": n_steps,
        "has_gripper": bool(ep["has_gripper"]),
        "segment_method": cfg.segment_method,
        "n_segments": len(segments),
        "segment_lengths": _segment_lengths(segments),
        "coverage_ok": _coverage_ok(segments, n_steps),
        "segments": segments,
    }
    (ep_out / "segments.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    annotated = ""
    try:
        view = cfg.primary_view if cfg.primary_view in ep["frames"] else None
        if ep["frames"]:
            annotated = seg.write_annotated_video(
                ep, segments, str(ep_out / "annotated.mp4"), view=view
            )
    except Exception as e:  # video is a nice-to-have, never fatal
        record["annotated_error"] = f"{type(e).__name__}: {e}"
    record["annotated_mp4"] = annotated

    return record


def main() -> int:
    ap = argparse.ArgumentParser(description="INSIGHT segmentation pilot")
    # Source A: our policy rollouts (pkl+mp4). Source B: LeRobot expert demos.
    ap.add_argument("--root", default=None, help="rollout root dir to glob (source=rollout)")
    ap.add_argument("--task", action="append", default=[],
                    help="rollout task spec 'substr:nsucc:nfail' (repeatable). "
                         "If omitted, takes --limit rollouts from --root.")
    ap.add_argument("--lerobot-root", default=None,
                    help="LeRobot expert-demo source: either a single '.../lerobot' dir "
                         "or a robocasa pretrain/atomic base (multiple tasks).")
    ap.add_argument("--lerobot-task", default=None,
                    help="substring filter on task name when --lerobot-root is a base dir")
    ap.add_argument("--episodes", default="auto:3",
                    help="LeRobot episodes: 'auto:N' (first N per task) or '0,5,10'")
    ap.add_argument("--out", required=True, help="output dir for artifacts")
    ap.add_argument("--mock", dest="mock", action="store_true", default=None,
                    help="force MockVLMClient (no API calls)")
    ap.add_argument("--no-mock", dest="mock", action="store_false",
                    help="require a real VLM (error if no key)")
    ap.add_argument("--model", default=None, help="VLM model override")
    ap.add_argument("--segment-method", default="video",
                    choices=["video", "gripper", "action_change"])
    ap.add_argument("--num-keyframes", type=int, default=24)
    ap.add_argument("--no-refine", action="store_true",
                    help="skip state-changepoint boundary refinement")
    ap.add_argument("--limit", type=int, default=8,
                    help="max rollouts when no --task specs given")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build jobs: list of (loader_callable -> ep, ep_id). Loaded lazily in the
    # loop so we don't hold every episode's frames in memory at once.
    jobs: list[tuple] = []
    if args.lerobot_root:
        lr = Path(args.lerobot_root)
        if (lr / "meta" / "info.json").is_file():
            task_roots = [lr]
        else:
            task_roots = la.find_task_roots(lr, args.lerobot_task)
        for tr in task_roots:
            all_eps = la.list_lerobot_episodes(tr)
            sel = _parse_episodes(args.episodes, all_eps)
            for i in sel:
                jobs.append(((lambda tr=tr, i=i: la.load_lerobot_episode(tr, i)),
                             f"ep{i:06d}"))
        source = "lerobot"
    else:
        if not args.root:
            print("[run_pilot] need --root or --lerobot-root", file=sys.stderr)
            return 2
        root = Path(args.root)
        if args.task:
            specs = [_parse_task_spec(s) for s in args.task]
            pkls = ra.select_pilot_set(str(root), specs)
        else:
            pkls = ra.list_rollouts(str(root))[: args.limit]
        jobs = [((lambda p=p: ra.load_episode(p)), Path(p).stem) for p in pkls]
        source = "rollout"

    if not jobs:
        print("[run_pilot] No episodes matched.", file=sys.stderr)
        return 2

    vlm = vc.get_vlm_client(use_mock=args.mock, model=args.model)
    is_mock = isinstance(vlm, vc.MockVLMClient)

    cfg_kwargs = dict(
        segment_method=args.segment_method,
        model=args.model,
        num_keyframes=args.num_keyframes,
        refine_boundaries=not args.no_refine,
    )

    print(f"[run_pilot] {len(jobs)} episodes | source={source} | mock={is_mock} | "
          f"method={args.segment_method} | out={out_dir}")

    rows: list[dict] = []
    for i, (loader, ep_id) in enumerate(jobs):
        try:
            ep = loader()
            row = run_one(ep, ep_id, vlm, cfg_kwargs, out_dir)
            rows.append(row)
            print(f"  [{i+1}/{len(jobs)}] {row['task_name']}/{ep_id}: "
                  f"{row['n_segments']} segs, cover_ok={row['coverage_ok']}, "
                  f"succ={row['success']}, has_grip={row['has_gripper']}")
        except Exception as e:
            print(f"  [{i+1}/{len(jobs)}] FAILED {ep_id}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            traceback.print_exc()
            rows.append({"id": ep_id, "error": f"{type(e).__name__}: {e}"})

    # Aggregate metrics.
    ok = [r for r in rows if "error" not in r]
    all_lens = [L for r in ok for L in r.get("segment_lengths", [])]
    summary = {
        "root": str(args.lerobot_root or args.root),
        "source": source,
        "mock": is_mock,
        "model": args.model or vc.DEFAULT_MODEL,
        "segment_method": args.segment_method,
        "n_rollouts": len(rows),
        "n_ok": len(ok),
        "n_failed": len(rows) - len(ok),
        "n_coverage_ok": sum(1 for r in ok if r.get("coverage_ok")),
        "n_segments_mean": (float(np.mean([r["n_segments"] for r in ok]))
                            if ok else 0.0),
        "seg_len_mean": float(np.mean(all_lens)) if all_lens else 0.0,
        "seg_len_min": int(np.min(all_lens)) if all_lens else 0,
        "seg_len_max": int(np.max(all_lens)) if all_lens else 0,
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Flat TSV for quick scanning.
    tsv_lines = ["task\tep\tsuccess\thas_gripper\tn_steps\tn_segments\tcoverage_ok"]
    for r in ok:
        tsv_lines.append(
            f"{r['task_name']}\t{r['id']}\t{r['success']}\t"
            f"{r['has_gripper']}\t{r['n_steps']}\t{r['n_segments']}\t{r['coverage_ok']}"
        )
    (out_dir / "summary.tsv").write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")

    print(f"[run_pilot] done. ok={len(ok)}/{len(rows)}, "
          f"coverage_ok={summary['n_coverage_ok']}/{len(ok)}, "
          f"mean_segs={summary['n_segments_mean']:.1f}. "
          f"Wrote {out_dir}/summary.json + summary.tsv")
    if is_mock:
        print("[run_pilot] NOTE: mock VLM — labels are placeholders. Provide a "
              "GEMINI_API_KEY for real segmentation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
