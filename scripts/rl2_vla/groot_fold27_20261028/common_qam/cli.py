"""atomic65 common QAM CLI (CPU side). GPU steps are printed as commands for the main session.

  status            progress of every stage (stager -> staging -> manifest -> extraction -> train)
  stage             copy stager-COMPLETE packages 승준 -> compute staging (hash-gated, ≤2 jobs)
  manifest          build manifest.json (all 65) or --smoke-tasks A B -> manifest_smoke.json
  finalize-extract  verify per-episode records -> COMPLETE.json
  preflight         CPU checks before a GPU step (--step extract-smoke|extract|train-smoke|train)
  commands          print the ordered GPU/CPU command list
  cleanup-plan      print (never run) the staging deletion command if it is safe
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from experiment import (ADAPTER, N_TASKS, REPO, STAGER_RECEIPTS, STAGING_ROOT, WORK,  # noqa: E402
                        load_stage_rows, load_stager_receipt, sha256_path)

GIB = 1024 ** 3
PY = REPO / ".venvs/rl2_iid/bin/python"
REL_HERE = HERE.relative_to(REPO)


def status_doc(staging: Path, work: Path) -> dict:
    rows = load_stage_rows()
    stager = [r["task"] for r in rows if load_stager_receipt(r["task"])]
    staged = [r["task"] for r in rows if (staging / "receipts" / f"{r['task']}.json").exists()]
    doc = {"stager_complete": len(stager), "staged": len(staged), "tasks": N_TASKS,
           "staged_waiting": sorted(set(stager) - set(staged)),
           "manifest": (work / "metadata/manifest.json").exists(),
           "manifest_smoke": (work / "metadata/manifest_smoke.json").exists()}
    for name in ("smoke_latents", "latents"):
        d = work / name
        if d.is_dir():
            doc[name] = {"records": len([p for p in d.glob("*.json") if len(p.stem) == 64]),
                         "complete": (d / "COMPLETE.json").exists()}
    for name in ("scratch_smoke", "scratch_run"):
        d = work / name
        if d.is_dir():
            prog = d / "progress.jsonl"
            last = prog.read_text().splitlines()[-1] if prog.exists() and prog.stat().st_size else None
            doc[name] = {"complete": (d / "COMPLETE.json").exists(),
                         "last_step": json.loads(last)["step"] if last else None}
    if staging.exists():
        doc["staging_free_gib"] = round(shutil.disk_usage(staging).free / GIB, 1)
    return doc


def cmd_status(a) -> int:
    doc = status_doc(a.staging, a.work)
    a.work.mkdir(parents=True, exist_ok=True)
    (a.work / "STATUS.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc, indent=1))
    return 0


def cmd_stage(a) -> int:
    from staging import stage
    s = stage(a.staging, tasks=a.tasks, jobs=a.jobs, reserve=int(a.reserve_gib * GIB), transport=a.transport,
              reverify=a.reverify)
    print(json.dumps({k: v for k, v in s.items() if k != "staged_or_present"} | {"staged_or_present": len(s["staged_or_present"])}))
    return 1 if s["failed"] else 0


def cmd_manifest(a) -> int:
    from manifest import build, write
    m, errors = build(a.staging, tasks_subset=a.smoke_tasks)
    return write(m, errors, a.work / "metadata", "manifest_smoke.json" if a.smoke_tasks else "manifest.json")


def cmd_finalize(a) -> int:
    from extract import finalize
    name = "manifest_smoke.json" if a.smoke else "manifest.json"
    return finalize(a.work / "metadata" / name, a.work / ("smoke_latents" if a.smoke else "latents"), a.smoke)


def preflight(step: str, work: Path, staging: Path) -> list[str]:
    errs = []
    smoke = step.endswith("smoke")
    man = work / "metadata" / ("manifest_smoke.json" if smoke else "manifest.json")
    if not man.exists():
        errs.append(f"missing {man}")
        return errs
    m = json.loads(man.read_text())
    if not smoke and (m.get("mode") != "full" or m.get("n_tasks") != N_TASKS):
        errs.append("manifest is not the full atomic65 manifest")
    for t in m["tasks"]:   # staged packages still present and receipts consistent with the stager
        rp = staging / "receipts" / f"{t['task']}.json"
        if not rp.exists() or json.loads(rp.read_text())["package_manifest_sha256"] != t["package_manifest_sha256"]:
            errs.append(f"{t['task']}: staging receipt missing/changed")
    try:
        sys.path.insert(0, str(ADAPTER))
        from contract import resolve_cached_checkpoint_dir
        resolve_cached_checkpoint_dir()   # pinned config/metadata sha
    except Exception as e:
        errs.append(f"checkpoint contract: {e!r}")
    if step.startswith("extract"):
        if shutil.disk_usage(work).free < 20 * GIB:
            errs.append("work filesystem < 20 GiB free")
        img = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}",
                              "sha256:55008acf7d21c1ec2aefc72593b2248caec8094c6aa1829cf907ba98d9e373c9"],
                             capture_output=True, text=True)
        if img.returncode != 0:
            errs.append("model image not present (docker image inspect failed)")
    if step.startswith("train"):
        lat = work / ("smoke_latents" if smoke else "latents") / "COMPLETE.json"
        if not lat.exists():
            errs.append(f"missing {lat} (run finalize-extract)")
        elif json.loads(lat.read_text())["contract"]["manifest_sha256"] != sha256_path(man):
            errs.append("extraction was made from a different manifest")
        out = work / ("scratch_smoke" if smoke else "scratch_run")
        if out.exists():
            errs.append(f"{out} exists (train never overwrites; choose a new --out)")
        if shutil.disk_usage(work).free < 10 * GIB:
            errs.append("work filesystem < 10 GiB free")
    return errs


def cmd_preflight(a) -> int:
    errs = preflight(a.step, a.work, a.staging)
    for e in errs:
        print("BLOCKER", e)
    print("PREFLIGHT OK" if not errs else "PREFLIGHT BLOCKED", a.step)
    return 1 if errs else 0


def commands(work: Path) -> str:
    c = f"{PY} {REL_HERE}/cli.py"
    return "\n".join([
        "# CPU (repeat stage while the stager downloads; finished tasks are skipped)",
        f"{c} stage --jobs 2",
        f"{c} manifest --smoke-tasks <TaskA> <TaskB>      # two staged tasks",
        f"{c} preflight --step extract-smoke",
        "# GPU (main session; one GPU per shard, lease via harness guard)",
        f"GPUS=<g> HARNESS_SESSION=<session> bash {REL_HERE}/run_extract.sh --smoke",
        f"{c} finalize-extract --smoke",
        f"{c} preflight --step train-smoke",
        f"JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=<g> bash scripts/utils/with_gpu_lease.sh kanu <g> <session> atomic65-qam-smoke -- "
        f"{PY} -u {REL_HERE}/train.py --manifest {work}/metadata/manifest_smoke.json --extraction {work}/smoke_latents --out {work}/scratch_smoke --smoke",
        "# after all 65 staged",
        f"{c} manifest",
        f"{c} preflight --step extract",
        f"GPUS=<g0> HARNESS_SESSION=<session> bash {REL_HERE}/run_extract.sh --shard 0 --num-shards 2   # and --shard 1 on <g1>",
        f"{c} finalize-extract",
        f"{c} preflight --step train",
        f"JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=<g> setsid nohup bash scripts/utils/with_gpu_lease.sh kanu <g> <session> atomic65-qam-train -- "
        f"{PY} -u {REL_HERE}/train.py --manifest {work}/metadata/manifest.json --extraction {work}/latents --out {work}/scratch_run > {work}/train_launch.log 2>&1 < /dev/null &",
        f"{c} cleanup-plan    # prints a deletion command only if every condition holds",
    ])


def cmd_commands(a) -> int:
    print(commands(a.work))
    return 0


def cmd_cleanup_plan(a) -> int:
    """Staging may go only when: training COMPLETE + archived, every staged package still hashes
    to the stager receipt (originals verified on 승준), and no other consumer is registered."""
    from staging import manifest_sha
    reasons = []
    done = a.work / "scratch_run" / "COMPLETE.json"
    if not done.exists():
        reasons.append("training not COMPLETE")
    else:
        sel = json.loads(done.read_text())["selection"]
        if not sel.get("archive", {}).get("verified"):
            reasons.append("final checkpoint archive not verified")
    missing = []
    for r in load_stage_rows():
        srec = load_stager_receipt(r["task"])
        rp = a.staging / "receipts" / f"{r['task']}.json"
        if srec is None:
            missing.append(r["task"]); continue
        if rp.exists() and a.rehash and manifest_sha(a.staging / "robocasa" / r["rel_path"]) != srec["package_manifest_sha256"]:
            reasons.append(f"{r['task']}: staged copy differs from original receipt")
    if missing:
        reasons.append(f"{len(missing)} tasks without a verified 승준 original receipt (e.g. {missing[:3]})")
    cons = a.staging / "CONSUMERS.json"
    others = [c for c in (json.loads(cons.read_text()) if cons.exists() else []) if c.get("experiment") != "atomic65_common_qam_20261029"]
    if others:
        reasons.append(f"other consumers registered: {others}")
    if reasons:
        print("CLEANUP NOT SAFE:\n  " + "\n  ".join(reasons)); return 1
    print("# safe; review then run manually (main session):")
    print(f"rm -rf -- {a.staging}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--staging", type=Path, default=STAGING_ROOT)
    ap.add_argument("--work", type=Path, default=WORK)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    p = sub.add_parser("stage"); p.set_defaults(fn=cmd_stage)
    p.add_argument("--tasks", nargs="*"); p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--reserve-gib", type=float, default=20.0); p.add_argument("--reverify", action="store_true")
    p.add_argument("--transport", choices=["remote", "local"], default="remote")
    p = sub.add_parser("manifest"); p.set_defaults(fn=cmd_manifest); p.add_argument("--smoke-tasks", nargs=2)
    p = sub.add_parser("finalize-extract"); p.set_defaults(fn=cmd_finalize); p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("preflight"); p.set_defaults(fn=cmd_preflight)
    p.add_argument("--step", required=True, choices=["extract-smoke", "extract", "train-smoke", "train"])
    sub.add_parser("commands").set_defaults(fn=cmd_commands)
    p = sub.add_parser("cleanup-plan"); p.set_defaults(fn=cmd_cleanup_plan); p.add_argument("--rehash", action="store_true")
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
