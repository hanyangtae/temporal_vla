#!/usr/bin/env python3
"""Deploy isolated experiment code through git; move manifests/artifacts only.

GPU evaluation remains on the collection machine. Archive fitting is serial CPU
work and never claims a GPU. All remote operations use remote_compute.sh.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

CODE = Path(__file__).resolve().parents[4]
HELPER = CODE / "scripts/utils/remote_compute.sh"
MODULE = "scripts/analysis/grid_phase/lr_equal_budget"
INPUTS = ["manifest.tsv", "diagnostics.json", "mapping.json", "feasibility.json",
          "eval_cases.tsv", "preservation_eval.tsv", "sources/index_v6_complete_cells.tsv",
          "sources/collection_plan.json", "sources/scene_selection.json"]


def run(argv, *, env=None, capture=False, timeout=None):
    print("[exec]", shlex.join([str(x) for x in argv]), flush=True)
    return subprocess.run([str(x) for x in argv], env=env, check=True,
                          text=True, capture_output=capture, timeout=timeout)


def remote_env(machine, repo=None):
    env = dict(os.environ)
    if machine == "archive":
        env.update(REMOTE_USER="kimseungjun", REMOTE_HOST="166.104.146.37",
                   REMOTE_PORT="11112", REMOTE_REPO="~/workspace/temporal_vla",
                   REMOTE_PYTHON="~/anaconda3/bin/python", REMOTE_THREADS="8")
    elif machine == "srv48":
        env.update(REMOTE_USER="junhyeong", REMOTE_HOST="AISem_48_junhyeong",
                   REMOTE_PORT="22", REMOTE_REPO="~/pkt_ws/temporal_vla",
                   REMOTE_PYTHON="~/miniconda3/envs/lerobot_050_groot/bin/python")
    else:
        raise ValueError(machine)
    if repo:
        env["REMOTE_REPO"] = repo
    return env


def helper(machine, verb, *args, repo=None, capture=False, timeout=30):
    argv = ["bash", HELPER, verb, *args]
    if timeout is not None:
        # Kill the helper's whole SSH group on readiness timeout, not only bash.
        argv = ["timeout", "--kill-after=2s", f"{timeout}s", *argv]
    return run(argv, env=remote_env(machine, repo), capture=capture,
               timeout=None if timeout is None else timeout + 5)


def remote_code(machine, revision, branch):
    # Resolve only a hex commit, never a moving branch for execution.
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("full pinned git SHA required")
    base = remote_env(machine)["REMOTE_REPO"]
    dest = base + "/.claude/worktrees/lr-equal-budget-" + revision[:12]
    body = ("git fetch origin " + shlex.quote(branch) + " && "
            "if test ! -d " + dest + "; then git worktree add --detach " + dest + " " + revision + "; fi && "
            "test \"$(git -C " + dest + " rev-parse HEAD)\" = " + revision)
    helper(machine, "run", body, timeout=120)
    return dest


def data_files(root):
    # A local isolated checkout may symlink outputs to the shared main checkout.
    main = Path(str(CODE).split("/.claude/worktrees/")[0])
    try:
        rel = root.relative_to(CODE)
    except ValueError:
        rel = root.relative_to(main)
    if not str(rel).startswith("outputs/"):
        raise ValueError("experiment root must be under code checkout outputs/")
    return rel, [str(rel / p) for p in INPUTS + ["INPUT_SHA256.json"]]


def verify_inputs(root):
    expected = json.loads((root / "INPUT_SHA256.json").read_text())
    if set(expected) != set(INPUTS):
        raise ValueError("input snapshot inventory changed")
    for name, checksum in expected.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != checksum:
            raise ValueError("frozen input changed: " + name)


def discover_shards(manifest, shard_root):
    """Match small NPZ metadata by episode fingerprints, not mutable LR names."""
    import numpy as np
    with open(manifest) as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    wanted = {}
    for r in rows:
        k = r["canonical_instruction"].replace("/", "_") + "__s" + r["scene_idx"]
        wanted.setdefault(k, set()).add(r["sig"])
    candidates = []
    for folder in (shard_root / "segA_scene", shard_root / "segA"):
        if folder.exists():
            candidates.extend(sorted(folder.glob("*.npz")))
    mapping = {}
    for p in candidates:
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"]))
        sigs = set(meta.get("sigs", []))
        for key, required in wanted.items():
            if required <= sigs and key not in mapping:
                mapping[key] = {"path": str(p.resolve()), "source_instruction": meta["instruction"],
                                "selection": "all manifest episode fingerprints found in source metadata"}
    missing = sorted(set(wanted) - set(mapping))
    if missing:
        raise ValueError("unresolved shard identities: " + ", ".join(missing))
    return mapping


def fit_here(args):
    root = args.root.resolve()
    verify_inputs(root)
    mapping = discover_shards(root / "manifest.tsv", args.shard_root.expanduser())
    (root / "shard_map.json").write_text(json.dumps(mapping, indent=2) + "\n")
    run([sys.executable, CODE / MODULE / "fit.py", "--manifest", root / "manifest.tsv",
         "--plans", root / "diagnostics.json", "--shard-map", root / "shard_map.json",
         "--out", root / "fit", "--epochs", "25", "--hidden", "256", "--batch-size", "8",
         "--seed", "0", "--threads", "8"], timeout=None)
    plans = json.loads((root / "diagnostics.json").read_text())["plans"]
    expected = {p["pair_id"] for p in plans}
    actual = {p.parent.name for p in (root / "fit").glob("*/pair_report.json")}
    if actual != expected:
        raise ValueError(f"fit pair coverage mismatch: expected {len(expected)}, found {len(actual)}")
    (root / "FIT_DONE").write_text("FIT_DONE\n")


def archive_ready(args):
    # A short failed probe is a queue readiness failure, not a completed fit.
    body = ("test -x ~/anaconda3/bin/python && "
            "test -d ~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase_v6/segA_scene && "
            "test \"$(ps -u $(id -u) -o pcpu=,args= | "
            "awk '$1 >= 50 && /python/ {n++} END {print n+0}')\" -lt 2")
    helper("archive", "run", body, timeout=15)


def archive_fit(args):
    root = args.root.resolve()
    verify_inputs(root)
    rel, files = data_files(root)
    dest = remote_code("archive", args.revision, args.branch)
    helper("archive", "push-data", *files, repo=dest, timeout=120)
    archive_ready(args)
    cmd = ("~/anaconda3/bin/python " + MODULE + "/remote_stage.py fit-here --root " +
           shlex.quote(str(rel)) + " --shard-root "
           "~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase_v6")
    helper("archive", "run", cmd, repo=dest, timeout=None)
    # Bound artifact transfer and never fetch raw features.
    check = "du -sb " + shlex.quote(str(rel / "fit"))
    result = helper("archive", "run", check, repo=dest, capture=True)
    if int(result.stdout.split()[0]) > 2_000_000_000:
        raise ValueError("fit outputs unexpectedly exceed 2GB; inspect before retrieval")
    helper("archive", "pull-results", str(rel / "fit"), str(rel / "FIT_DONE"),
           str(rel / "shard_map.json"), repo=dest, timeout=600)


def eval_ready(args):
    if not (args.root / "FIT_DONE").is_file():
        raise ValueError("waiting for verified FIT_DONE")
    # GPU/lease ownership is rechecked and claimed by eval.py immediately before use.
    if args.machine == "srv48":
        helper("srv48", "run", "set -o pipefail; nvidia-smi -i " + str(args.gpu) +
               " --query-compute-apps=pid --format=csv,noheader | "
               "python3 -c 'import sys; sys.exit(bool(sys.stdin.read().strip()))'", timeout=15)
    else:
        r = run(["nvidia-smi", "-i", str(args.gpu), "--query-compute-apps=pid",
                 "--format=csv,noheader"], capture=True)
        if r.stdout.strip():
            raise ValueError("GPU has a process; wait for its owner")
        run(["docker", "exec", "lerobot", "nvidia-smi"], capture=True, timeout=15)


def eval_dispatch(args):
    root = args.root.resolve()
    verify_inputs(root)
    if args.machine == "kanu":
        run([sys.executable, CODE / MODULE / "eval.py", "--root", root,
             "--repo", args.runtime_repo, "--machine", "kanu", "--gpu", str(args.gpu)], timeout=None)
        return
    session = "lr-equal-budget-20260907"
    if not args.central_lease:
        # Central ledger lives in the main checkout, not in an isolated worktree.
        run(["bash", args.runtime_repo / "scripts/utils/with_gpu_lease.sh", "srv48", str(args.gpu),
             session, "equal-budget LR eval", "--", sys.executable, Path(__file__),
             "eval-dispatch", "--root", root, "--runtime-repo", args.runtime_repo,
             "--machine", "srv48", "--gpu", str(args.gpu), "--revision", args.revision,
             "--branch", args.branch, "--central-lease"], timeout=None)
        return
    rel, files = data_files(root)
    dest = remote_code("srv48", args.revision, args.branch)
    # Keep data under the runtime repo, which is mounted into RoboCasa Docker.
    helper("srv48", "push-data", *files, str(rel / "fit"), str(rel / "FIT_DONE"), timeout=600)
    cmd = ("LR_CENTRAL_LEASE_SESSION=" + shlex.quote(session) +
           " LR_CENTRAL_LEASE_CONFIRMED=1 python3 " + dest + "/" + MODULE +
           "/eval.py --root ~/pkt_ws/temporal_vla/" + str(rel) +
           " --repo ~/pkt_ws/temporal_vla --machine srv48 --gpu " + str(args.gpu))
    helper("srv48", "run", cmd, timeout=None)
    helper("srv48", "pull-results", str(rel / "eval" / "srv48"), timeout=600)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["archive-ready", "archive-fit", "fit-here", "eval-ready", "eval-dispatch"])
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--runtime-repo", type=Path, default=Path("/home/dongkyu/pkt_ws/temporal_vla"))
    ap.add_argument("--revision", default="")
    ap.add_argument("--branch", default="exp/lr-equal-budget-20260907")
    ap.add_argument("--shard-root", type=Path)
    ap.add_argument("--machine", choices=["kanu", "srv48"], default="kanu")
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--central-lease", action="store_true")
    args = ap.parse_args()
    globals()[args.stage.replace("-", "_")](args)


if __name__ == "__main__":
    main()
