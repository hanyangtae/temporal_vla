#!/usr/bin/env python3
"""Sequential home-machine eval of equal-budget models; no remote dispatch.

Dry-run never contacts a GPU or claims a lease. Real srv48 runs additionally
require the caller's central-kanu lease attestation in LR_CENTRAL_LEASE_SESSION
and LR_CENTRAL_LEASE_CONFIRMED=1; the caller must keep that central owner alive.
This process claims the actual home machine's local ledger with its own PID.
It never terminates unrelated jobs. The existing runner owns its serve cleanup.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import socket
import subprocess
from collections import defaultdict
from pathlib import Path


def table(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def key(row):
    return tuple(int(row[k]) for k in ("scene_idx", "jitter_idx", "noise_idx", "env_seed", "inference_seed"))


def completed(path, expected, *, require_failure=False):
    """False only for absent/empty results; partial or wrong sets fail loudly.

    Never trust row count: duplicates, extra coordinates, wrong seeds, malformed
    outcomes, or nonmatching collection labels are rejected before runner resume.
    Partial results require a separately reviewed missing-case resume manifest.
    """
    path = Path(path)
    if not path.exists():
        return False
    rows = table(path)
    if not rows:
        return False
    want = {key(r): int(r["success"]) for r in expected}
    actual = [key(r) for r in rows]
    if len(set(actual)) != len(actual):
        raise ValueError(f"duplicate completion coordinates: {path}")
    if set(actual) != set(want):
        raise ValueError(f"partial/mismatched completion: {path}; missing={sorted(set(want)-set(actual))}, extra={sorted(set(actual)-set(want))}")
    for row in rows:
        if row.get("success") not in ("0", "1"):
            raise ValueError(f"invalid success outcome: {path}")
        if int(row.get("collection_success", -1)) != want[key(row)]:
            raise ValueError(f"source outcome mismatch: {path}")
        if require_failure and row["success"] != "0":
            raise ValueError(f"baseline replay gate did not reproduce failure: {path}")
    return True


def build_jobs(root, repo, machine, gpu, *, fit_root=None, repeat0=False, port=None):
    root, repo = Path(root).resolve(), Path(repo).resolve()
    fit_root = Path(fit_root).resolve() if fit_root else root / "fit"
    port = port if port is not None else (8860 if machine == "kanu" else 8890) + gpu
    plans = json.loads((root / "diagnostics.json").read_text())["plans"]
    manifests = table(root / "manifest.tsv")
    failed = table(root / "eval_cases.tsv")
    preserved = table(root / "preservation_eval.tsv")
    by_pair, evaluations = defaultdict(list), defaultdict(list)
    for row in manifests:
        by_pair[row["pair_id"]].append(row)
    for row in failed + preserved:
        evaluations[row["pair_id"]].append(row)
    runner = repo / "scripts/steer/online_gated/run_online_gated_eval.sh"
    if not runner.is_file():
        raise FileNotFoundError(runner)
    selected = [p for p in plans if p["status"] == "planned" and (not repeat0 or p["request"]["repeat"] == 0)]
    jobs, coverage, gates = [], [], {}
    for plan in selected:
        pair = plan["pair_id"]
        rows = sorted(evaluations[pair], key=key)
        if not rows:
            raise ValueError(f"no evaluation cases for {pair}")
        original = {r["original_key"] for r in rows}
        machines = {r["machine"] for r in rows}
        aliases = {"kanu": {"kanu"}, "srv48": {"worker1", "srv48"}}[machine]
        if not machines <= aliases:
            if machines & aliases:
                raise ValueError(f"ambiguous collection machine for {pair}: {machines}")
            continue
        if len(original) != 1 or len({key(r) for r in rows}) != len(rows):
            raise ValueError(f"ambiguous/duplicated evaluation coordinates for {pair}")
        slug = next(iter(original)).replace("/", "_")
        scene_jitter = {(r["scene_idx"], r["jitter_idx"]) for r in rows}
        if len(scene_jitter) != 1:
            raise ValueError(f"multiple target cells in {pair}")
        scene, jitter = next(iter(scene_jitter))
        arms = sorted({r["arm"] for r in by_pair[pair]})
        if len(arms) != 2:
            raise ValueError(f"expected two fit arms: {pair}")
        common = {"SLUGS": slug, "GPUS": str(gpu), "SERVES_PER_GPU": "1", "PORT_BASE": str(port),
                  "ALLOW_BUSY_GPU": "0", "EP_MODE": "replay", "EVAL_SCENES": scene, "EVAL_JITTERS": jitter,
                  "REPLAY_MACHINE": next(iter(machines)), "PLAN_JSON": str(root / "sources/collection_plan.json"),
                  "INDEX_TSV": str(root / "sources/index_v6_complete_cells.tsv"), "FAILURE_TASK": pair,
                  "FAILURE_ALPHA": "0.1", "STEER_BETA": "0.8", "STEER_OP": "setpoint",
                  "PERSTEP_FALLBACK": "reseed", "RESEED_OFFSET": "900000", "NAS": "5", "MAXEP": "720",
                  "CAPTURE_FEATURES": "0", "EP_META_DIR": "", "EP_META_LOAD_ENV_NAME": "",
                  "CLUSTER_BUNDLE": "", "TOKEN_POOL": "all_token_full",
                  "DETECTOR_LAYERS": "0,2,4,8,10,12,15", "PROX": "0",
                  "FIT_SCENES": scene, "FIT_NOISES": ",".join(sorted({r['noise_idx'] for r in by_pair[pair]})),
                  "SERVE_MODE": "docker" if machine == "kanu" else "host"}
        if machine == "srv48":
            common.update(SERVE_PY=str(Path.home() / "miniconda3/envs/lerobot_050_groot/bin/python"),
                          SERVE_PYTHONPATH=str(repo / "lerobot/src"))

        def job(model_arm, condition, cases, gate=False):
            fit = fit_root / pair / model_arm
            variant = condition if condition in ("plain", "lr_jfair") else "plain"
            outer = root / "eval" / machine / pair / model_arm / condition
            if gate:
                gate_hash = hashlib.sha256(plan["target_key"].encode()).hexdigest()[:16]
                outer = root / "eval" / machine / "replay_gates" / gate_hash
            runner_arm = "ps_setm" if condition in ("plain", "lr_jfair") else condition
            npzroot = root / "eval" / machine / "npz_roots" / pair / model_arm
            env = {**common, "ARMS": runner_arm, "DETECTOR_CKPT": str(fit / "detector.pt"),
                   "OUT_ROOT": str(outer), "EVAL_NOISES": ",".join(str(n) for n in sorted({int(r['noise_idx']) for r in cases})),
                   "NPZ_ROOT": str(npzroot), "NPZ_VARIANT": variant}
            return {"pair_id": pair, "model_arm": model_arm, "condition": condition, "gate": gate,
                    "expected": cases, "env": env, "command": ["bash", str(runner)],
                    "result": str(outer / slug / runner_arm / "per_episode.tsv"),
                    "operator_link": str(npzroot / slug / variant), "operator_source": str(fit / "operators" / variant),
                    "fit_report": str(fit / "fit_report.json"), "pair_report": str(fit_root / pair / "pair_report.json"),
                    "log": str(outer / "eval_wrapper.log")}

        if plan["target_key"] not in gates:
            failures = [r for r in rows if int(r["success"]) == 0]
            if not failures:
                raise ValueError(f"no failure for replay gate: {pair}")
            solo = next(a for a in arms if a.endswith("_only"))
            gate = job(solo, "ps_base", failures[:1], gate=True)
            gates[plan["target_key"]] = gate
            jobs.append(gate)
        for arm in arms:
            fit_report = fit_root / pair / arm / "fit_report.json"
            registered = json.loads(fit_report.read_text())["registered"] if fit_report.exists() else None
            jobs.append(job(arm, "ps_reseed", rows))
            for variant in ("plain", "lr_jfair"):
                phases = registered.get(variant, []) if registered is not None else None
                coverage.append({"pair_id": pair, "arm": arm, "variant": variant,
                                 "registered_phases": phases, "status": "awaiting_fit" if phases is None else ("supported" if phases else "unsupported_no_phase")})
                if phases is None or phases:
                    jobs.append(job(arm, variant, rows))
    return jobs, coverage


def check_idle(gpu, port):
    query = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid", "--format=csv,noheader"],
                           check=True, capture_output=True, text=True)
    pids = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if pids:
        details = subprocess.run(["ps", "-o", "user,pid,args", "-p", ",".join(pids)], capture_output=True, text=True)
        raise RuntimeError(f"GPU {gpu} is occupied; no processes may be shared:\n{details.stdout}")
    with socket.socket() as sock:
        sock.bind(("0.0.0.0", port))
    # The existing runner cleans up by this port; reject stale serve processes
    # even if they are currently not listening, to avoid killing others' work.
    processes = subprocess.run(["ps", "-eo", "pid,args"], check=True, capture_output=True, text=True).stdout
    for line in processes.splitlines():
        if "serve/lerobot.py" in line and (f"--port {port} " in line + " " or f"--port={port} " in line + " "):
            raise RuntimeError(f"serve port {port} already owned: {line}")


def run_owned(command, *, env, cwd, stream):
    """On interruption, terminate only our runner's group and await its cleanup."""
    def interrupted(signum, frame):
        raise InterruptedError(f"eval interrupted by signal {signum}")

    saved = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    proc = None
    try:
        proc = subprocess.Popen(command, env=env, cwd=cwd, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
        code = proc.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    except BaseException:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
        raise
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--machine", choices=("kanu", "srv48"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--fit-root", type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--repeat0", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.gpu < 0 or args.gpu >= (8 if args.machine == "kanu" else 4):
        parser.error("GPU index outside machine inventory")
    args.root, args.repo = args.root.resolve(), args.repo.resolve()
    jobs, coverage = build_jobs(args.root, args.repo, args.machine, args.gpu,
                               fit_root=args.fit_root, repeat0=args.repeat0, port=args.port)
    if not jobs:
        raise RuntimeError("no jobs for this machine/selection")
    output = args.root / "eval" / args.machine
    output.mkdir(parents=True, exist_ok=True)
    schedule = {"jobs": jobs, "operator_coverage": coverage, "dry_run": args.dry_run,
                "n_jobs": len(jobs), "n_episode_runs": sum(len(j["expected"]) for j in jobs)}
    (output / ("dry_run.json" if args.dry_run else "schedule.json")).write_text(json.dumps(schedule, indent=2) + "\n")
    if args.dry_run:
        print(json.dumps({"dry_run": True, "n_jobs": len(jobs), "n_episode_runs": schedule["n_episode_runs"],
                          "awaiting_operator_fits": sum(c["status"] == "awaiting_fit" for c in coverage),
                          "output": str(output / "dry_run.json")}))
        return
    if any(c["status"] == "awaiting_fit" for c in coverage):
        raise RuntimeError("fit reports missing; no GPU claimed")
    for task in jobs:
        for required in (task["env"]["DETECTOR_CKPT"], task["fit_report"], task["pair_report"],
                         task["env"]["PLAN_JSON"], task["env"]["INDEX_TSV"]):
            if not Path(required).is_file():
                raise FileNotFoundError(required)
        if json.loads(Path(task["pair_report"]).read_text())["status"] != "complete":
            raise RuntimeError(f"incomplete fit: {task['pair_report']}")
        completed(task["result"], task["expected"], require_failure=task["gate"])
    session = os.environ.get("LR_CENTRAL_LEASE_SESSION", f"lr-equal-{os.getpid()}")
    if args.machine != "kanu" and os.environ.get("LR_CENTRAL_LEASE_CONFIRMED") != "1":
        raise RuntimeError("srv48 requires live central lease wrapper attestation")
    lease = args.repo / "scripts/utils/gpu_lease.sh"
    lease_meta = args.repo / "outputs/gpu_leases" / f"{args.machine}_gpu{args.gpu}" / "meta"
    # Local lease always belongs to this eval PID; reject any preexisting owner.
    subprocess.run(["bash", str(lease), "claim", args.machine, str(args.gpu), session,
                    "lr_equal_budget_eval"], env={**os.environ, "LEASE_PID": str(os.getpid())}, check=True)
    try:
        for task in jobs:
            if completed(task["result"], task["expected"], require_failure=task["gate"]):
                continue
            meta = dict(line.split("=", 1) for line in lease_meta.read_text().splitlines())
            if meta.get("session") != session or meta.get("pid") != str(os.getpid()):
                raise RuntimeError("local lease ownership changed")
            check_idle(args.gpu, int(task["env"]["PORT_BASE"]))
            if task["condition"] in ("plain", "lr_jfair"):
                source, link = Path(task["operator_source"]), Path(task["operator_link"])
                if not list(source.rglob("*.npz")):
                    raise RuntimeError(f"registered operator has no NPZ: {source}")
                link.parent.mkdir(parents=True, exist_ok=True)
                if link.is_symlink():
                    if link.resolve() != source.resolve():
                        raise RuntimeError(f"wrong existing operator link: {link}")
                elif link.exists():
                    raise RuntimeError(f"refuse replacing non-symlink: {link}")
                else:
                    # Relative links resolve under both host and Docker mounts.
                    link.symlink_to(os.path.relpath(source, link.parent), target_is_directory=True)
            log = Path(task["log"])
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as stream:
                run_owned(task["command"], env={**os.environ, **task["env"]}, cwd=args.repo, stream=stream)
            if not completed(task["result"], task["expected"], require_failure=task["gate"]):
                raise RuntimeError(f"runner returned without complete results: {task['result']}")
        for task in jobs:
            if not completed(task["result"], task["expected"], require_failure=task["gate"]):
                raise RuntimeError("final exact completion audit failed")
        check_idle(args.gpu, int(jobs[0]["env"]["PORT_BASE"]))
        sentinel = "EVAL_REPEAT0_DONE" if args.repeat0 else "EVAL_DONE"
        (output / sentinel).write_text(f"{sentinel}_{args.machine}\n")
    finally:
        # This only releases our own ledger entry; cleanup of serve is delegated
        # to the existing subprocess runner's EXIT trap.
        subprocess.run(["bash", str(lease), "release", args.machine, str(args.gpu), session], check=True)


if __name__ == "__main__":
    main()
