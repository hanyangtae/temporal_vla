#!/usr/bin/env python3
"""Durable, stage-gated orchestration for the Stage 2 SAFE/QAM/eval run.

The default mode is a dry-run.  ``--run`` is required to create child
processes; every child is owned by this process and is terminated on Ctrl-C.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


SAFE_GPUS = ("2", "3", "4", "6", "7")
QAM_GPUS = ("0", "1")
EVAL_GPUS = ("0", "1", "2", "3", "4", "6", "7")
HERE = Path(__file__).resolve().parent


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _record(status: dict[str, Any], path: Path, stage: str, state: str, **extra: Any) -> None:
    status.setdefault("stages", {})[stage] = {"state": state, "updated_at": _now(), **extra}
    _atomic_json(path, status)


def _py(args: argparse.Namespace, script: str) -> list[str]:
    return [args.python, str(HERE / script)]


def build_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Return the complete plan without checking or mutating experiment data."""
    cache = Path(args.cache).resolve()
    parent = cache.parent
    output = Path(args.output).resolve()
    manifest = parent / "manifest.json"
    clusters = parent / "clusters.json"
    safe = output / "safe_sweep"
    selected = safe / "selected"
    qam_base = output / "qam_base"
    qam_shaped = output / "qam_cluster_potential"
    common_qam = ["--qam-root", str(Path(args.rl2_root).resolve() / "third_party/qam"), "--training-cache", str(cache),
                  "--manifest", str(manifest), "--checkpoint", str(Path(args.qam_initial).resolve()),
                  "--warmup-steps", "5000", "--steps", "50000", "--seed", "42"]
    safe_jobs = []
    gpu_slots = [g for g in args.safe_gpus for _ in (0, 1)]
    for shard, gpu in enumerate(gpu_slots):
        safe_jobs.append({"gpu": gpu, "shard": shard, "command": _py(args, "safe_sweep.py") +
                          ["--cache", str(cache), "--safe-root", str(Path(args.rl2_root).resolve() / "third_party/SAFE"),
                           "--output", str(safe), "--shards", "10", "--shard", str(shard), "--epochs", "1000", "2000"]})
    qam_jobs = [
        {"gpu": args.qam_gpus[0], "mode": "base", "output": qam_base,
         "command": _py(args, "train_qam.py") + common_qam + ["--reward-mode", "base", "--output", str(qam_base)]},
        {"gpu": args.qam_gpus[1], "mode": "cluster_potential", "output": qam_shaped,
         "command": _py(args, "train_qam.py") + common_qam + ["--reward-mode", "cluster_potential", "--cluster-bundle", str(clusters), "--output", str(qam_shaped)]},
    ]
    eval_common = [_py(args, "run_experiment.py") + ["--mode", "eval", "--rl2-root", str(Path(args.rl2_root).resolve()),
                    "--gpus", *args.eval_gpus, "--output", str(output / "eval"), "--policy", args.policy,
                    "--training-cache", str(cache), "--manifest", str(manifest), "--safe-dir", str(selected),
                    "--qam-original", str(Path(args.qam_initial).resolve()), "--qam-base", str(qam_base / "params_50000.pkl"),
                    "--qam-shaped", str(qam_shaped / "params_50000.pkl"), "--trials", "25", "--run"]]
    eval_common[0] += ["--seeds", "42"]
    eval_pilot = eval_common[0]
    eval_followup = eval_common[0][:-3] + ["--seeds", "0", "7", "--run"]
    return [
        {"name": "safe_sweep", "jobs": safe_jobs, "expected_jobs": 576},
        {"name": "safe_finalize", "jobs": [{"gpu": args.safe_gpus[0], "command": _py(args, "safe_sweep.py") + ["--cache", str(cache), "--safe-root", str(Path(args.rl2_root).resolve() / "third_party/SAFE"), "--output", str(safe), "--mode", "finalize"]}]},
        {"name": "qam", "jobs": qam_jobs, "expected_jobs": 2},
        {"name": "eval_pilot", "jobs": [{"command": eval_pilot}]},
        {"name": "eval_followup", "jobs": [{"command": eval_followup}]},
        {"name": "analyze", "jobs": [{"command": _py(args, "analyze.py") + ["--eval-root", str(output / "eval"), "--output", str(output / "summary.json")]}]},
    ]


def _verify_qam(job: dict[str, Any], args: argparse.Namespace) -> bool:
    out=Path(job['output']); done=out/'DONE.json'; flags=out/'flags.json'
    if not done.is_file() or not flags.is_file():return False
    try:
        d,f=json.loads(done.read_text()),json.loads(flags.read_text());c=Path(d['checkpoint'])
        if not c.is_absolute():c=Path.cwd()/c
        stage=f.get('stage2',{})
        return (c.is_file() and c.resolve()==(out/'params_50000.pkl').resolve()
                and d.get('sha256')==_sha256(c)
                and f.get('manifest_sha256')==_sha256(Path(args.cache).resolve().parent/'manifest.json')
                and f.get('initial_checkpoint_sha256')==_sha256(Path(args.qam_initial).resolve())
                and all(stage.get(k)==v for k,v in {'steps':50000,'warmup_steps':5000,'seed':42,'batch_size':256,'reward_mode':job['mode']}.items()))
    except (OSError,KeyError,ValueError):return False


def _contract(args):
    cache=Path(args.cache).resolve()
    policy=Path(args.policy).resolve()
    return {'cache_sha256':_sha256(cache),'manifest_sha256':_sha256(cache.parent/'manifest.json'),
            'qam_initial_sha256':_sha256(Path(args.qam_initial)),
            'policy_sha256':_sha256(policy/'model.safetensors'),
            'policy_config_sha256':_sha256(policy/'config.json'),
            'source_sha256':{f:_sha256(HERE/f) for f in ('safe_sweep.py','train_qam.py','run_experiment.py')},
            'safe_epochs':[1000,2000],'qam_updates':50000,'qam_warmup':5000,'qam_seed':42,
            'eval_trials':25,'eval_seeds':[42,0,7],'rl2_root':str(Path(args.rl2_root).resolve())}


def _run_jobs(stage: str, jobs: list[dict[str, Any]], output: Path, status: dict[str, Any], status_path: Path) -> None:
    procs: list[tuple[subprocess.Popen[str], Path]] = []
    try:
        for i, job in enumerate(jobs):
            log = output / "logs" / stage / f"worker_{i:02d}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",TF_FORCE_GPU_ALLOW_GROWTH="true",XLA_PYTHON_CLIENT_PREALLOCATE="false",PYTHONUNBUFFERED="1")
            if "gpu" in job:
                env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
            handle = log.open("a", encoding="utf-8")
            proc = subprocess.Popen(job["command"], stdout=handle, stderr=subprocess.STDOUT, text=True, env=env)
            handle.close()
            procs.append((proc, log))
            job["pid"] = proc.pid
        _record(status, status_path, stage, "running", jobs=jobs)
        codes = [proc.wait() for proc, _ in procs]
    except KeyboardInterrupt:
        for proc, _ in procs:
            if proc.poll() is None:
                proc.terminate()
        raise
    if any(code != 0 for code in codes):
        raise RuntimeError(f"{stage} failed: exit codes {codes}")


def run(args: argparse.Namespace) -> int:
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(output/'pipeline.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path=output/'status.json'
    status={'schema':1,'started_at':_now(),'run':bool(args.run),'pid':os.getpid(),'stages':{}}
    plan=build_plan(args); stages={x['name']:x for x in plan}
    if not args.run:
        _atomic_json(output/'plan.json',{'created_at':_now(),'stages':plan})
        for stage in plan:_record(status,status_path,stage['name'],'dryrun',jobs=stage['jobs'])
        return 0
    name='preflight'
    try:
        contract=_contract(args);path=output/'contract.json'
        if path.exists() and json.loads(path.read_text())!=contract:raise ValueError('pipeline restart contract mismatch')
        _atomic_json(path,contract);_atomic_json(output/'plan.json',{'created_at':_now(),'stages':plan})
        qam_pending=[]
        for job in stages['qam']['jobs']:
            if _verify_qam(job,args):continue
            if Path(job['output']).exists():raise RuntimeError(f"partial/invalid QAM output preserved: {job['output']}")
            qam_pending.append(job)
        name='training'
        _run_jobs(name,stages['safe_sweep']['jobs']+qam_pending,output,status,status_path)
        if not all(_verify_qam(j,args) for j in stages['qam']['jobs']):raise RuntimeError('QAM DONE/provenance invalid')
        for label in ('training','safe_sweep','qam'):_record(status,status_path,label,'complete')
        for name in ('safe_finalize','eval_pilot','eval_followup','analyze'):
            jobs=stages[name]['jobs'];_run_jobs(name,jobs,output,status,status_path)
            if name=='safe_finalize':
                from safe_provenance import validate
                safe=output/'safe_sweep'/'selected';validate(safe,safe/'provenance.json')
            _record(status,status_path,name,'complete',jobs=jobs)
        _record(status,status_path,'pipeline','complete')
        return 0
    except (Exception,KeyboardInterrupt) as exc:
        _record(status,status_path,name,'failed',error=str(exc));return 1
    finally:
        lock.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True); p.add_argument("--rl2-root", required=True); p.add_argument("--output", required=True)
    p.add_argument("--policy", required=True); p.add_argument("--qam-initial", required=True); p.add_argument("--python", default=sys.executable)
    p.add_argument("--safe-gpus", nargs="+", default=list(SAFE_GPUS)); p.add_argument("--qam-gpus", nargs="+", default=list(QAM_GPUS)); p.add_argument("--eval-gpus", nargs="+", default=list(EVAL_GPUS)); p.add_argument("--run", action="store_true")
    args = p.parse_args(argv)
    if len(args.safe_gpus) != 5 or len(args.qam_gpus) != 2 or len(args.eval_gpus) < 1:
        p.error("expected five SAFE GPUs, two QAM GPUs, and at least one eval GPU")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
