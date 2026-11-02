"""Persistent atomic65 common-QAM orchestrator (run detached by the main session).

  stage until 65/65 -> full manifest -> extraction shards (one per GPU, harness guard)
  -> finalize -> preflight -> fresh QAM 500k (GPU lease) -> archive check -> DONE.json

- missing-only resume: every step is skipped when its output is already complete
  (staging receipts, identical manifest, per-episode extraction records, COMPLETE.json).
- source freeze: hashes of every code file the run depends on are recorded on first start;
  any later difference blocks (re-freeze only with --refreeze after review).
- gates: production extraction needs the smoke extraction COMPLETE; training needs the
  smoke training COMPLETE (20 steps). A train dir without COMPLETE is never reused.
- status: <work>/pipeline/STATUS.json (stage, pids, counts), events.jsonl, per-step logs;
  on any error STATUS=blocked, ALERTS.jsonl entry, and $NOTIFY_CMD "<message>" if set.
This file only orchestrates; it does not modify extract/run_extract/train.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from experiment import ADAPTER, N_TASKS, RECIPE, REPO, STAGING_ROOT, TARGET13, WORK, sha256_path  # noqa: E402

PY = REPO / ".venvs/rl2_iid/bin/python"
REL = HERE.relative_to(REPO)
FROZEN_GLOBS = [(HERE, "*.py"), (HERE, "*.sh"), (ADAPTER, "*.py"), (TARGET13, "multitask_data.py"),
                (REPO / "scripts/rl2_vla/groot_multitask_20261021", "extract.py"),
                (REPO / "scripts/rl2_vla/atomic65_stage_20261028", "atomic65_stage.py"),
                (REPO / "scripts/utils", "remote_compute.sh"), (REPO / "scripts/utils", "with_gpu_lease.sh"),
                (REPO / "scripts/utils", "groot_launch_guard.sh")]


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Blocked(RuntimeError):
    pass


class Pipeline:
    def __init__(self, a, *, runner=None, stage_fn=None, manifest_fn=None, finalize_fn=None, sleep=time.sleep):
        self.a = a
        self.work: Path = a.work
        self.pdir = self.work / "pipeline"
        self.pdir.mkdir(parents=True, exist_ok=True)
        self.runner = runner or self._run
        self.stage_fn = stage_fn
        self.manifest_fn = manifest_fn
        self.finalize_fn = finalize_fn
        self.sleep = sleep
        self.status = {"pid": os.getpid(), "started_utc": now()}

    # ---------------------------------------------------------------- bookkeeping
    def event(self, stage: str, **kw):
        self.status.update(stage=stage, utc=now(), **kw)
        tmp = self.pdir / "STATUS.json.tmp"
        tmp.write_text(json.dumps(self.status, indent=1) + "\n"); tmp.rename(self.pdir / "STATUS.json")
        with open(self.pdir / "events.jsonl", "a") as f:
            f.write(json.dumps({"utc": now(), "stage": stage, **kw}) + "\n")
        print(json.dumps({"stage": stage, **kw}), flush=True)

    def alert(self, msg: str):
        with open(self.pdir / "ALERTS.jsonl", "a") as f:
            f.write(json.dumps({"utc": now(), "message": msg}) + "\n")
        cmd = os.environ.get("NOTIFY_CMD")
        if cmd:
            subprocess.run(shlex.split(cmd) + [f"[atomic65-qam] {msg}"], timeout=60, check=False)

    def _run(self, name: str, argv: list[str], env: dict | None = None, wait: bool = True):
        log = self.pdir / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        f = open(log, "a")
        f.write(f"# {now()} {' '.join(map(shlex.quote, argv))}\n"); f.flush()
        p = subprocess.Popen(argv, cwd=REPO, env=dict(os.environ, **(env or {})), stdout=f,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=False)
        if not wait:
            return p, f
        rc = p.wait(); f.close()
        return rc

    # ---------------------------------------------------------------- source freeze
    def freeze(self):
        files = {}
        for d, pat in FROZEN_GLOBS:
            for p in sorted(d.glob(pat)):
                files[os.path.relpath(p, REPO)] = sha256_path(p)
        fp = self.pdir / "SOURCE_FREEZE.json"
        if fp.exists() and not self.a.refreeze:
            old = json.loads(fp.read_text())
            diff = sorted(k for k in set(old) | set(files) if old.get(k) != files.get(k))
            if diff:
                raise Blocked(f"source changed since freeze: {diff[:8]} (review, then --refreeze)")
        else:
            fp.write_text(json.dumps(files, indent=1) + "\n")
            if self.a.refreeze:   # reviewed re-freeze applies once, at startup; later checks are strict
                self.a.refreeze = False
                self.event("source_refrozen", files=len(files))
        self.status["source_freeze_sha256"] = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()

    # ---------------------------------------------------------------- steps
    def staged_count(self) -> int:
        return len(list((self.a.staging / "receipts").glob("*.json"))) if (self.a.staging / "receipts").exists() else 0

    def step_stage(self):
        deadline = time.time() + self.a.stage_timeout_h * 3600
        while True:
            self.freeze()
            from staging import StageLocked
            try:
                if self.stage_fn:
                    s = self.stage_fn()
                else:
                    from staging import stage
                    s = stage(self.a.staging, tasks=None, jobs=2, reserve=int(self.a.reserve_gib * 1024 ** 3))
            except StageLocked as e:   # another stager (e.g. the main session) is running: wait, never overlap
                self.event("staging", waiting_for_lock=str(e), staged=self.staged_count())
                if time.time() > deadline:
                    raise Blocked(f"staging lock still held at timeout: {e}")
                self.sleep(self.a.poll_s)
                continue
            if s["failed"]:
                raise Blocked(f"staging failed: {s['failed']}")
            n = self.staged_count()
            self.event("staging", staged=n, waiting_for_stager=len(s["waiting_for_stager"]))
            if n == N_TASKS:
                return
            if time.time() > deadline:
                raise Blocked(f"staging timeout: {n}/{N_TASKS}")
            self.sleep(self.a.poll_s)

    def step_manifest(self):
        man = self.work / "metadata/manifest.json"
        if man.exists():
            self.event("manifest", skipped=True, sha256=sha256_path(man)); return
        rc = self.manifest_fn() if self.manifest_fn else self.runner(
            "manifest", [str(PY), str(REL / "cli.py"), "--work", str(self.work), "--staging", str(self.a.staging), "manifest"])
        if rc != 0 or not man.exists():
            raise Blocked("full manifest blocked (see metadata/manifest.json.errors.json)")
        self.event("manifest", sha256=sha256_path(man))

    def preflight(self, step: str):
        rc = self.runner(f"preflight_{step}", [str(PY), str(REL / "cli.py"), "--work", str(self.work),
                                               "--staging", str(self.a.staging), "preflight", "--step", step])
        if rc != 0:
            raise Blocked(f"preflight {step} blocked (see pipeline/logs/preflight_{step}.log)")

    # ---------------------------------------------------------------- GPU admission
    def gpu_busy(self) -> dict[int, list[int]]:
        """{gpu index: compute-process pids}. Any nvidia-smi failure -> every GPU busy (fail closed)."""
        try:
            idx = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=60, check=True).stdout
            apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=60, check=True).stdout
        except Exception as e:
            self.event("gpu_probe_failed", error=repr(e))
            return {g: [-1] for g in self.a.gpu_pool}
        uuid = {u.strip(): int(i) for i, u in (l.split(",") for l in idx.splitlines() if l.strip())}
        busy = {g: [] for g in uuid.values()}
        for l in apps.splitlines():
            if l.strip():
                u, pid = (x.strip() for x in l.split(","))
                busy.setdefault(uuid.get(u, -1), []).append(int(pid))
        return busy

    def free_gpus(self, in_use: set[int]) -> list[int]:
        busy = self.gpu_busy()
        return [g for g in self.a.gpu_pool if g not in in_use and not busy.get(g, [-1])]

    @staticmethod
    def admission_refused(log_text: str) -> bool:
        """Harness/lease refused to start (nothing ran): wait and retry, never count as a failure."""
        return ("[harness] BLOCKED" in log_text) or ("[with_gpu_lease]" in log_text and "발사 중단" in log_text)

    def _launch(self, name, argv, env):
        log = self.pdir / "logs" / f"{name}.log"
        offset = log.stat().st_size if log.exists() else 0
        p, f = self.runner(name, argv, env, wait=False)
        return {"proc": p, "file": f, "log": log, "offset": offset}

    @staticmethod
    def _segment(job) -> str:
        try:
            with open(job["log"], "rb") as f:
                f.seek(job["offset"]); return f.read().decode("utf-8", "replace")
        except FileNotFoundError:
            return ""

    def shard_missing(self, k: int, n: int) -> int:
        """Episodes of shard k (manifest order j % n == k) without a committed record (missing-only)."""
        from extract import ident   # identity function only (no GPU imports)
        m = json.loads((self.work / "metadata/manifest.json").read_text())
        lat = self.work / "latents"
        return sum(1 for j, e in enumerate(m["episodes"]) if j % n == k and not (lat / (ident(e) + ".json")).exists())

    def step_extract(self):
        if (self.work / "latents/COMPLETE.json").exists():
            self.event("extract", skipped=True); return
        if not (self.work / "smoke_latents/COMPLETE.json").exists():
            raise Blocked("smoke extraction not COMPLETE")
        self.preflight("extract")
        n = self.a.num_shards
        pending = [k for k in range(n) if self.shard_missing(k, n)]
        self.event("extract_queue", pending={f"shard{k}": self.shard_missing(k, n) for k in pending},
                   complete_shards=[k for k in range(n) if k not in pending])
        running, attempts = {}, {k: 0 for k in range(n)}
        deadline = time.time() + self.a.queue_timeout_h * 3600
        while pending or running:
            for k in list(running):
                job = running[k]; rc = job["proc"].poll()
                if rc is None:
                    continue
                job["file"].close(); del running[k]
                missing = self.shard_missing(k, n)
                if rc == 0 and not missing:
                    self.event("extract_shard_done", shard=k, gpu=job["gpu"])
                elif self.admission_refused(self._segment(job)):
                    self.event("extract_admission_refused", shard=k, gpu=job["gpu"]); pending.append(k)
                else:
                    attempts[k] += 1
                    self.alert(f"extract shard{k} gpu{job['gpu']} rc={rc} missing={missing} attempt {attempts[k]}")
                    if attempts[k] > self.a.max_retries:
                        raise Blocked(f"extract shard{k} failed {attempts[k]}x (last rc={rc}); records kept")
                    pending.append(k)
            capacity = self.a.max_gpus - len(running)
            if pending and capacity > 0:
                for g in self.free_gpus({j["gpu"] for j in running.values()})[:capacity]:
                    if not pending:
                        break
                    self.freeze()
                    k = pending.pop(0)
                    env = {"GPUS": str(g), "HARNESS_SESSION": f"{self.a.session}-extract-s{k}"}
                    argv = ["bash", str(REL / "run_extract.sh"), "--shard", str(k), "--num-shards", str(n)]
                    running[k] = dict(self._launch(f"extract_shard{k}of{n}", argv, env), gpu=g)
                    self.event("extract_launch", shard=k, gpu=g, pid=running[k]["proc"].pid,
                               missing=self.shard_missing(k, n))
            if pending and not running:
                self.event("extract_waiting_for_gpu", pending=pending, pool=self.a.gpu_pool)
            if time.time() > deadline:
                raise Blocked(f"extract queue timeout (pending={pending}, running={list(running)})")
            if pending or running:
                self.sleep(self.a.poll_s)
        rc = self.finalize_fn() if self.finalize_fn else self.runner(
            "finalize_extract", [str(PY), str(REL / "cli.py"), "--work", str(self.work), "finalize-extract"])
        if rc != 0:
            raise Blocked("finalize-extract blocked")
        self.event("extract", complete=True)

    def step_train(self):
        out = self.work / "scratch_run"
        done = out / "COMPLETE.json"
        if done.exists():
            self.event("train", skipped=True); return
        smoke = self.work / "scratch_smoke/COMPLETE.json"
        if not smoke.exists() or json.loads(smoke.read_text()).get("steps") != RECIPE["smoke_steps"]:
            raise Blocked("train smoke (20 steps) not COMPLETE")
        attempts = 0
        deadline = time.time() + self.a.queue_timeout_h * 3600
        while not done.exists():
            from train import resumable_steps   # file-level checkpoint scan only (no JAX import)
            resume = out.exists()
            if resume:
                ok = resumable_steps(out)
                if not ok:
                    raise Blocked(f"{out} exists but has no verified checkpoint with sampler_state.json; not reusable")
            else:
                self.preflight("train")
            free = self.free_gpus(set())
            if not free:
                if time.time() > deadline:
                    raise Blocked("train queue timeout: no free GPU in pool")
                self.event("train_waiting_for_gpu", pool=self.a.gpu_pool, resume=resume)
                self.sleep(self.a.poll_s); continue
            g = str(free[0])
            self.freeze()
            env = {"JAX_PLATFORMS": "cuda", "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "CUDA_VISIBLE_DEVICES": g,
                   "OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "4"}
            argv = ["bash", "scripts/utils/with_gpu_lease.sh", "kanu", g, f"{self.a.session}-train",
                    "atomic65-qam-train500k", "--", str(PY), "-u", str(REL / "train.py"),
                    "--manifest", str(self.work / "metadata/manifest.json"),
                    "--extraction", str(self.work / "latents"), "--out", str(out)] + (["--resume"] if resume else [])
            job = self._launch("train", argv, env)
            self.event("train", running={"gpu": g, "pid": job["proc"].pid}, resume=resume,
                       resume_from=resumable_steps(out)[-1] if resume else None)
            rc = job["proc"].wait(); job["file"].close()
            if done.exists():
                break
            if self.admission_refused(self._segment(job)):
                self.event("train_admission_refused", gpu=g); self.sleep(self.a.poll_s); continue
            attempts += 1
            self.alert(f"training rc={rc} on gpu{g} (attempt {attempts}); will resume from last verified checkpoint")
            if attempts > self.a.max_retries:
                raise Blocked(f"training failed {attempts}x (last rc={rc}); see pipeline/logs/train.log")
            if not out.exists() or not (out / "run_manifest.json").exists():
                raise Blocked(f"training rc={rc} before writing run_manifest.json; inspect pipeline/logs/train.log")
            self.sleep(self.a.poll_s)
        self.event("train", complete=True)

    def step_signal(self):
        c = json.loads((self.work / "scratch_run/COMPLETE.json").read_text())
        sel = c["selection"]
        if c["steps"] != RECIPE["steps"] or sel["step"] != RECIPE["steps"] or not sel["archive"].get("verified"):
            raise Blocked("final checkpoint/archive not verified")
        doc = {"experiment": "atomic65_common_qam_20261029", "status": "TRAINED_ARCHIVED", "utc": now(),
               "final_step": sel["step"], "macro_val_mse": sel["macro"], "archive": sel["archive"],
               "checkpoint": os.path.relpath(self.work / "scratch_run/checkpoints" / f"step_{sel['step']:07d}", REPO),
               "manifest_sha256": sha256_path(self.work / "metadata/manifest.json"),
               "source_freeze_sha256": self.status.get("source_freeze_sha256"),
               "next": "actor export + evaluation are separate (main session)"}
        (self.pdir / "DONE.json").write_text(json.dumps(doc, indent=1) + "\n")
        self.event("done", final_step=sel["step"])
        self.alert("training complete and archived; DONE.json written")

    def run(self) -> int:
        try:
            self.freeze()
            self.event("start", gpu_pool=self.a.gpu_pool, max_gpus=self.a.max_gpus, num_shards=self.a.num_shards)
            self.step_stage()
            self.step_manifest()
            self.freeze(); self.step_extract()
            self.freeze(); self.step_train()
            self.step_signal()
            return 0
        except Blocked as e:
            self.event("blocked", error=str(e)); self.alert(f"BLOCKED: {e}")
            return 1
        except Exception as e:  # unexpected: same visibility
            self.event("blocked", error=repr(e)); self.alert(f"ERROR: {e!r}")
            return 2


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, default=WORK)
    ap.add_argument("--staging", type=Path, default=STAGING_ROOT)
    ap.add_argument("--gpu-pool", type=int, nargs="+", required=True,
                    help="kanu GPUs this run may use when they have no compute process")
    ap.add_argument("--max-gpus", type=int, default=3, help="concurrent GPUs (kanu rule ≤3)")
    ap.add_argument("--num-shards", type=int, default=3, help="extraction shard split (3 = existing run)")
    ap.add_argument("--session", required=True, help="harness/lease session prefix")
    ap.add_argument("--poll-s", type=float, default=300)
    ap.add_argument("--stage-timeout-h", type=float, default=24)
    ap.add_argument("--queue-timeout-h", type=float, default=72, help="max wait for free GPUs per step")
    ap.add_argument("--max-retries", type=int, default=2, help="real (non-admission) failures per shard/train")
    ap.add_argument("--reserve-gib", type=float, default=20)
    ap.add_argument("--refreeze", action="store_true")
    a = ap.parse_args(argv)
    if not 1 <= a.max_gpus <= 3 or len(set(a.gpu_pool)) != len(a.gpu_pool):
        ap.error("kanu rule: at most 3 concurrent GPUs; pool entries distinct")
    return a


if __name__ == "__main__":
    sys.exit(Pipeline(parse()).run())
