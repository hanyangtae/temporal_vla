"""Stage stager-COMPLETE atomic65 packages from 승준 to the compute staging root.

Per task (only tasks whose stager receipt is COMPLETE):
  - already staged (staging receipt + package)       -> skip (optionally re-hash with --reverify)
  - package dir present without staging receipt      -> adopt only if hash == stager receipt, else stop
  - kanu cache copy with identical package hash      -> hardlink/copy locally (no network)
  - otherwise                                        -> `remote_compute.sh run "tar -cf - ..."` stream from 승준
Every copy lands in <staging>/.partial/<task>/, is hashed with the stager's manifest
rule, and is renamed into place only if the hash matches the stager receipt.
Nothing is overwritten or deleted; stale partials are moved aside.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shlex
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiment import (KANU_ROBOCASA, REPO, SEUNGJUN_ROOT, load_stage_rows, load_stager_receipt,
                        sha256_path, write_json_once)
from atomic65_stage import file_manifest   # same bytes as the stager's shell manifest (tested there)

WRAPPER = REPO / "scripts/utils/remote_compute.sh"
GIB = 1024 ** 3
MAX_JOBS = 2


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def manifest_sha(pkg: Path) -> str:
    return hashlib.sha256(file_manifest(pkg).encode()).hexdigest()


def fetch_seungjun(rel: str, dest_parent: Path, source_root: str = SEUNGJUN_ROOT, transport: str = "remote") -> None:
    """Stream <source_root>/<rel> (a .../lerobot dir) as tar into dest_parent/lerobot.
    transport=local reads source_root on this machine (tests)."""
    parent, name = rel.rsplit("/", 1)
    tar_cmd = f"tar -C {shlex.quote(source_root + '/' + parent)} -cf - {shlex.quote(name)}"
    argv = ["bash", str(WRAPPER), "run", tar_cmd] if transport == "remote" else ["bash", "-c", tar_cmd]
    src = subprocess.Popen(argv, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)
    dst = subprocess.run(["tar", "-xf", "-", "-C", str(dest_parent), "--no-same-owner"], stdin=src.stdout)
    src.stdout.close()
    rc_src = src.wait()
    if rc_src != 0 or dst.returncode != 0:
        raise RuntimeError(f"tar stream failed: source rc={rc_src} extract rc={dst.returncode}")


def link_or_copy_tree(src: Path, dst: Path) -> str:
    """Hardlink every file (same filesystem) else copy; returns the method used."""
    method = "hardlink"
    for p in sorted(src.rglob("*")):
        q = dst / p.relative_to(src)
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
            continue
        q.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(p, q)
        except OSError:
            shutil.copy2(p, q)
            method = "copy"
    return method


def stage_task(row: dict, receipt: dict, staging: Path, *, reserve: int, transport: str,
               source_root: str, kanu_root: Path, reverify: bool) -> dict:
    task, rel = row["task"], row["rel_path"]
    want = receipt["package_manifest_sha256"]
    dest = staging / "robocasa" / rel
    rec_path = staging / "receipts" / f"{task}.json"
    if rec_path.exists():
        r = json.loads(rec_path.read_text())
        if not dest.is_dir():
            raise RuntimeError(f"{task}: staging receipt without package")
        if r["package_manifest_sha256"] != want:
            raise RuntimeError(f"{task}: staging receipt hash != stager receipt (source changed?)")
        if reverify and manifest_sha(dest) != want:
            raise RuntimeError(f"{task}: staged package no longer matches its hash")
        return {"task": task, "status": "ALREADY_STAGED", "source": r["source"]}

    if dest.exists():   # present without our receipt: adopt only if identical
        got = manifest_sha(dest)
        if got != want:
            raise RuntimeError(f"{task}: existing {dest} differs from stager receipt; untouched")
        source = "preexisting_identical"
    else:
        free = shutil.disk_usage(staging).free
        need = int(receipt["package_bytes"]) + reserve
        if free < need:
            raise RuntimeError(f"{task}: INSUFFICIENT_SPACE free={free} need={need}")
        part = staging / ".partial" / task
        if part.exists():
            part.rename(part.with_name(f"{task}.stale.{dt.datetime.now().strftime('%Y%m%dT%H%M%S%f')}"))
        part.mkdir(parents=True)
        kanu = kanu_root / rel
        source = None
        if kanu.is_dir() and manifest_sha(kanu) == want:
            source = "kanu_cache_" + link_or_copy_tree(kanu, part / "lerobot")
        else:
            fetch_seungjun(rel, part, source_root, transport)
            source = "seungjun_tar_stream"
        got = manifest_sha(part / "lerobot")
        if got != want:
            part.rename(part.with_name(f"{task}.badhash.{dt.datetime.now().strftime('%Y%m%dT%H%M%S%f')}"))
            raise RuntimeError(f"{task}: staged copy hash {got[:16]} != stager {want[:16]} (moved aside)")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            raise RuntimeError(f"{task}: {dest} appeared concurrently")
        (part / "lerobot").rename(dest)
        part.rmdir()
    out = {"schema": "atomic65_compute_staging_v1", "task": task, "status": "STAGED", "source": source,
           "package_rel": "robocasa/" + rel, "package_manifest_sha256": want,
           "stager_archive_sha256": receipt["archive_sha256"],
           "stager_receipt_sha256": sha256_path(Path(row["_receipt_path"])),
           "episodes_total": receipt["episodes_total"], "package_bytes": receipt["package_bytes"],
           "staged_utc": now()}
    write_json_once(rec_path, out)
    return {"task": task, "status": "STAGED", "source": source}


class StageLocked(RuntimeError):
    """Another stage() holds <staging>/.stage.lock."""


def register_consumer(staging: Path, experiment: str, note: str) -> None:
    """Append-only consumer registry; cleanup-plan refuses while another experiment is listed.
    Other experiments reading this staging root should add themselves here."""
    p = staging / "CONSUMERS.json"
    cons = json.loads(p.read_text()) if p.exists() else []
    if not any(c.get("experiment") == experiment for c in cons):
        cons.append({"experiment": experiment, "note": note, "registered_utc": now()})
        tmp = p.with_name(p.name + ".tmp"); tmp.write_text(json.dumps(cons, indent=1) + "\n"); tmp.rename(p)


def stage(staging: Path, *, tasks: list[str] | None, jobs: int, reserve: int, transport: str = "remote",
          source_root: str = SEUNGJUN_ROOT, kanu_root: Path = KANU_ROBOCASA, reverify: bool = False,
          rows: list[dict] | None = None, receipts: dict | None = None, log: Path | None = None) -> dict:
    """Stage every task whose stager receipt is COMPLETE. Fail-fast: no new task after an error."""
    if jobs > MAX_JOBS:
        raise SystemExit(f"jobs {jobs} > {MAX_JOBS}")
    rows = rows if rows is not None else load_stage_rows()
    unknown = sorted(set(tasks or []) - {r["task"] for r in rows})
    if unknown:
        raise SystemExit(f"unknown task selector(s): {unknown}")
    staging.mkdir(parents=True, exist_ok=True)
    import fcntl   # one stager per staging root (two would fight over .partial/<task>)
    lock = open(staging / ".stage.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise StageLocked(f"{staging}/.stage.lock held by another stager")
    try:
        return _stage_locked(staging, rows=rows, tasks=tasks, jobs=jobs, reserve=reserve, transport=transport,
                             source_root=source_root, kanu_root=kanu_root, reverify=reverify,
                             receipts=receipts, log=log)
    finally:
        lock.close()


def _stage_locked(staging: Path, *, rows, tasks, jobs, reserve, transport, source_root, kanu_root, reverify,
                  receipts, log) -> dict:
    from experiment import STAGER_RECEIPTS
    (staging / "receipts").mkdir(exist_ok=True)
    register_consumer(staging, "atomic65_common_qam_20261029", "staged by common_qam/staging.py")
    ready, waiting = [], []
    for r in rows:
        if tasks and r["task"] not in tasks:
            continue
        rec = receipts[r["task"]] if receipts is not None else load_stager_receipt(r["task"])
        if rec is None:
            waiting.append(r["task"])
            continue
        r = dict(r, _receipt_path=str(STAGER_RECEIPTS / f"{r['task']}.json") if receipts is None
                 else r.get("_receipt_path"))
        ready.append((r, rec))
    log = log or staging / "logs" / f"stage_{dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({"utc": now(), "event": "start", "ready": [r["task"] for r, _ in ready],
                               "waiting_for_stager": waiting}) + "\n")
    results, failed, stop = {}, {}, False
    with ThreadPoolExecutor(max_workers=jobs) as ex, open(log, "a") as lf:
        pending = {}
        it = iter(ready)

        def submit_next():
            for r, rec in it:
                pending[ex.submit(stage_task, r, rec, staging, reserve=reserve, transport=transport,
                                  source_root=source_root, kanu_root=kanu_root, reverify=reverify)] = r["task"]
                return True
            return False

        for _ in range(jobs):
            submit_next()
        while pending:
            fut = next(as_completed(pending))
            t = pending.pop(fut)
            try:
                res = fut.result()
                results[t] = res
            except Exception as e:  # error propagation: record and stop scheduling
                failed[t] = repr(e)
                stop = True
                res = {"task": t, "status": "FAILED", "error": repr(e)}
            lf.write(json.dumps({"utc": now(), **res}) + "\n"); lf.flush()
            print(json.dumps(res), flush=True)
            if not stop:
                submit_next()
    summary = {"utc": now(), "staged_or_present": sorted(results), "failed": failed,
               "not_started": sorted(set(r["task"] for r, _ in ready) - set(results) - set(failed)),
               "waiting_for_stager": waiting, "log": str(log)}
    with open(log, "a") as lf:
        lf.write(json.dumps({"event": "summary", **summary}) + "\n")
    return summary
