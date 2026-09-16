#!/usr/bin/env python3
"""Ship completed RL2 episode metadata and activation sidecars to Seungjun.

The remote compute helper is deliberately the only transport interface.  A local
file is removed only after the remote size and SHA-256 agree with the bytes that
were observed locally, and a durable receipt has been written.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any


REPO = Path(__file__).resolve().parents[3]
HELPER = REPO / "scripts" / "utils" / "remote_compute.sh"
_SAFE_REL = re.compile(r"^[A-Za-z0-9._/-]+$")


class ShipError(ValueError):
    """A file is not safe or complete enough to ship."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "sha256": sha256_file(path)}


def _relative_repo_path(path: Path) -> str:
    """Return a helper-safe path, rejecting symlinks and traversal."""
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(REPO)
    except ValueError as exc:
        raise ShipError(f"path is outside repository: {path}") from exc
    if path.is_symlink() or not path.is_file():
        raise ShipError(f"expected a regular non-symlink file: {path}")
    text = relative.as_posix()
    if not text or text.startswith("/") or ".." in Path(text).parts or not _SAFE_REL.fullmatch(text):
        raise ShipError(f"unsafe repository-relative path: {text!r}")
    return text


def _sidecar_path(root: Path, value: Any) -> Path:
    if not isinstance(value, dict):
        raise ShipError("activation_file must be an object")
    raw = value.get("path")
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute() or Path(raw).name != raw:
        raise ShipError("activation_file.path must be a relative basename")
    if raw in {".", ".."} or not _SAFE_REL.fullmatch(raw):
        raise ShipError("unsafe activation_file.path")
    path = root / raw
    if path.is_symlink() or not path.is_file():
        raise ShipError(f"activation sidecar is not a regular file: {path}")
    return path


def _run_helper(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(HELPER), *args],
        cwd=str(REPO),
        text=True,
        capture_output=True,
        check=False,
    )


def _push(path: Path) -> None:
    rel = _relative_repo_path(path)
    result = _run_helper("push-data", rel)
    if result.returncode:
        raise ShipError(f"push-data failed for {rel}: {result.stderr.strip()}")


def _verify_remote(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    rel = _relative_repo_path(path)
    # Paths have been restricted above before entering the helper's shell command.
    result = _run_helper("run", f"sha256sum -- '{rel}' && stat -c '%s' -- '{rel}'")
    if result.returncode:
        raise ShipError(f"remote verification failed for {rel}: {result.stderr.strip()}")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ShipError(f"remote verification returned malformed output for {rel}")
    remote_sha = lines[0].split()[0]
    try:
        remote_size = int(lines[-1])
    except ValueError as exc:
        raise ShipError(f"remote verification returned invalid size for {rel}") from exc
    if remote_sha != expected["sha256"] or remote_size != expected["size_bytes"]:
        raise ShipError(
            f"remote identity mismatch for {rel}: "
            f"got {remote_sha}/{remote_size}, expected {expected['sha256']}/{expected['size_bytes']}"
        )
    return {"path": rel, "sha256": remote_sha, "size_bytes": remote_size}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _unchanged(path: Path, expected: dict[str, Any]) -> bool:
    if path.is_symlink():
        return False
    try:
        current = identity(path)
    except (FileNotFoundError, OSError):
        return False
    return current["size_bytes"] == expected["size_bytes"] and current["sha256"] == expected["sha256"]


def _remove_if_unchanged(path: Path, expected: dict[str, Any]) -> bool:
    if not _unchanged(path, expected):
        return False
    path.unlink()
    return True


def _receipt_path(receipts: Path, episode: Path) -> Path:
    return receipts / episode.name


def _remote_identity() -> dict[str, str]:
    return {"user": os.environ.get("REMOTE_USER", "kimseungjun"),
            "host": os.environ.get("REMOTE_HOST", "166.104.146.37"),
            "port": os.environ.get("REMOTE_PORT", "11112"),
            "repo": os.environ.get("REMOTE_REPO", "~/workspace/temporal_vla")}


def _lane_file(entry: Any, lane: Path) -> Path:
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise ShipError("receipt contains invalid file identity")
    relative = Path(entry["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ShipError("receipt path traversal")
    path = REPO / relative
    if path.is_symlink() or path.parent.resolve() != lane.resolve():
        raise ShipError("receipt must name a non-symlink lane file")
    try:
        path.relative_to(lane.resolve())
    except ValueError as exc:
        raise ShipError("receipt file escapes its lane") from exc
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ShipError("receipt file is not a regular local file")
    return path


def _finish_receipt(receipt: Path, lane: Path) -> int:
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        files = payload["files"]
        remote = payload["verified_remote"]
        if payload["remote"] != _remote_identity() or not isinstance(files, list) or not isinstance(remote, list):
            return 0
        by_path = {entry["path"]: entry for entry in remote}
        episode_entries = [entry for entry in files if entry.get("role") == "episode"]
        if len(episode_entries) != 1 or payload.get("episode_sha256") != episode_entries[0].get("sha256") or payload.get("episode_size_bytes") != episode_entries[0].get("size_bytes"):
            return 0
        deleted = 0
        for entry in files:
            path = _lane_file(entry, lane)
            remote_entry = by_path.get(entry.get("path"))
            if not remote_entry or remote_entry.get("sha256") != entry.get("sha256") or remote_entry.get("size_bytes") != entry.get("size_bytes"):
                return 0
            if _remove_if_unchanged(path, entry):
                deleted += 1
        return deleted
    except (OSError, ValueError, KeyError, TypeError, ShipError):
        return 0


def ship_episode(root: Path, episode: Path, receipts: Path) -> bool:
    receipt = _receipt_path(receipts, episode)
    if receipt.exists():
        _finish_receipt(receipt, root)
        return True
    try:
        payload = json.loads(episode.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ShipError("episode JSON must be an object")
        if payload.get("schema_version") != 1 or payload.get("end_reason") not in ("success", "horizon", "terminated", "truncated"):
            raise ShipError("episode is incomplete or has an invalid schema/end_reason")
        if not isinstance(payload.get("chunks"), list) or not payload["chunks"]:
            raise ShipError("episode has no completed chunks")
        sidecar = _sidecar_path(root, payload["activation_file"]) if "activation_file" in payload else None
        episode_id = identity(episode)
        sidecar_id = identity(sidecar) if sidecar else None
        if sidecar:
            declared = payload["activation_file"]
            if declared.get("sha256") != sidecar_id["sha256"] or declared.get("size_bytes") != sidecar_id["size_bytes"]:
                raise ShipError("activation sidecar local identity does not match episode declaration")
        # Sidecar must arrive before its episode metadata.
        remote = []
        if sidecar:
            _push(sidecar)
            remote.append({**_verify_remote(sidecar, sidecar_id), "role": "activation"})
        _push(episode)
        remote.append({**_verify_remote(episode, episode_id), "role": "episode"})
        files = ([{**sidecar_id, "path": _relative_repo_path(sidecar), "role": "activation"}] if sidecar else [])
        files.append({**episode_id, "path": _relative_repo_path(episode), "role": "episode"})
        summary_fields = (
            "episode_id", "reset_id", "task_id", "env_seed", "policy_seed",
            "success", "end_reason", "schema_version", "feature_contract",
            "action_contract", "policy_checkpoint", "elapsed_seconds",
        )
        episode_summary = {key: payload[key] for key in summary_fields if key in payload}
        _atomic_json(receipt, {
            "schema_version": 1,
            "episode_path": _relative_repo_path(episode),
            "episode": episode_summary,
            "episode_sha256": episode_id["sha256"],
            "episode_size_bytes": episode_id["size_bytes"],
            "files": files,
            "verified_remote": remote,
            "remote": _remote_identity(),
        })
        for entry in files:
            _remove_if_unchanged(REPO / entry["path"], entry)
        print(f"[archive-shipper] verified and archived {episode}", flush=True)
        return True
    except (OSError, ShipError, json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"[archive-shipper] {episode.name}: {exc}", file=sys.stderr)
        return False


def run_once(root: Path) -> int:
    root = root.resolve()
    try:
        root.relative_to(REPO)
    except ValueError as exc:
        raise SystemExit(f"--root must be under repository: {root}") from exc
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".archive_shipper.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        count = 0
        for episode in sorted(root.rglob("episode_*.json")):
            if episode.is_symlink() or not episode.is_file() or ".archive_receipts" in episode.parts:
                continue
            lane = episode.parent
            receipts = lane / ".archive_receipts"
            receipts.mkdir(parents=True, exist_ok=True)
            if ship_episode(lane, episode, receipts):
                count += 1
        return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="episode directory under this repository")
    parser.add_argument("--once", action="store_true", help="scan once and exit")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between scans")
    args = parser.parse_args(argv)
    if args.interval <= 0:
        parser.error("--interval must be positive")
    while True:
        run_once(args.root)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
