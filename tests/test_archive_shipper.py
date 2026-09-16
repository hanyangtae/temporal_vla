import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/rl2_vla/stage2_cluster_reward/archive_shipper.py"
SPEC = importlib.util.spec_from_file_location("archive_shipper", SCRIPT)
shipper = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(shipper)


def _make_episode(monkeypatch, tmp_path, *, declared=None):
    root = tmp_path / "episodes"
    root.mkdir()
    sidecar = root / "activation_1.bin"
    sidecar.write_bytes(b"activation")
    episode = root / "episode_1.json"
    digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    metadata = {"schema_version": 1, "end_reason": "horizon", "chunks": [{"ok": True}], "activation_file": {"path": sidecar.name, "sha256": declared or digest, "size_bytes": sidecar.stat().st_size}, "n": 1}
    episode.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(shipper, "REPO", tmp_path)
    monkeypatch.setattr(shipper, "_relative_repo_path", lambda p: p.resolve().relative_to(tmp_path).as_posix())
    return root, episode, sidecar


def test_upload_verifies_both_files_then_deletes_and_receipts(monkeypatch, tmp_path):
    root, episode, sidecar = _make_episode(monkeypatch, tmp_path)
    calls = []

    def fake_helper(*args):
        calls.append(args)
        class Result:
            returncode = 0
            stderr = ""
            stdout = f"{shipper.sha256_file(sidecar)}  x\n{sidecar.stat().st_size}\n" if args[0] == "run" and "activation_1" in args[1] else f"{shipper.sha256_file(episode)}  x\n{episode.stat().st_size}\n"
        return Result()

    monkeypatch.setattr(shipper, "_run_helper", fake_helper)
    assert shipper.run_once(root) == 1
    assert not episode.exists() and not sidecar.exists()
    assert [c[0] for c in calls] == ["push-data", "run", "push-data", "run"]
    receipt = root / ".archive_receipts" / episode.name
    saved = json.loads(receipt.read_text())
    assert saved["episode"]["end_reason"] == "horizon"
    assert "chunks" not in saved["episode"]
    assert "activation_file" not in saved["episode"]
    assert saved["episode_sha256"] == saved["files"][-1]["sha256"]
    assert saved["episode_size_bytes"] == saved["files"][-1]["size_bytes"]
    assert {entry["role"] for entry in saved["verified_remote"]} == {"activation", "episode"}


def test_remote_mismatch_keeps_both_files_and_retry_can_succeed(monkeypatch, tmp_path):
    root, episode, sidecar = _make_episode(monkeypatch, tmp_path)
    bad = True

    def fake_helper(*args):
        class Result:
            returncode = 0
            stderr = ""
            target = sidecar if "activation_1" in args[1] else episode
            stdout = "0" * 64 + "  x\n999\n" if bad and args[0] == "run" else f"{shipper.sha256_file(target)}  x\n{target.stat().st_size}\n"
        return Result()

    monkeypatch.setattr(shipper, "_run_helper", fake_helper)
    assert shipper.run_once(root) == 0
    assert episode.exists() and sidecar.exists()
    bad = False
    assert shipper.run_once(root) == 1
    assert not episode.exists() and not sidecar.exists()


def test_path_traversal_is_rejected_without_transfer(monkeypatch, tmp_path):
    root, episode, sidecar = _make_episode(monkeypatch, tmp_path)
    payload = json.loads(episode.read_text())
    payload["activation_file"]["path"] = "../outside.bin"
    episode.write_text(json.dumps(payload))
    monkeypatch.setattr(shipper, "_run_helper", lambda *args: pytest.fail("must not transfer unsafe path"))
    assert shipper.run_once(root) == 0
    assert episode.exists() and sidecar.exists()


def test_receipt_recovers_after_interrupted_deletion(monkeypatch, tmp_path):
    root, episode, sidecar = _make_episode(monkeypatch, tmp_path)
    def fake_helper(*args):
        class Result:
            returncode = 0; stderr = ""
            target = sidecar if "activation_1" in args[1] else episode
            stdout = f"{shipper.sha256_file(target)}  x\n{target.stat().st_size}\n"
        return Result()
    monkeypatch.setattr(shipper, "_run_helper", fake_helper)
    original = shipper._remove_if_unchanged
    calls = [0]
    def interrupt(path, expected):
        calls[0] += 1
        if calls[0] == 1:
            path.unlink()
            raise OSError("simulated interruption")
        return original(path, expected)
    monkeypatch.setattr(shipper, "_remove_if_unchanged", interrupt)
    assert shipper.run_once(root) == 0
    assert not sidecar.exists() and episode.exists()
    monkeypatch.setattr(shipper, "_remove_if_unchanged", original)
    assert shipper.run_once(root) == 1
    assert not episode.exists()
