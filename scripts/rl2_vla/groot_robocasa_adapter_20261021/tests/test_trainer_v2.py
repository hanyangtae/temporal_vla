"""train_qam_baseline v2: 이식 가능한 경로 기록·tmp checkpoint 거부(삭제 없음)·disk preflight."""
import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO_ROOT
from contract import ContractError

import train_qam_baseline as tqb


def _strings(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)
    elif isinstance(obj, str):
        yield obj


def _assert_no_abs(obj, *forbidden):
    for s in _strings(obj):
        assert not s.startswith("/"), s
        for f in forbidden:
            assert str(f) not in s, (f, s)


# ---------------- 경로 참조 ----------------

def test_path_ref_aliases_and_hash(tmp_path):
    from scripts.path_setup import DATA_ROOT

    f = tmp_path / "split.json"
    f.write_text("{}")
    roots = {"repo": REPO_ROOT, "datasets_root": Path(DATA_ROOT), "run": tmp_path}
    r = tqb.path_ref(REPO_ROOT / tqb.AUX_SPLIT_V2_REL, roots)
    assert r == {"root": "repo", "rel": tqb.AUX_SPLIT_V2_REL}
    r = tqb.path_ref(f, roots, hash_file=True)
    assert r["root"] == "run" and r["rel"] == "split.json" and len(r["sha256"]) == 64
    r = tqb.path_ref("/nonexistent_root_xyz/abc/ext", roots)
    assert r == {"root": "unresolved", "name": "ext"}


def test_portable_args_have_no_absolute_paths(tmp_path):
    from scripts.path_setup import DATA_ROOT

    a = tqb.build_parser().parse_args(["--extraction", str(REPO_ROOT / "outputs/rl2_vla/x"),
                                       "--out", str(tmp_path / "run"), "--data-root", str(DATA_ROOT)])
    roots = {"repo": REPO_ROOT, "datasets_root": Path(DATA_ROOT), "run": a.out}
    args = tqb.portable_args(a, roots)
    _assert_no_abs(args, REPO_ROOT, DATA_ROOT, tmp_path)
    assert args["extraction"] == {"root": "repo", "rel": "outputs/rl2_vla/x"}
    assert args["data_root"] == {"root": "datasets_root", "is_path_setup_default": True}
    assert args["split"]["rel"] == tqb.AUX_SPLIT_V2_REL and len(args["split"]["sha256"]) == 64
    assert args["steps"] == 500_000 and args["batch_size"] == 256 and args["eval_every"] == 50_000
    json.dumps(args)  # 직렬화 가능 (Path 객체 없음)


def test_scrub_paths():
    roots = {"repo": Path("/a/b"), "run": Path("/a/b/outputs/run")}
    assert tqb.scrub_paths("x /a/b/outputs/run/checkpoints y /a/b/z", roots) == \
        "x <run>/checkpoints y <repo>/z"


def test_defaults_unchanged():
    a = tqb.build_parser().parse_args(["--extraction", "e", "--out", "o"])
    assert (a.steps, a.batch_size, a.eval_every, a.profile) == (500_000, 256, 50_000, "production")
    assert tqb.eval_steps(a.steps, a.eval_every) == [i * 50_000 for i in range(11)]
    assert tqb.VAL_METRIC == "val/action_mse_exec_prefix"


# ---------------- tmp checkpoint ----------------

class _FakeTrainer:
    def __init__(self, step):
        self.steps_done = step

    def save(self, d, *, feature_signature):
        Path(d, f"params_{self.steps_done}.pkl").write_bytes(b"x" * 10)
        Path(d, f"flags_{self.steps_done}.json").write_text("{}")


def test_save_checkpoint_rejects_existing_tmp_without_deleting(tmp_path):
    import os

    tmp = tmp_path / f".tmp_step_{3:07d}_{os.getpid()}"
    tmp.mkdir()
    (tmp / "keep.bin").write_bytes(b"precious")
    with pytest.raises(ContractError, match="tmp checkpoint"):
        tqb.save_checkpoint(_FakeTrainer(3), tmp_path, feature_signature="s")
    assert (tmp / "keep.bin").read_bytes() == b"precious"     # 지우지 않음
    assert not (tmp_path / "step_0000003").exists()


def test_save_checkpoint_ok_and_no_overwrite(tmp_path):
    ck = tqb.save_checkpoint(_FakeTrainer(5), tmp_path, feature_signature="s")
    assert ck["dir"] == "step_0000005" and ck["bytes"] == 12
    assert not any(p.name.startswith(".tmp") for p in tmp_path.iterdir())
    with pytest.raises(ContractError, match="이미 존재"):
        tqb.save_checkpoint(_FakeTrainer(5), tmp_path, feature_signature="s")


# ---------------- disk preflight ----------------

def test_disk_preflight_math(tmp_path, monkeypatch):
    G = 1024 ** 3
    monkeypatch.setattr(tqb, "_disk_free", lambda p: 10 * G)
    rec = tqb.disk_preflight(tmp_path, ckpt_bytes=G // 2, remaining=8)      # 10 - 4 = 6 GiB → OK
    assert rec["free_after_projection_bytes"] == 6 * G
    with pytest.raises(tqb.DiskBudget):
        tqb.disk_preflight(tmp_path, ckpt_bytes=G // 2 + 1, remaining=8)  # 6GiB 미만
    with pytest.raises(tqb.DiskBudget):
        tqb.disk_preflight(tmp_path, ckpt_bytes=1, remaining=0, min_free=11 * G)
    assert tqb.MIN_FREE_BYTES >= 6 * 10 ** 9


# ---------------- CLI (synthetic extraction, 실제 split ID) ----------------

def _smoke_argv(ext, out):
    return ["--extraction", str(ext), "--out", str(out), "--split", str(tqb.REPO_ROOT / tqb.AUX_SPLIT_V2_REL),
            "--steps", "4", "--eval-every", "2", "--log-every", "1", "--profile", "smoke_reduced_network",
            "--batch-size", "8", "--allow-subset-smoke", "--skip-source-verify"]


@pytest.fixture
def fake_ext(tmp_path, cached_checkpoint_dir):
    from scripts.path_setup import DATA_ROOT
    from contract import load_checkpoint_contract
    from data_contract import OPENDRAWER_PACKAGE_REL
    import test_baseline_pipeline as tbp

    if not (DATA_ROOT / OPENDRAWER_PACKAGE_REL).is_dir():
        pytest.skip("로컬 package 없음")
    if not tbp.SPLIT_V2.is_file():
        pytest.skip("aux split v2 없음")
    split_doc = json.loads(tbp.SPLIT_V2.read_text())
    codec = load_checkpoint_contract(cached_checkpoint_dir).codec
    ext = tmp_path / "ext"
    tbp._fake_extraction(ext, split_doc, codec, picks=tbp._picks(split_doc))
    return ext


def test_cli_artifacts_portable_and_all_ckpts_kept(tmp_path, fake_ext, monkeypatch):
    from scripts.path_setup import DATA_ROOT

    monkeypatch.setattr(tqb, "_disk_free", lambda p: 100 * 1024 ** 3)
    out = tmp_path / "run"
    assert tqb.main(_smoke_argv(fake_ext, out)) == 0
    rm = json.loads((out / "run_manifest.json").read_text())
    final = json.loads((out / "selection_final.json").read_text())
    _assert_no_abs(rm, REPO_ROOT, DATA_ROOT, tmp_path)
    _assert_no_abs(final, REPO_ROOT, DATA_ROOT, tmp_path)
    for line in (out / "progress.jsonl").read_text().splitlines():
        _assert_no_abs(json.loads(line), REPO_ROOT, DATA_ROOT, tmp_path)
    assert rm["args"]["out"] == {"root": "run", "rel": "."}
    assert rm["args"]["extraction"] == {"root": "unresolved", "name": "ext"}   # tmp 는 어느 root 밖
    assert sorted(p.name for p in (out / "checkpoints").iterdir()) == \
        ["step_0000000", "step_0000002", "step_0000004"]
    ev = [json.loads(l) for l in (out / "progress.jsonl").read_text().splitlines()]
    disk = [e for e in ev if e["event"] == "disk"]
    assert [d["remaining_checkpoints"] for d in disk] == [2, 1]   # 마지막 저장 뒤 검사 없음
    assert ev[-1]["event"] == "done"


def test_cli_disk_budget_stops_after_first_ckpt_without_deleting(tmp_path, fake_ext, monkeypatch):
    G = 1024 ** 3
    # 시작 검사는 통과 (6GiB+α), 첫 checkpoint 뒤 남은 2 개 투영이 6GiB 를 깨도록.
    monkeypatch.setattr(tqb, "_disk_free", lambda p: 6 * G + 1000)
    out = tmp_path / "run"
    assert tqb.main(_smoke_argv(fake_ext, out)) == 2
    failed = json.loads((out / "FAILED.json").read_text())
    assert failed["type"] == "DiskBudget" and failed["step"] == 0
    _assert_no_abs(failed, REPO_ROOT, tmp_path)
    assert [p.name for p in (out / "checkpoints").iterdir()] == ["step_0000000"]   # 보존
    assert not (out / "selection_final.json").exists()


def test_cli_disk_budget_at_start(tmp_path, fake_ext, monkeypatch):
    monkeypatch.setattr(tqb, "_disk_free", lambda p: 5 * 1024 ** 3)
    out = tmp_path / "run"
    assert tqb.main(_smoke_argv(fake_ext, out)) == 2
    assert json.loads((out / "FAILED.json").read_text())["type"] == "DiskBudget"
    assert list((out / "checkpoints").iterdir()) == []
