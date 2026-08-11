"""verify_grid 결함 주입 테스트 — 주입한 결함을 각각 실제로 잡는지 확인.

검증기가 스스로 만든 데이터를 통과시키는 건 증명이 아니다(순환) — 여기서는 결함을
하나씩 심고 그 결함"만" 검출되는지를 본다 (selfcheck_inspect 와 같은 방법론).
"""

from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.collect.plan import CollectionPlan  # noqa: E402

VERIFY = REPO_ROOT / "scripts" / "collect" / "verify_grid.py"

LAYERS = [0, 2]
K = 4


def make_plan(tmp_path: Path) -> tuple[CollectionPlan, Path]:
    plan = CollectionPlan(
        name="t", model="groot", version="n15", ckpt="ck",
        capture_layers=LAYERS, denoise_k=K, token_mode="all_token_full",
        instructions={"OpenDrawer/left": [100010, 100011]},
        noise_seeds=[1300000],
    )
    plan_dir = tmp_path / "plan"
    plan.save(plan_dir)
    return plan, plan_dir


def write_cell(root: Path, plan: CollectionPlan, machine: str, s_idx: int,
               *, meta_over: dict | None = None, drop: tuple[str, ...] = (),
               shape=None) -> Path:
    instr = "OpenDrawer/left"
    d = root / plan.plan_id / machine / instr / f"s{s_idx}" / "n0" / "base"
    d.mkdir(parents=True, exist_ok=True)
    shape = shape or [len(LAYERS), K, 49, 8]
    payload = {"hidden_states": [np.zeros(shape, dtype=np.float16)],
               "record_shape": shape}
    blob = pickle.dumps(payload)
    (d / "rollout.pkl").write_bytes(blob)
    sha = hashlib.sha256(blob).hexdigest()
    meta = {
        "sig": sha[:16], "pkl_sha256": sha,
        "plan_id": plan.plan_id, "machine": machine, "ckpt": "ck",
        "grid_instruction": instr, "scene_idx": s_idx, "noise_idx": 0,
        "env_seed": [100010, 100011][s_idx], "inference_seed": 1300000,
        "success": 1,
        "capture_token_mode": "all_token_full",
        "feature_kind": "groot_n15_dit_block_residual_tokens",
        "feature_axes": ["layer", "denoising_step", "model_token", "feature_dim"],
        "record_shape": shape, "capture_layers": LAYERS,
    }
    meta.update(meta_over or {})
    (d / "meta.json").write_text(json.dumps(meta))
    (d / "traj.csv").write_text("a,b\n0.0,0.0\n")
    (d / "video.mp4").write_bytes(b"v")
    for name in drop:
        (d / name).unlink()
    return d


def run_verify(tmp_path: Path, plan_dir: Path, *extra: str):
    proc = subprocess.run(
        [sys.executable, str(VERIFY), "--grid-root", str(tmp_path / "grid"),
         "--plan-json", str(plan_dir), *extra],
        capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def test_clean_grid_passes_and_reports_missing(tmp_path):
    plan, plan_dir = make_plan(tmp_path)
    write_cell(tmp_path / "grid", plan, "kanu", 0)
    rc, out = run_verify(tmp_path, plan_dir)
    assert rc == 0, out                        # 결손은 기본적으로 오류 아님
    assert "결손 1 / 2" in out                  # s1 이 빠졌다고 보고
    # 완주 검수 모드에서는 결손이 오류
    rc2, out2 = run_verify(tmp_path, plan_dir, "--require-complete")
    assert rc2 == 1 and "결손" in out2


def test_done_list_covers_shipped_cell(tmp_path):
    plan, plan_dir = make_plan(tmp_path)
    write_cell(tmp_path / "grid", plan, "kanu", 0)
    dl = tmp_path / "shipped.txt"
    dl.write_text("OpenDrawer/left|s1|n0\n")   # s1 은 이미 배송됨
    rc, out = run_verify(tmp_path, plan_dir, "--done-list", str(dl), "--require-complete")
    assert rc == 0, out
    assert "결손 0 / 2" in out


def test_detects_injected_faults_individually(tmp_path):
    plan, plan_dir = make_plan(tmp_path)
    grid = tmp_path / "grid"

    # 결함 1: pkl 변조 (sha 불일치)
    d0 = write_cell(grid, plan, "kanu", 0)
    (d0 / "rollout.pkl").write_bytes(b"corrupted")
    # 결함 2: meta 좌표 오기 (scene_idx)
    write_cell(grid, plan, "kanu", 1, meta_over={"scene_idx": 9})
    rc, out = run_verify(tmp_path, plan_dir)
    assert rc == 1
    assert "sha256 불일치" in out
    assert "meta.scene_idx=9" in out


def test_detects_missing_file_and_pooled_tokens(tmp_path):
    plan, plan_dir = make_plan(tmp_path)
    grid = tmp_path / "grid"
    write_cell(grid, plan, "kanu", 0, drop=("video.mp4",))
    write_cell(grid, plan, "kanu", 1, shape=[len(LAYERS), K, 1, 8])  # 토큰축 pooled
    rc, out = run_verify(tmp_path, plan_dir)
    assert rc == 1
    assert "video.mp4 없음" in out
    assert "토큰축 1" in out


def test_detects_duplicate_machines_and_seed_mismatch(tmp_path):
    plan, plan_dir = make_plan(tmp_path)
    grid = tmp_path / "grid"
    write_cell(grid, plan, "kanu", 0)
    write_cell(grid, plan, "srv50", 0)                       # 같은 셀 두 머신
    write_cell(grid, plan, "kanu", 1, meta_over={"env_seed": 999})  # 계획과 다른 seed
    rc, out = run_verify(tmp_path, plan_dir)
    assert rc == 1
    assert "중복 수집" in out
    assert "env_seed=999" in out


def test_deep_detects_nan(tmp_path):
    plan, plan_dir = make_plan(tmp_path)
    grid = tmp_path / "grid"
    d = grid / plan.plan_id / "kanu" / "OpenDrawer/left" / "s0" / "n0" / "base"
    write_cell(grid, plan, "kanu", 0)
    shape = [len(LAYERS), K, 49, 8]
    bad = np.zeros(shape, dtype=np.float16); bad[0, 0, 0, 0] = np.nan
    payload = {"hidden_states": [bad], "record_shape": shape}
    blob = pickle.dumps(payload)
    (d / "rollout.pkl").write_bytes(blob)
    sha = hashlib.sha256(blob).hexdigest()
    meta = json.loads((d / "meta.json").read_text())
    meta.update({"sig": sha[:16], "pkl_sha256": sha})
    (d / "meta.json").write_text(json.dumps(meta))
    rc, out = run_verify(tmp_path, plan_dir, "--deep", "1")
    assert rc == 1
    assert "NaN/Inf" in out
