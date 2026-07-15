#!/usr/bin/env python3
"""pq3 Gate A 테스트 (계획서 v9 §A Gate A — 배선 사고 방지 유닛/parity/회귀).

lerobot 컨테이너(CPU)에서 실행 — pytest 없이도 동작:
  docker compose run --rm lerobot python /temporal_vla/scripts/safe/groot_n15/robocasa/steer/pq3/tests/test_gate_a.py

항목:
  1 default parity     : action_token_mean assemble == 구 공식(마지막 horizon mean) bitwise
  2 full-capture=raw   : all_token_full assemble == raw stack (fp16 캐스트 동일)
  3 full→mean 일치     : full 의 action 세그먼트 mean ≈ 구 mode 출력 (allclose)
  4 β=0 무개입         : M=I hook 출력 == 미등록 출력 (CPU fp32 bitwise)
  5 per-step 스와핑    : call k 에 M_k, reset 재현, K 초과 fire RuntimeError
  6 3 생성자 전달      : single/multi/gated 모두 token_select·per-step M 전달
  7 구/신 pkl 게이트   : --require-capture-token-mode 구 pkl rc=4, 신 pkl 정상 fit
                         + per-step NPZ 키·loader 왕복 (S3 선행 검증)
  8 seed 분리 회귀     : eval 예약 seed 를 fit manifest 에 주입 → check rc=5,
                         freeze 동결 위반 → abort
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
PQ3 = HERE.parent
REPO = PQ3.parents[5]
SERVE = REPO / "scripts" / "serve"
for p in (str(REPO), str(SERVE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import safe_hooks  # noqa: E402
import steering_hooks  # noqa: E402

HORIZON = 16
T_FULL = 49  # 1 state + 32 future + 16 action (S1 실측 대상 가정)
D = 12
K = 4


# ── 더미 GR00T 스텁 ────────────────────────────────────────────────────────────
class _Block(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class _DiT(nn.Module):
    def __init__(self, n_blocks: int = 2):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block() for _ in range(n_blocks)])

    def forward(self, x):
        for b in self.transformer_blocks:
            x = b(x)
        return x


def _make_policy():
    head = SimpleNamespace(
        model=_DiT(),
        action_horizon=HORIZON,
        num_inference_timesteps=K,
        vlln=nn.Identity(),
        vl_self_attention=nn.Identity(),
    )
    gm = SimpleNamespace(action_head=head)
    return SimpleNamespace(_groot_model=gm), gm


def test_1_default_parity_and_2_full_raw_and_3_consistency():
    torch.manual_seed(0)
    policy, _gm = _make_policy()
    xs = [torch.randn(1, T_FULL, D) for _ in range(K)]

    def run_capture(token_pool):
        cap = safe_hooks.SafeFeatureCapture(
            policy, "groot", groot_dit_layers=[0, 1], groot_dit_token_pool=token_pool
        )
        with cap:
            for x in xs:
                policy._groot_model.action_head.model(x)
        return cap

    cap_mean = run_capture("action_token_mean")
    got_mean = cap_mean.assemble_blocks()
    # 구 공식 재현 (변경 전 assemble_blocks 그대로): stack[:,0,-H:,:].mean(1) → [K,D] → [L,K,D]
    expect_layers = []
    for layer in (0, 1):
        stack = torch.stack(cap_mean.block_bufs[layer], dim=0)
        expect_layers.append(stack[:, 0, -HORIZON:, :].mean(dim=1))
    expect_mean = torch.stack(expect_layers, dim=0).numpy().astype(np.float16)
    assert got_mean.shape == (2, K, D), got_mean.shape
    assert np.array_equal(got_mean, expect_mean), "1 default parity 실패 (bitwise 불일치)"

    cap_full = run_capture("all_token_full")
    got_full = cap_full.assemble_blocks()
    expect_full = torch.stack(
        [torch.stack(cap_full.block_bufs[layer], dim=0)[:, 0, :, :] for layer in (0, 1)],
        dim=0,
    ).numpy().astype(np.float16)
    assert got_full.shape == (2, K, T_FULL, D), got_full.shape
    assert np.array_equal(got_full, expect_full), "2 full-capture=raw 실패"

    # 3: full 의 마지막 HORIZON 토큰 mean(fit 시점 연산) ≈ 구 mode (fp16 반올림 허용)
    re_mean = got_full[:, :, -HORIZON:, :].astype(np.float32).mean(axis=2)
    assert np.allclose(re_mean, got_mean.astype(np.float32), atol=2e-3), "3 full→mean 불일치"
    print("[gate-a] 1 default parity / 2 full=raw / 3 full→mean OK")


def test_4_beta0_bitwise():
    torch.manual_seed(1)
    _policy, gm = _make_policy()
    x = torch.randn(1, T_FULL, D)
    y_ref = gm.action_head.model(x).clone()
    hook = steering_hooks.ConceptorSteering(
        gm, np.eye(D), pathway="dit", layer=0, token_select="all"
    ).register()
    try:
        y_hook = gm.action_head.model(x)
    finally:
        hook.unregister()
    assert torch.equal(y_ref, y_hook), "4 β=0(M=I) bitwise 불일치"
    y_after = gm.action_head.model(x)
    assert torch.equal(y_ref, y_after), "4 unregister 후 원복 실패"
    print("[gate-a] 4 β=0 bitwise OK")


def test_5_per_step_swap():
    torch.manual_seed(2)
    _policy, gm = _make_policy()
    scales = [2.0, 3.0, 4.0, 5.0]
    m_seq = [np.eye(D) * s for s in scales]
    hook = steering_hooks.ConceptorSteering(
        gm, m_seq, pathway="dit", layer=0, token_select="all"
    ).register()
    x = torch.randn(1, T_FULL, D)
    try:
        for k, s in enumerate(scales):
            y = gm.action_head.model(x)
            assert torch.allclose(y, x * s, atol=1e-5), f"5 step{k} M_k 미적용"
        try:
            gm.action_head.model(x)
            raise AssertionError("5 K 초과 fire 에 RuntimeError 미발생")
        except RuntimeError:
            pass
        hook.reset_step_counter()
        y0 = gm.action_head.model(x)
        assert torch.allclose(y0, x * scales[0], atol=1e-5), "5 reset 후 step0 재현 실패"
        # set_matrices 스위칭(gated /steering_phase 경로) 후 카운터 리셋 확인
        hook.set_matrices([np.eye(D) * 7.0] * K)
        y7 = gm.action_head.model(x)
        assert torch.allclose(y7, x * 7.0, atol=1e-5), "5 set_matrices 스위칭 실패"
        # vl pathway 에 per-step 리스트는 배선 오류 → ValueError
        try:
            steering_hooks.ConceptorSteering(gm, m_seq, pathway="vl")
            raise AssertionError("5 vl+per-step 에 ValueError 미발생")
        except ValueError:
            pass
    finally:
        hook.unregister()
    print("[gate-a] 5 per-step 스와핑 OK")


def test_6_three_constructors_pass_token_select(tmp: Path):
    # scripts/serve/lerobot.py 를 별칭 모듈로 로드 — `import lerobot` 은 실제 lerobot
    # 패키지를 가리므로 사용 금지 (이름 충돌).
    import importlib.util

    spec = importlib.util.spec_from_file_location("serve_lerobot", SERVE / "lerobot.py")
    lr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lr)

    calls = []

    class SpyHook:
        def __init__(self, gm, M, **kw):
            calls.append({"M": M, **kw})

        def register(self):
            return self

        def unregister(self):
            pass

        def set_matrices(self, M):
            pass

        def reset_step_counter(self):
            pass

    orig = (
        steering_hooks.ConceptorSteering,
        steering_hooks.load_steering_matrix,
        steering_hooks.load_steering_matrices_per_step,
    )
    steering_hooks.ConceptorSteering = SpyHook
    steering_hooks.load_steering_matrix = lambda *a, **k: np.eye(3)
    steering_hooks.load_steering_matrices_per_step = lambda *a, **k: [np.eye(3)] * K
    lr._policy_type = "groot"
    policy, _gm = _make_policy()

    def base_args(**kw):
        d = dict(
            steering_npz=None, steering_npz_dir=None, steering_layers=None,
            steering_phase_npz_base=None, steering_beta=0.1, steering_alpha=None,
            steering_key="C_steer", steering_pathway="dit", steering_layer=None,
            steering_token_select="all", steering_denoise="per_step",
        )
        d.update(kw)
        return SimpleNamespace(**d)

    try:
        # single
        calls.clear()
        lr._register_steering_if_requested(policy, base_args(steering_npz="x.npz"))
        assert calls and calls[0]["token_select"] == "all", f"6 single token_select 미전달: {calls}"
        assert isinstance(calls[0]["M"], list) and len(calls[0]["M"]) == K, "6 single per-step M 미전달"
        # multi
        d_multi = tmp / "multi" / "dit_L0"
        d_multi.mkdir(parents=True)
        (d_multi / "conceptors.npz").write_bytes(b"")
        calls.clear()
        lr._register_steering_if_requested(
            policy, base_args(steering_npz_dir=str(tmp / "multi"), steering_layers="0")
        )
        assert calls and calls[0]["token_select"] == "all", "6 multi token_select 미전달"
        assert isinstance(calls[0]["M"], list) and len(calls[0]["M"]) == K, "6 multi per-step M 미전달"
        # gated
        d_gated = tmp / "gated" / "pull" / "dit_L0"
        d_gated.mkdir(parents=True)
        (d_gated / "conceptors.npz").write_bytes(b"")
        calls.clear()
        lr._register_steering_if_requested(
            policy,
            base_args(steering_phase_npz_base=str(tmp / "gated"), steering_layers="0"),
        )
        assert calls and calls[0]["token_select"] == "all", "6 gated token_select 미전달"
        idm = calls[0]["M"]  # gated 는 identity 로 초기화 — per-step 이면 [I]×K
        assert isinstance(idm, list) and len(idm) == K, "6 gated per-step identity 미통일"
    finally:
        (
            steering_hooks.ConceptorSteering,
            steering_hooks.load_steering_matrix,
            steering_hooks.load_steering_matrices_per_step,
        ) = orig
        lr._steering = []
        lr._gated_registry = {}
    print("[gate-a] 6 three-constructor token_select/per-step 전달 OK")


def _write_pkl(path: Path, *, full: bool, succ: int, n: int = 3):
    if full:
        recs = [np.random.randn(2, K, T_FULL, 8).astype(np.float16) for _ in range(n)]
    else:
        recs = [np.random.randn(2, K, 8).astype(np.float16) for _ in range(n)]
    d = {
        "hidden_states": recs,
        "feature_phases": ["reach-to-handle"] * n,
        "episode_success": succ,
        "capture_layers": [0, 1],
        "model_action_horizon": HORIZON,
    }
    if full:
        d["capture_token_mode"] = "all_token_full"
    path.write_bytes(pickle.dumps(d))


def test_7_pkl_gate_and_per_step_npz(tmp: Path):
    np.random.seed(3)
    fit = REPO / "scripts/safe/groot_n15/robocasa/steer/fit_phase_conceptor_n15.py"
    new_dir = tmp / "new"; new_dir.mkdir()
    man_new = tmp / "man_new.tsv"
    rows = []
    for i in range(6):
        p = new_dir / f"ep{i}.pkl"
        _write_pkl(p, full=True, succ=int(i < 3))
        rows.append(f"{p}\t{int(i < 3)}\tseed{i}")
    man_new.write_text("\n".join(rows) + "\n")

    out_dir = tmp / "npz_out"
    r = subprocess.run(
        [sys.executable, str(fit), "--cell", "x/pq3_test", "--manifest", str(man_new),
         "--groups", "global", "--denoise", "per_step", "--alphas", "1,10",
         "--min-per-class", "3", "--layers", "0",
         "--require-capture-token-mode", "all_token_full", "--out-dir", str(out_dir)],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert r.returncode == 0, f"7 신 pkl fit 실패 rc={r.returncode}\n{r.stdout}\n{r.stderr}"
    npz_path = out_dir / "global" / "dit_L0" / "conceptors.npz"
    assert npz_path.exists(), "7 per-step NPZ 미생성"
    z = np.load(npz_path)
    for k in range(K):
        assert any(key.startswith(f"step{k}_alpha") and key.endswith("_C_steer") for key in z.files), \
            f"7 step{k} 키 없음: {z.files}"
    meta = json.loads((out_dir / "global" / "dit_L0" / "metadata.json").read_text())
    assert meta["denoise_mode"] == "per_step" and len(meta["selected_alpha_per_step"]) == K
    # loader 왕복 (serve 소비 계약)
    mats = steering_hooks.load_steering_matrices_per_step(npz_path, beta=0.1, num_steps=K)
    assert len(mats) == K and mats[0].shape == (8, 8), "7 per-step loader 왕복 실패"

    # 구 pkl → require 게이트 rc=4
    old_dir = tmp / "old"; old_dir.mkdir()
    man_old = tmp / "man_old.tsv"
    rows = []
    for i in range(6):
        p = old_dir / f"ep{i}.pkl"
        _write_pkl(p, full=False, succ=int(i < 3))
        rows.append(f"{p}\t{int(i < 3)}\tseed{i}")
    man_old.write_text("\n".join(rows) + "\n")
    r = subprocess.run(
        [sys.executable, str(fit), "--cell", "x/pq3_test", "--manifest", str(man_old),
         "--groups", "global", "--require-capture-token-mode", "all_token_full",
         "--out-dir", str(tmp / "npz_old")],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert r.returncode == 4, f"7 구 pkl 게이트 rc={r.returncode} (기대 4)\n{r.stderr}"
    print("[gate-a] 7 구/신 pkl 게이트 + per-step NPZ 왕복 OK")


def test_8_seed_separation_regression(tmp: Path):
    mk = PQ3 / "make_pq3_manifests.py"
    seeds_tsv = tmp / "seeds.tsv"
    header = "cell_index\tcell_id\ttask\tenv_name\tscenario_seed\tinstruction\tcanonical_instruction\tep_meta_path"
    lines = [header]
    for i in range(40):
        lines.append(f"8\topen_drawer_left\tOpenDrawer\tenvX\t{100000 + i}\tOpen the left drawer.\tOpen the left drawer.\t-")
    seeds_tsv.write_text("\n".join(lines) + "\n")
    mroot = tmp / "manifests"
    base = [sys.executable, str(mk)]
    r = subprocess.run(base + ["plan", "--seeds-tsv", str(seeds_tsv), "--cell-id", "pq3_drawer_left",
                               "--tsv-cell-index", "8", "--out-dir", str(mroot), "--n-plan", "40"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    coll = tmp / "collected"; coll.mkdir()
    for i in range(15):
        (coll / f"task8--ep{i}--succ{int(i % 2 == 0)}.csv").write_text("x")
    r = subprocess.run(base + ["freeze", "--seeds-tsv", str(seeds_tsv), "--cell-id", "pq3_drawer_left",
                               "--tsv-cell-index", "8", "--collected-dir", str(coll),
                               "--out-dir", str(mroot)], capture_output=True, text=True)
    assert r.returncode == 0, f"8 freeze 실패\n{r.stdout}\n{r.stderr}"
    cell_dir = mroot / "pq3_drawer_left"
    reserved = json.loads((cell_dir / "eval_reserved.json").read_text())
    assert not set(reserved["unseen_seeds"]) & set(reserved["fit_used_seeds"]), "8 unseen∩fit ≠ ∅"
    fit_manifest = cell_dir / "fit_manifest.tsv"
    r = subprocess.run(base + ["check", "--fit-manifest", str(fit_manifest),
                               "--eval-reserved", str(cell_dir / "eval_reserved.json")],
                       capture_output=True, text=True)
    assert r.returncode == 0, "8 정상 fit manifest check 실패"
    # 회귀: eval 예약(unseen) seed 를 fit manifest 에 주입 → rc=5
    bad = tmp / "fit_bad.tsv"
    lines = fit_manifest.read_text().splitlines()
    parts = lines[1].split("\t"); parts[2] = str(reserved["unseen_seeds"][0])
    lines[1] = "\t".join(parts)
    bad.write_text("\n".join(lines) + "\n")
    r = subprocess.run(base + ["check", "--fit-manifest", str(bad),
                               "--eval-reserved", str(cell_dir / "eval_reserved.json")],
                       capture_output=True, text=True)
    assert r.returncode == 5, f"8 주입 회귀 rc={r.returncode} (기대 5)"
    # freeze 동결: 수집이 늘어 fit 집합이 바뀌면 재-freeze 는 abort
    (coll / "task8--ep15--succ1.csv").write_text("x")
    r = subprocess.run(base + ["freeze", "--seeds-tsv", str(seeds_tsv), "--cell-id", "pq3_drawer_left",
                               "--tsv-cell-index", "8", "--collected-dir", str(coll),
                               "--out-dir", str(mroot)], capture_output=True, text=True)
    assert r.returncode != 0, "8 동결 위반이 통과됨"
    print("[gate-a] 8 seed 분리 회귀 + 동결 OK")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="pq3_gate_a_") as td:
        tmp = Path(td)
        test_1_default_parity_and_2_full_raw_and_3_consistency()
        test_4_beta0_bitwise()
        test_5_per_step_swap()
        test_6_three_constructors_pass_token_select(tmp)
        test_7_pkl_gate_and_per_step_npz(tmp)
        test_8_seed_separation_regression(tmp)
    print("[gate-a] ALL PASS")


if __name__ == "__main__":
    main()
