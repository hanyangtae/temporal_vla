"""patching_hooks.PatchSteering 유닛 — 커서·창·고갈·fail-loud 규약 검증.

torch CPU 만 필요 (GR00T 불요 — dummy module 트리). lerobot 컨테이너에서:
  docker exec lerobot python -m pytest \
    /temporal_vla/.claude/worktrees/patching-ceiling/tests/test_patching_hooks.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/serve"))

from patching_hooks import PatchSteering, load_donor_npz  # noqa: E402

K, T, D, R = 4, 6, 5, 3
HORIZON = 2


class _Block(nn.Module):
    def forward(self, x):
        return x


class _TupleBlock(nn.Module):
    def forward(self, x):
        return (x, "aux")


def _make_groot(block_cls=_Block, n_blocks=3):
    model = nn.Module()
    model.transformer_blocks = nn.ModuleList([block_cls() for _ in range(n_blocks)])
    head = SimpleNamespace(model=model, action_horizon=HORIZON)
    return SimpleNamespace(action_head=head)


def _donor() -> np.ndarray:
    """donor[r, k] = 상수 (r*10 + k) 로 채운 [R,K,T,D] — 어느 (r,k) 가 대입됐는지 식별."""
    arr = np.zeros((R, K, T, D), dtype=np.float32)
    for r in range(R):
        for k in range(K):
            arr[r, k] = r * 10 + k
    return arr


def _run_records(groot, hook, n_records, *, layer=1, batch=1):
    """요청(=record) n개 시뮬레이션. serve 규약: 요청 진입마다 reset_step_counter,
    요청 내 K회 발화. 반환: outs[record][k] 텐서."""
    block = groot.action_head.model.transformer_blocks[layer]
    outs = []
    for _ in range(n_records):
        hook.reset_step_counter()
        per_k = []
        for _k in range(K):
            x = torch.zeros(batch, T, D)
            y = block(x)
            per_k.append(y[0] if isinstance(y, tuple) else y)
        outs.append(per_k)
    return outs


def _hook(groot, layer=1, token_select="all"):
    return PatchSteering(
        groot, layer=layer, expected_k=K, token_select=token_select
    ).register()


def test_window_cursor_and_values():
    groot = _make_groot()
    hook = _hook(groot)
    hook.arm(_donor(), start_record=2, tag="t")
    outs = _run_records(groot, hook, 5)
    # record 0,1: 창 이전 — 원본(0) 유지
    assert torch.all(outs[0][0] == 0) and torch.all(outs[1][K - 1] == 0)
    # record 2,3,4 → donor 0,1,2; 값 = r_donor*10 + k
    for rec, d_rec in [(2, 0), (3, 1), (4, 2)]:
        for k in range(K):
            assert torch.all(outs[rec][k] == d_rec * 10 + k), (rec, k)
    st = hook.status()
    assert st["fired_records"] == [2, 3, 4]
    assert st["fired_total"] == 3 * K
    assert st["exhausted_at"] == []


def test_donor_exhaustion_records_only():
    groot = _make_groot()
    hook = _hook(groot)
    hook.arm(_donor(), start_record=0, tag="t")
    outs = _run_records(groot, hook, R + 2)
    # donor 고갈(R=3) 이후 record 3,4 는 패치 없이 원본 — 합성/freeze 금지 규약
    assert torch.all(outs[R][0] == 0) and torch.all(outs[R + 1][K - 1] == 0)
    st = hook.status()
    assert st["fired_records"] == [0, 1, 2]
    assert st["exhausted_at"] == [R, R + 1]


def test_patch_len_window():
    groot = _make_groot()
    hook = _hook(groot)
    hook.arm(_donor(), start_record=1, patch_len=1, tag="t")
    outs = _run_records(groot, hook, 4)
    assert torch.all(outs[1][0] == 0 * 10 + 0)  # donor r=0
    assert torch.all(outs[2][0] == 0)  # 창 밖 — 원본
    assert hook.status()["fired_records"] == [1]


def test_donor_start_offset():
    groot = _make_groot()
    hook = _hook(groot)
    hook.arm(_donor(), start_record=0, donor_start=2, tag="t")
    outs = _run_records(groot, hook, 2)
    assert torch.all(outs[0][1] == 2 * 10 + 1)  # donor r=2 부터 재생
    assert hook.status()["exhausted_at"] == [1]  # r=3 없음


def test_reset_episode_keeps_arm():
    groot = _make_groot()
    hook = _hook(groot)
    hook.arm(_donor(), start_record=0, tag="t")
    _run_records(groot, hook, 2)
    hook.reset_episode()
    assert hook.armed and hook.status()["fired_total"] == 0
    outs = _run_records(groot, hook, 1)
    assert torch.all(outs[0][0] == 0 * 10 + 0)  # 커서가 0 부터 다시


def test_disarm_stops_patching():
    groot = _make_groot()
    hook = _hook(groot)
    hook.arm(_donor(), start_record=0, tag="t")
    hook.disarm()
    outs = _run_records(groot, hook, 1)
    assert torch.all(outs[0][0] == 0) and not hook.armed


def test_overfire_raises():
    groot = _make_groot()
    hook = _hook(groot)
    hook.arm(_donor(), start_record=0, tag="t")
    hook.reset_step_counter()
    block = groot.action_head.model.transformer_blocks[1]
    for _ in range(K):
        block(torch.zeros(1, T, D))
    with pytest.raises(RuntimeError, match="over-fire"):
        block(torch.zeros(1, T, D))


def test_token_shape_mismatch_raises():
    groot = _make_groot()
    hook = _hook(groot)
    hook.arm(_donor(), start_record=0, tag="t")
    hook.reset_step_counter()
    block = groot.action_head.model.transformer_blocks[1]
    with pytest.raises(RuntimeError, match="full-token donor"):
        block(torch.zeros(1, T + 1, D))


def test_tuple_output_preserved():
    groot = _make_groot(block_cls=_TupleBlock)
    hook = _hook(groot)
    hook.arm(_donor(), start_record=0, tag="t")
    hook.reset_step_counter()
    out = groot.action_head.model.transformer_blocks[1](torch.zeros(1, T, D))
    assert isinstance(out, tuple) and out[1] == "aux"
    assert torch.all(out[0] == 0 * 10 + 0)


def test_batch_broadcast():
    groot = _make_groot()
    hook = _hook(groot)
    hook.arm(_donor(), start_record=0, tag="t")
    outs = _run_records(groot, hook, 1, batch=2)
    assert len(outs) == 1
    assert outs[0][0].shape == (2, T, D)
    assert torch.all(outs[0][0][0] == outs[0][0][1])


def test_action_token_select():
    groot = _make_groot()
    hook = _hook(groot, token_select="action")
    hook.arm(_donor(), start_record=0, tag="t")
    outs = _run_records(groot, hook, 1)
    y = outs[0][0]
    assert torch.all(y[..., -HORIZON:, :] == 0)  # donor r=0,k=0 값이 0 이라 구분 안 됨 —
    # 마지막 horizon 밖 토큰이 원본(0)인지로만 검증하면 모호하므로 k=1 로 재확인
    y1 = outs[0][1]
    assert torch.all(y1[..., -HORIZON:, :] == 1)  # donor r=0,k=1
    assert torch.all(y1[..., : T - HORIZON, :] == 0)  # 나머지 토큰 원본 유지


def test_sham_self_donor_identity():
    """sham 규약: donor == 실제 출력이면 패치가 no-op 과 완전 일치해야 한다."""
    groot = _make_groot()
    hook = _hook(groot)
    sham = np.full((R, K, T, D), 7.0, dtype=np.float32)
    hook.arm(sham, start_record=0, tag="sham")
    hook.reset_step_counter()
    block = groot.action_head.model.transformer_blocks[1]
    y = block(torch.full((1, T, D), 7.0))
    assert torch.all(y == 7.0)


def test_arm_validation():
    groot = _make_groot()
    hook = _hook(groot)
    with pytest.raises(ValueError):
        hook.arm(_donor()[:, :2], start_record=0)  # K 불일치
    with pytest.raises(ValueError):
        hook.arm(_donor(), start_record=-1)
    with pytest.raises(ValueError):
        hook.arm(_donor(), start_record=0, donor_start=R)


def test_load_donor_npz_roundtrip(tmp_path):
    p = tmp_path / "donor.npz"
    meta = {"cell": "c", "episode_idx": 61, "scenario_seed": 300033,
            "inference_seed": 61000, "n_records": R}
    np.savez(
        p,
        L1=_donor().astype(np.float16),
        meta_json=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
    )
    arrays, m, sha12 = load_donor_npz(p, [1], expected_k=K)
    assert arrays[1].shape == (R, K, T, D) and m["episode_idx"] == 61 and len(sha12) == 12
    with pytest.raises(ValueError, match="num_inference_timesteps"):
        load_donor_npz(p, [1], expected_k=K + 1)
    with pytest.raises(ValueError, match="없음"):
        load_donor_npz(p, [2], expected_k=K)
