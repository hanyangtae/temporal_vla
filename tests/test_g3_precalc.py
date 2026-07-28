"""g3_precalc.py 합성 자기검증 (실데이터 불필요).

사양 = `exp5-1_next_computations.txt` §2 의 판정 논리와 §7 "잔차 공간 밖의 방향을 재사용하지
말 것(LOSO 파기)" 를 합성으로 못 박는다.

  1. 1차원 심음  → 계산 B 사다리가 **2단계(r̂₁ 제거 후)에서 붕괴**
  2. 부분공간 심음 → 차원 판정이 **부분공간**으로 나옴
  3. LOSO 위반 조작 → fold 방향이 held-out scene 을 보지 않음 (보면 값이 달라짐)

★ 2번의 판정자가 사양 문면(사다리 3단계 잔존)이 아니라 `subspace_capture` 인 이유:
사양의 사다리는 리더(scene-평균 mean-diff)의 방향을 그대로 사영 제거하므로 잔차의
scene 평균이 **정확히 0** 이 된다 (선형사영: m − (m·r̂)r̂ = 0). 즉 데이터가 실제로 몇
차원이든 2단계에서 반드시 붕괴한다 — `test_ladder_collapse_is_arithmetically_forced` 가
그 사실 자체를 못 박는다. 실제 차원 판정은 held-out scene 방향의 top-k 포착률로 한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "scene_sae"))
sys.path.insert(0, str(REPO))

from g2_residual_read import within_dir                        # noqa: E402
from g3_precalc import (                                        # noqa: E402
    Z_KEEP,
    capture_verdict,
    deflation_ladder,
    fold_direction,
    ladder_verdict,
    pairwise_cos,
    residual_scene_mean_ratio,
    subspace_capture,
)


def synth_episodes(seed=0, n_scene=10, n_ep=12, D=24, n_out_dir=1, sig=1.4, scene_amp=6.0):
    """episode 단위 합성 (U, y, sc).

    scene 마다 큰 오프셋(= scene 암기 성분)을 주고, outcome 은 `n_out_dir` 개 방향 중
    scene 별로 하나를 골라 succ/fail 을 가른다. n_out_dir=1 → 1차원, >1 → 부분공간.
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(D, n_out_dir + 2)))
    outs = [Q[:, i] for i in range(n_out_dir)]
    U, y, sc = [], [], []
    for s in range(n_scene):
        cen = scene_amp * rng.normal(size=D)
        d = outs[s % n_out_dir]
        for j in range(n_ep):
            succ = int(j % 2 == 0)
            U.append(cen + rng.normal(scale=0.5, size=D) + (-sig if succ else sig) * d)
            y.append(succ)
            sc.append(1000 + s)
    return np.asarray(U), np.asarray(y), np.asarray(sc)


def _zs(steps):
    return [s.get("null_z") for s in steps]


def test_one_dim_signal_collapses_at_step2():
    U, y, sc = synth_episodes(seed=1, n_out_dir=1)
    steps, _ = deflation_ladder(U, y, sc, n_steps=3, n_perm=120,
                               rng=np.random.default_rng(0))
    z = _zs(steps)
    assert z[0] is not None and z[0] > Z_KEEP, f"기준선부터 신호가 없다: {z}"
    assert steps[0]["auroc"] > 0.9
    assert z[1] is None or z[1] <= Z_KEEP, f"1차원 심음인데 r̂₁ 제거 후 잔존: {z}"
    assert ladder_verdict(steps)[0] == "one_dimensional"


def test_ladder_collapse_is_arithmetically_forced():
    """사양 사다리의 2단계 붕괴는 데이터 차원과 무관하게 강제된다 (퇴화 증명).

    1차원 심음과 3차원 심음 **양쪽에서** 잔차 scene 평균 비율이 ~0 이어야 한다.
    이 테스트가 깨지면 위 판정 주석(그리고 보고서의 경고)이 틀린 것이다.
    """
    for nd in (1, 3):
        U, y, sc = synth_episodes(seed=2, n_scene=12, n_out_dir=nd, sig=1.8)
        w = within_dir(U, y, sc)
        w = w / np.linalg.norm(w)
        U2 = U - np.outer(U @ w, w)
        assert residual_scene_mean_ratio(U, y, sc) > 0.05
        assert residual_scene_mean_ratio(U2, y, sc) < 1e-10, f"n_out_dir={nd} 에서 비퇴화"


def test_subspace_signal_detected_by_capture():
    """부분공간 심음 → held-out 방향 포착률이 k 를 늘릴수록 크게 오른다 (= 부분공간)."""
    U1, y1, s1 = synth_episodes(seed=1, n_scene=12, n_out_dir=1)
    U3, y3, s3 = synth_episodes(seed=2, n_scene=12, n_out_dir=3, sig=1.8)
    c1, c3 = subspace_capture(U1, y1, s1, 4), subspace_capture(U3, y3, s3, 4)
    assert c1["capture_mean"][1] > 0.7, f"1차원 심음인데 k=1 포착이 낮다: {c1['capture_mean']}"
    assert capture_verdict(c1)[0] == "one_dimensional_capture"
    assert c3["capture_mean"][1] < c1["capture_mean"][1]
    assert c3["capture_mean"][3] - c3["capture_mean"][1] > 0.10, \
        f"부분공간 심음인데 k 증분이 없다: {c3['capture_mean']}"
    assert capture_verdict(c3)[0] == "subspace_capture"
    assert c3["participation_ratio"] > c1["participation_ratio"]


def test_fold_direction_excludes_heldout_scene():
    """LOSO 계약 — fold 방향은 held-out scene 행을 절대 보지 않는다.

    scene s0 에만 고유 방향을 크게 심는다. 올바른 fold 방향은 그 성분이 우연수준
    (등방 D 차원에서 |cos| ~ 1/√D) 이어야 하고, held-out 을 포함한 '조작(전역)' 방향은
    뚜렷한 성분을 갖는다 → 둘이 달라야 한다.
    """
    D = 200
    U, y, sc = synth_episodes(seed=3, n_scene=8, n_out_dir=1, D=D)
    rng = np.random.default_rng(7)
    uniq = rng.normal(size=D)
    uniq /= np.linalg.norm(uniq)
    s0 = int(np.unique(sc)[0])
    U = U.copy()
    U[(sc == s0) & (y == 1)] += 40.0 * uniq             # s0 안에서만 succ/fail 을 가르는 축

    w_fold = fold_direction(U, y, sc, s0)               # 정상 (train scene 만)
    w_cheat = within_dir(U, y, sc)                      # 조작: held-out 포함 전역
    w_cheat = w_cheat / np.linalg.norm(w_cheat)

    chance = 3.0 / np.sqrt(D)                           # 등방 우연 |cos| 의 ~3σ
    assert abs(float(w_fold @ uniq)) < chance, "held-out scene 고유축이 fold 방향에 샜다"
    assert abs(float(w_cheat @ uniq)) > 0.5, "조작 방향이 고유축을 못 잡았다 (테스트 무력)"
    assert abs(float(w_fold @ w_cheat)) < 0.9


def test_pairwise_cos_shapes_and_range():
    U, y, sc = synth_episodes(seed=4, n_scene=5, n_out_dir=1)
    st = pairwise_cos(U, y, sc)
    assert st["n_scene_mixed"] == 5
    assert st["n_pairs"] == 10
    assert -1.0 <= st["min"] <= st["mean"] <= st["max"] <= 1.0
    assert st["mean"] > 0.5, "공통 1차원 심음이면 scene 간 cos 이 높아야 한다"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
