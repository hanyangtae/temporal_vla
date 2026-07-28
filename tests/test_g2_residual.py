"""scripts/scene_sae/g2_residual_read.py — 합성 데이터 단위테스트.

G2 는 "scene 성분을 제거하면 succ/fail read 가 남는가"를 판정한다. 판정이 신뢰되려면
스크립트가 **아는 정답**에서 아래 셋을 지켜야 한다:

  (i)   scene 방향만 심은 합성 (outcome 이 scene 축 **안**에 있음)
        → 제거 후 scene probe 붕괴 + outcome null 수준.
  (ii)  scene ⊥ outcome 합성
        → 제거 후 scene probe 붕괴 + outcome 잔존.
  (iii) LOSO 부분공간 재추정이 held-out scene 을 보지 않는다
        (스파이 검사 + 누수판(전체 fit)과의 값 차이).

전부 CPU·소형 합성으로 수초 안에 끝난다.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCENE_SAE = REPO / "scripts" / "scene_sae"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test",
                                                  SCENE_SAE / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


g2 = _load("g2_residual_read")


# ────────────────────────────────────────────────────────────── 합성 생성기
SCENE_RANK = 2                     # 합성 scene 부분공간 차원 (제거에 필요한 r)


def _make(n_scene=10, n_ep=10, D=16, outcome_in_scene=True, noise=0.3, seed=0,
          shared_plane=True):
    """episode 단위 벡터 (V, y, sc).

    scene 축 = 공유 평면 span{u1,u2} 위의 원형 배치 (scene 마다 다른 위치 → 선형 probe 로
    쉽게 읽히고, between-scatter rank 2 로 완전히 제거 가능).
    outcome_in_scene=True 면 succ/fail 차이가 그 평면 **안**(u1), False 면 평면과 직교(v).
    shared_plane=False 면 scene 마다 **독립** 방향 (LOSO 부분공간이 held-out scene 성분을
    원리적으로 못 지우는 경우 — fold-wise 제거 probe 가 이걸 잡아내야 한다).
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(D, 3)))
    u1, u2, v = Q[:, 0], Q[:, 1], Q[:, 2]
    ind, _ = np.linalg.qr(rng.normal(size=(D, n_scene + 3)))
    V, y, sc = [], [], []
    for s in range(n_scene):
        th = 2 * np.pi * s / n_scene
        centre = (8.0 * (np.cos(th) * u1 + np.sin(th) * u2) if shared_plane
                  else 8.0 * ind[:, 3 + s])
        for j in range(n_ep):
            succ = int(j % 2 == 0)
            x = centre + rng.normal(scale=noise, size=D)
            x = x + (1.0 if succ else -1.0) * (u1 if outcome_in_scene else v)
            V.append(x)
            y.append(succ)
            sc.append(100 + s)
    return np.asarray(V), np.asarray(y), np.asarray(sc)


def _ep_ids(sc):
    """행마다 고유 episode id (합성은 행 1개 = episode 1개)."""
    return np.arange(len(sc))


def _read(V, y, sc, r=SCENE_RANK, method="between", n_perm=100, seed=1):
    """LOSO fold 안에서 r 차원 scene 부분공간 제거 후 within-scene read."""
    rng = np.random.default_rng(seed)
    folds = g2.fold_projections(V, sc, g2.BASIS_FNS[method], [r])[r]
    return g2.read_with_null(V, y, sc, n_perm, rng, Vfolds=folds)


def _scene_probe_before_after(V, sc, r=SCENE_RANK, method="between", seed=0):
    """train/test 를 scene 안에서 반씩 갈라 scene(=scenario_seed) probe 전/후."""
    tr = np.zeros(len(sc), bool)
    for s in np.unique(sc):
        idx = np.where(sc == s)[0]
        tr[idx[: len(idx) // 2]] = True
    te = ~tr
    before = g2.scene_probe(V[tr], sc[tr], V[te], sc[te], seed=seed)
    Q = g2.BASIS_FNS[method](V[tr], sc[tr], r)          # 부분공간은 train 행만으로
    Vr = g2.project_out(V, Q)
    after = g2.scene_probe(Vr[tr], sc[tr], Vr[te], sc[te], seed=seed)
    return before, after


# ─────────────────────────────────────────────── (i) scene 축 안의 outcome
def test_scene_only_synthetic_collapses():
    V, y, sc = _make(outcome_in_scene=True)

    base = g2.read_with_null(V, y, sc, 100, np.random.default_rng(1))
    assert base["auroc"] > 0.9 and base["null_z"] > 3, "제거 전에는 읽혀야 한다"

    before, after = _scene_probe_before_after(V, sc)
    assert before["acc"] > 0.8, before
    assert after["acc"] < before["acc"], (before, after)
    assert g2.removal_ok(before, after) is True

    res = _read(V, y, sc)
    # 유한표본 부분공간 추정오차 탓에 정확히 0.5 로는 안 떨어진다 (평면 추정 tilt 가
    # outcome 성분을 조금 남긴다) — 판정 기준은 순열 null 대비 z 다.
    assert res["auroc"] < base["auroc"] - 0.25, (base, res)
    assert res["null_z"] < 3.0
    assert g2.verdict_of(res, True) == "separation_was_scene"


# ─────────────────────────────────────────── (ii) scene ⊥ outcome — 잔존
def test_orthogonal_outcome_survives():
    V, y, sc = _make(outcome_in_scene=False)

    before, after = _scene_probe_before_after(V, sc)
    assert g2.removal_ok(before, after) is True, (before, after)

    res = _read(V, y, sc)
    assert res["auroc"] > 0.9
    assert res["null_z"] > 3.0
    assert g2.verdict_of(res, True) == "outcome_separable_from_scene"


# ────────────────────────── (리뷰 #1) 제거 probe = fold 사영 + 같은 scene 라벨
def test_foldwise_probe_matches_read_projection():
    """공유 평면 scene → fold 사영이 held-out scene 성분까지 지운다 (제거 성공)."""
    V, _y, sc = _make(shared_plane=True)
    base = g2.foldwise_scene_probe(V, sc, _ep_ids(sc))               # 사영 없음 = 기준선
    after = g2.foldwise_scene_probe(V, sc, _ep_ids(sc), g2.BASIS_FNS["between"], SCENE_RANK)
    assert base["acc"] > 0.8 and base["projection"] == "none"
    assert after["projection"] == "fold" and after["n_folds"] == len(np.unique(sc))
    assert after["acc"] < base["acc"]
    assert g2.removal_ok(base, after) is True
    # fold pooled test 행은 서로 겹치지 않는다 → per_fold n 합 = pooled n
    assert sum(f["n_test"] for f in after["per_fold"]) == after["n_test"]


def test_foldwise_probe_catches_leak_that_split_train_probe_misses():
    """scene 마다 독립 방향이면 LOSO 사영은 held-out scene 을 못 지운다.

    전체(=held-out 포함) 행으로 Q 를 뽑는 split-train probe 는 "제거 성공"이라 하지만,
    read 가 실제로 쓰는 fold 사영에서는 held-out scene 이 그대로 읽혀야 한다 (리뷰 #1).
    """
    V, _y, sc = _make(n_scene=6, n_ep=8, D=24, shared_plane=False, seed=3)
    r = 6
    before, after = _scene_probe_before_after(V, sc, r=r)             # 참고치(누수형)
    assert g2.removal_ok(before, after) is True, (before, after)

    fb = g2.foldwise_scene_probe(V, sc, _ep_ids(sc))
    fa = g2.foldwise_scene_probe(V, sc, _ep_ids(sc), g2.BASIS_FNS["between"], r)
    assert fb["acc"] > 0.8
    assert g2.removal_ok(fb, fa) is False, (fb, fa)


def test_foldwise_probe_split_is_within_scene():
    """probe 분할은 scene 안 episode 단위 — scene-heldout meta split 에서도 성립해야 한다."""
    sc = np.repeat([1, 2, 3], 6)
    ep = np.arange(18)
    tr, te = g2.scene_stratified_holdout(sc, ep)
    assert (tr | te).all() and not (tr & te).any()
    for s in (1, 2, 3):                     # 모든 scene 이 train/test 양쪽에 있다
        assert (tr & (sc == s)).any() and (te & (sc == s)).any()
    # 같은 episode 행이 갈라지지 않는다
    ep2 = np.repeat(np.arange(9), 2)
    tr2, te2 = g2.scene_stratified_holdout(np.repeat([1, 2, 3], 6), ep2)
    for e in np.unique(ep2):
        assert len(np.unique(tr2[ep2 == e])) == 1


# ─────────────────────── (리뷰 #4) 제거 전 기준선에 분리가 없으면 판정 안 함
def test_no_baseline_separation_precondition():
    res = {"null_z": 5.0}
    assert g2.verdict_of(res, True, base_read={"null_z": 6.0}) == "outcome_separable_from_scene"
    for bad in ({"null_z": 3.0}, {"null_z": 1.2}, {"null_z": None}, {}):
        assert g2.verdict_of(res, True, base_read=bad) == "no_baseline_separation"
        assert g2.verdict_of(res, None, base_read=bad) == "no_baseline_separation"
    # base_read 미지정이면 기존 동작 그대로
    assert g2.verdict_of(res, True) == "outcome_separable_from_scene"


def test_removal_failure_is_flagged_undecidable():
    """r 이 모자라 scene 이 안 지워지면 outcome 이 남아도 '판정불가' 여야 한다."""
    V, y, sc = _make(outcome_in_scene=False)
    before, _after = _scene_probe_before_after(V, sc)
    weak = {"acc": before["acc"], "chance": before["chance"]}      # 하나도 안 깎인 경우
    assert g2.removal_ok(before, weak) is False
    res = _read(V, y, sc)
    assert g2.verdict_of(res, False) == "undecidable_removal_failed"
    assert g2.verdict_of(res, None) == "undecidable_scene_probe_baseline"


# ──────────────────────────────────── (iii) LOSO 부분공간 — held-out 미열람
def test_loso_basis_never_sees_heldout_scene():
    V, y, sc = _make()
    spy = []
    g2.fold_projections(V, sc, g2.BASIS_FNS["between"], [1, 2], spy=spy)
    assert len(spy) == len(np.unique(sc))
    for rec in spy:
        assert rec["heldout"] not in rec["train_scenes"]
    # 스파이가 본 train scene 합집합 = 전체 − held-out
    all_sc = {int(s) for s in np.unique(sc)}
    for rec in spy:
        assert set(rec["train_scenes"]) == all_sc - {rec["heldout"]}


@pytest.mark.parametrize("method", ["between", "logreg"])
def test_loso_basis_is_invariant_to_heldout_values(method):
    """값 조작 테스트 — held-out scene 의 **값**을 크게 흔들어도 그 fold 의 부분공간은 불변.

    scene s 의 행을 새 방향 w 로 크게 밀면,
      · fold(held-out=s) 의 basis 는 s 를 안 보므로 **바뀌면 안 된다**,
      · 반대로 전체 fit(누수판) basis 는 w 를 흡수해 **반드시 바뀐다** (조작이 유효했다는 증거).
    """
    V, y, sc = _make(n_scene=6, n_ep=6, D=20, seed=7)
    s0 = int(np.unique(sc)[0])
    w = np.zeros(V.shape[1])
    w[-1] = 1.0
    V2 = V.copy()
    V2[sc == s0] += 50.0 * w                              # ← held-out scene 값 조작

    r = SCENE_RANK
    f1 = g2.fold_projections(V, sc, g2.BASIS_FNS[method], [r])[r]
    f2 = g2.fold_projections(V2, sc, g2.BASIS_FNS[method], [r])[r]
    # fold s0 의 사영 결과는 s0 이 아닌 행에서 완전히 동일해야 한다 (basis 불변 증거)
    keep = sc != s0
    assert np.allclose(f1[s0][keep], f2[s0][keep], atol=1e-8)

    # 조작 유효성: 전체 fit(누수판) basis 는 조작에 반응해야 한다 (안 그러면 테스트가 공허)
    Q1 = g2.BASIS_FNS[method](V, sc, r)
    Q2 = g2.BASIS_FNS[method](V2, sc, r)
    proj1, proj2 = float(w @ Q1 @ Q1.T @ w), float(w @ Q2 @ Q2.T @ w)
    assert abs(proj2 - proj1) > 1e-3, (method, proj1, proj2)
    if method == "between":         # 평균 기반이라 큰 이동을 통째로 흡수한다
        assert proj2 > 0.5 > proj1
        # 누수판은 조작 방향까지 지우므로 fold basis 와 실제로 다른 사영을 만든다
        assert not np.allclose(g2.project_out(V2, Q2)[keep], f2[s0][keep], atol=1e-6)


def test_loso_identity_folds_match_plain():
    """Vfolds 훅이 원본 loso 계산을 바꾸지 않는다 (이식 정합성)."""
    V, y, sc = _make()
    a0, n0 = g2.loso(V, y, sc)
    a1, n1 = g2.loso(V, y, sc, Vfolds={int(s): V for s in np.unique(sc)})
    assert (a0, n0) == (a1, n1)


def test_mixed_scene_only_evaluated():
    """단일 라벨 scene 은 평가에서 빠진다 (혼재 scene 만) — 원본 loso 규약."""
    V, y, sc = _make(n_scene=4, n_ep=6)
    y = y.copy()
    y[sc == sc.min()] = 1                       # 한 scene 을 전부 succ 으로
    _a, n = g2.loso(V, y, sc)
    assert n == int((sc != sc.min()).sum())


# ──────────────────────────────────────────── record 집계 (per-token → record)
def test_record_aggregation_matches_manual():
    X, meta = g2.synth(seed=0, n_scene=3, n_ep_per_scene=2, n_rec=3, T=5, D=7)
    grp = g2.RowGrouping(meta, window=3)
    sums = g2.stream_group_sums(X, grp, {"raw": (lambda r: r)}, batch_rows=13, verbose=False)
    V, ep, rec = g2.record_vectors(sums["raw"], grp, g2.AGG_SEGS["future"])

    for e, r in [(0, 0), (2, 1), (5, 2)]:
        m = ((meta["episode_idx"] == e) & (meta["record_idx"] == r)
             & (meta["token_seg"] == 1))
        want = X[m].astype(np.float64).mean(0)
        got = V[(ep == e) & (rec == r)][0]
        assert np.allclose(got, want, atol=1e-6)

    # window 밖 record 는 들어오지 않는다
    assert rec.max() == 2
    # 'all' 집계는 전 토큰 평균
    Va, epa, reca = g2.record_vectors(sums["raw"], grp, g2.AGG_SEGS["all"])
    m = (meta["episode_idx"] == 1) & (meta["record_idx"] == 0)
    assert np.allclose(Va[(epa == 1) & (reca == 0)][0],
                       X[m].astype(np.float64).mean(0), atol=1e-6)


def test_chunking_is_invariant_to_batch_size():
    X, meta = g2.synth(seed=1, n_scene=2, n_ep_per_scene=2, n_rec=3, T=6, D=5)
    grp = g2.RowGrouping(meta, window=3)
    t = {"raw": (lambda r: r)}
    a = g2.stream_group_sums(X, grp, t, batch_rows=7, verbose=False)["raw"]
    b = g2.stream_group_sums(X, grp, t, batch_rows=10_000, verbose=False)["raw"]
    assert np.allclose(a, b)


# ───────────────────────────────────────────────────── SAE arm (잔차화 배선)
def _tiny_ckpt(tmp_path, D=8, m=16, k=4, seed=0, name="ckpt",
               split_col="split_scene", scene_axis=True, train_scenes=(1000, 1001, 1002, 1003)):
    """train_scene_sae.py 산출 형식과 같은 최소 ckpt (model.pt/config.json/stats.npz).

    기본값은 **scene-heldout 축** ckpt (리뷰 #2 의 기본 요구). split_col/scene_axis 를 바꾸면
    episode 축 ckpt 를 만들어 에러 경로를 테스트할 수 있다.
    """
    torch = pytest.importorskip("torch")
    train_mod = _load("train_scene_sae")
    from src.sae.models import build_model

    sae_cfg = train_mod.build_sae_cfg(D, m, k)
    torch.manual_seed(seed)
    model = build_model(sae_cfg)
    d = tmp_path / name
    d.mkdir()
    torch.save(model.state_dict(), d / "model.pt")
    (d / "config.json").write_text(json.dumps(
        {"input_dim": D, "m": m, "k": k, "sae_cfg": sae_cfg,
         "split_col": split_col, "split_axis_scene_heldout": bool(scene_axis),
         "train_scenes": list(train_scenes) if scene_axis else None,
         "train_episodes": []}))
    np.savez(d / "stats.npz", mean=np.zeros(D, np.float32), std=np.ones(D, np.float32))
    return d, D, m


def test_sae_arm_zeroing_changes_reconstruction(tmp_path):
    d, D, m = _tiny_ckpt(tmp_path)
    model, cfg, mu, sd, src = g2.load_sae_bundle(d, None)
    assert src.endswith("stats.npz") and cfg["m"] == m

    rng = np.random.default_rng(0)
    raw = rng.normal(size=(64, D)).astype(np.float32) * 3.0
    tf = g2.make_sae_transforms(model, mu, sd, {"full": np.zeros(0, np.int64),
                                                "drop": np.arange(4)}, "cpu", 32)
    full, drop = tf["full"](raw), tf["drop"](raw)
    assert full.shape == raw.shape == drop.shape
    assert not np.allclose(full, drop), "selective feature 를 0 으로 만들면 재구성이 달라져야 한다"
    # 제거 집합이 비면 full 재구성과 완전히 같아야 한다 (통제 arm 의 정의)
    tf2 = g2.make_sae_transforms(model, mu, sd, {"none": np.zeros(0, np.int64)}, "cpu", 32)
    assert np.allclose(tf2["none"](raw), full)


def test_selective_features_reads_probe_json(tmp_path):
    p = tmp_path / "probe.json"
    p.write_text(json.dumps({"selectivity": {
        "n_selective": 5,
        "top_features": [{"feature": 7, "selective": True, "gap": 0.9},
                         {"feature": 3, "selective": True, "gap": 0.8},
                         {"feature": 1, "selective": False, "gap": 0.1}]}}))
    idx, info = g2.selective_features(p, "all")
    assert list(idx) == [7, 3] and info["truncated"] is True
    idx1, _ = g2.selective_features(p, 1)
    assert list(idx1) == [7]
    assert list(g2.selective_features(p, 99)[0]) == [7, 3]


# ───────────────────────────────────────────────────────────── end-to-end
def _run_cli(tmp_path, *extra, name="g2.json", timeout=900, expect_ok=True):
    out = tmp_path / name
    r = subprocess.run(
        [sys.executable, str(SCENE_SAE / "g2_residual_read.py"), *extra, "--out", str(out)],
        capture_output=True, text=True, cwd=str(REPO), timeout=timeout)
    if not expect_ok:
        return r, None
    assert r.returncode == 0, r.stderr[-4000:]
    return r, json.loads(out.read_text())


def test_smoke_cli_runs_and_labels(tmp_path):
    _r, res = _run_cli(tmp_path, "--smoke", "--arms", "raw,linear", "--r-list", "1,8",
                       "--n-perm", "20", "--token-agg", "future", "--linear-methods", "between",
                       name="g2_smoke.json", timeout=600)
    tags = {x["tag"]: x for x in res["results"]}
    assert res["criteria"]["z_keep"] == g2.Z_KEEP
    assert tags["raw|future"]["verdict"] == "baseline_no_removal"
    # 합성은 scene ⊥ outcome 이므로 scene 이 지워진 설정에서 outcome 이 남아야 한다
    full = tags["linear_between_r8|future"]
    assert full["scene_removed"] is True, full["scene_probe_foldwise"]
    # 기본 smoke 는 episode 축 split → 미열람 scene 표본 (b) 이 없다 → 탐색 등급
    assert full["verdict"] == "outcome_separable_from_scene_exploratory"
    assert full["verdict_scope"] == "all_scenes"
    assert full["verdict_flags"] == ["all_scene_read"]
    assert full["loso_folds"]["heldout_never_in_train"] is True
    # 판정은 fold-wise probe 로, split-train probe 는 참고치로 병기 (리뷰 #1)
    assert full["scene_probe_foldwise"]["projection"] == "fold"
    assert full["scene_probe"] is not None and "scene_removed_split_train_ref" in full
    # r 이 모자란 설정은 제거 실패로 표시
    assert tags["linear_between_r1|future"]["scene_removed"] is False
    # 리뷰 #5 — 창 내 record 수 분포와 길이-confound 주석
    rc = tags["raw|future"]["window_mean"]["record_counts"]
    assert rc["succ_mean"] == rc["fail_mean"] == 6.0
    assert "fixed_t" in rc["note"]


def test_cli_reports_unseen_scene_read_separately(tmp_path):
    """리뷰 #3 — scene 축 split 이면 (a) 전체 / (b) 미열람 scene 두 벌이 나온다."""
    _r, res = _run_cli(tmp_path, "--smoke", "--smoke-scene-split", "--arms", "raw,linear",
                       "--r-list", "8", "--n-perm", "20", "--token-agg", "future",
                       "--linear-methods", "between", name="g2_unseen.json", timeout=600)
    assert len(res["unseen_scenes"]) == 2
    full = next(x for x in res["results"] if x["tag"] == "linear_between_r8|future")
    assert full["verdict_scope"] == "unseen_scenes"
    assert full["verdict_flags"] == []
    assert not full["verdict"].endswith("_exploratory")
    b = full["window_mean_unseen_scenes"]
    assert b is not None and b["scenes"] == res["unseen_scenes"]
    assert b["n_sample"] < full["window_mean"]["n_sample"]      # (b) 는 부분표본


def test_cli_no_baseline_separation(tmp_path):
    """리뷰 #4 — 제거 전 기준선에 분리가 없으면 제거 판정을 하지 않는다."""
    X, meta = g2.synth(seed=0, n_scene=6, n_ep_per_scene=8, n_rec=3, T=5, D=24)
    # 라벨을 데이터의 outcome 축과 무상관으로 재배정 (scene 안 succ/fail 균형 유지)
    meta = dict(meta)
    meta["success"] = ((meta["episode_idx"] % 8) < 4).astype(np.int64)
    d = tmp_path / "inputs"
    d.mkdir()
    np.savez(d / "X_L1.npz", X=X, row_fingerprint=np.asarray("fp"))
    np.savez(d / "meta.npz", row_fingerprint=np.asarray("fp"), **meta)

    _r, res = _run_cli(tmp_path, "--x", str(d / "X_L1.npz"), "--meta", str(d / "meta.npz"),
                       "--arms", "raw,linear", "--r-list", "2", "--linear-methods", "between",
                       "--window", "3", "--token-agg", "future", "--n-perm", "30",
                       name="g2_null.json")
    raw = next(x for x in res["results"] if x["tag"] == "raw|future")
    assert raw["window_mean"]["null_z"] <= g2.Z_KEEP, raw["window_mean"]
    lin = next(x for x in res["results"] if x["tag"] == "linear_between_r2|future")
    assert lin["verdict"] == "no_baseline_separation", lin["window_mean"]
    assert lin["baseline_window_mean_null_z"] == raw["window_mean"]["null_z"]


def _sae_fixture(tmp_path, **ck_kw):
    """scene 축 split 합성 + ckpt + probe json (SAE arm e2e 공용)."""
    pytest.importorskip("torch")
    X, meta = g2.synth(seed=0, n_scene=6, n_ep_per_scene=4, n_rec=4, T=6, D=32,
                       scene_split=True)              # scene 1000..1003 train / 1004,1005 test
    fp = "testfp"
    d = tmp_path / "inputs"
    d.mkdir()
    np.savez(d / "X_L9.npz", X=X.astype(np.float16), row_fingerprint=np.asarray(fp))
    np.savez(d / "meta.npz", row_fingerprint=np.asarray(fp), **meta)
    ck, _D, _m = _tiny_ckpt(tmp_path, D=32, m=64, k=8, **ck_kw)
    pj = tmp_path / f"probe_{ck.name}.json"
    pj.write_text(json.dumps({"selectivity": {
        "n_selective": 3,
        "top_features": [{"feature": i, "selective": True, "gap": 1.0 - 0.1 * i}
                         for i in range(3)]}}))
    return d, ck, pj


def _sae_argv(d, ck, pj, *extra):
    return ["--x", str(d / "X_L9.npz"), "--meta", str(d / "meta.npz"),
            "--ckpt-dir", str(ck), "--probe-json", str(pj), "--layer", "9",
            "--window", "4", "--arms", "raw,sae,linear", "--r-list", "2",
            "--sel-top-list", "1,all", "--linear-methods", "between",
            "--token-agg", "future,all", "--n-perm", "10", "--device", "cpu", *extra]


def test_cli_end_to_end_with_sae_arm(tmp_path):
    """실 입력 계약(X_L*.npz + meta.npz + ckpt + probe json)으로 세 arm 전부 통과."""
    d, ck, pj = _sae_fixture(tmp_path)
    _r, res = _run_cli(tmp_path, *_sae_argv(d, ck, pj))
    tags = {x["tag"] for x in res["results"]}
    for want in ("raw|future", "sae_full|future", "sae_top1|future", "sae_topall|future",
                 "linear_between_r2|future", "raw|all", "sae_topall|all"):
        assert want in tags, (want, sorted(tags))
    assert res["sae"]["sets"]["sae_topall"]["n_removed"] == 3
    # sae_full 은 통제(제거 없음) → 판정 대상 아님
    full = next(x for x in res["results"] if x["tag"] == "sae_full|future")
    assert full["verdict"] == "baseline_no_removal" and full["scene_removed"] is None
    # 제거 arm 은 sae_full 을 기준선으로 삼는다 (참고치·판정용 probe 둘 다)
    top = next(x for x in res["results"] if x["tag"] == "sae_topall|future")
    assert top["scene_probe_baseline"] == full["scene_probe"]
    assert top["scene_probe_foldwise_baseline"] == full["scene_probe_foldwise"]
    assert top["scene_probe_foldwise"]["projection"] == "none"     # SAE 는 masked-decode 표현


def test_sae_ckpt_axis_recorded_and_graded(tmp_path):
    """리뷰 #2 — 평가 scene 이 SAE train_scenes 에 있으면 등급이 _exploratory."""
    d, ck, pj = _sae_fixture(tmp_path)                # train_scenes = 1000..1003 (평가에 포함)
    _r, res = _run_cli(tmp_path, *_sae_argv(d, ck, pj), name="g2_sae_grade.json")
    s = res["sae"]
    assert s["scene_axis_ok"] is True and s["split_col_of_sae"] == "split_scene"
    assert s["eval_scenes"] == 6 and s["eval_scenes_seen_by_sae"] == 4
    assert abs(s["eval_scene_seen_frac"] - 4 / 6) < 1e-9
    assert "재학습" in s["refit_note"]
    top = next(x for x in res["results"] if x["tag"] == "sae_topall|future")
    assert "sae_saw_eval_scenes" in top["verdict_flags"]
    if top["verdict"] in [v + "_exploratory" for v in g2.DECISIVE]:
        assert top["verdict"].endswith("_exploratory")
    # linear arm 은 SAE 사전과 무관 → 그 플래그가 붙지 않는다
    lin = next(x for x in res["results"] if x["tag"] == "linear_between_r2|future")
    assert "sae_saw_eval_scenes" not in lin["verdict_flags"]


def test_sae_ckpt_episode_axis_is_error(tmp_path):
    """리뷰 #2 — episode 축 ckpt 는 에러, --allow-nonscene-sae 로만 강행."""
    d, ck, pj = _sae_fixture(tmp_path, name="ckpt_ep", split_col="split_episode",
                             scene_axis=False)
    r, _ = _run_cli(tmp_path, *_sae_argv(d, ck, pj), name="g2_bad.json", expect_ok=False)
    assert r.returncode != 0
    assert "scene-heldout" in r.stdout + r.stderr
    assert "--allow-nonscene-sae" in r.stdout + r.stderr

    r2, res = _run_cli(tmp_path, *_sae_argv(d, ck, pj, "--allow-nonscene-sae"),
                       name="g2_bad_ok.json")
    assert res["sae"]["scene_axis_ok"] is False
    assert "[warn]" in r2.stdout
