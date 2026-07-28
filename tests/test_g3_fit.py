"""scripts/scene_sae/fit_g3_direction.py — 합성 데이터 단위테스트.

G3 fit 은 "within-scene 실패축 r̂ 를 뽑아 setM NPZ 로 내보낸다". 아는 정답에서 지켜야 할 것:

  (i)   심어 둔 실패 방향을 찾는다 (cos > 0.9).
  (ii)  **부호 규약** — r̂ 는 실패 방향, −r̂ 가 성공 방향 (성공 평균의 r̂ 좌표 < 실패 평균).
  (iii) fold 제외가 실제로 그 inference_seed 판을 fit 에서 뺀다.
  (iv)  잔차화 모드에서 r̂ 가 제거된 between 부분공간과 직교한다.
  (v)   산출 NPZ 가 serve 로더(steering_hooks.load_steering_segment)로 로드된다.
  (vi)  fold 제외가 r̂/Q 를 실제로 바꾼다 + `fit_identities` 가 fit 판 신원을 다 담는다.
  (vii) 전승(성공만) scene 을 섞어 넣어도 setpoint 가 흔들리지 않는다 (혼재 scene 한정).
  (viii) row_fingerprint 가 한쪽만 있으면 에러, `--allow-legacy-inputs` 로만 우회.

전부 CPU·소형 합성 (T=49 는 세그먼트 계약이라 고정, D 만 작게).
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
T = 49                                       # state1 + future32 + action16 (계약)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test",
                                                  SCENE_SAE / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


g3 = _load("fit_g3_direction")


# ─────────────────────────────────────────────────────────────── 합성 입력 생성
def make_inputs(tmp: Path, *, D=24, n_scene=6, n_inf=4, n_rec=8, seed=0,
                fail_amp=3.0, scene_amp=12.0, layer=12, n_pure_succ_scene=0,
                fingerprint="testfp00", name=""):
    """scene-matched 격자 모사 → X_L{layer}.npz + meta.npz.

    scene 중심 = 공유 평면 span{u1,u2} 위 원형 배치(= between-scatter rank 2 로 제거 가능),
    실패 방향 = 그 평면과 직교인 `fail_dir` (episode 가 실패면 +fail_amp).
    episode = (scene, inference_seed) 격자. 라벨은 scene 마다 성공/실패가 섞이게 준다.
    `n_pure_succ_scene` > 0 이면 **전승(성공만) scene** 을 뒤에 덧붙인다 (혼재 scene 한정
    로직 검증용 — 앞선 rng 소비 순서를 바꾸지 않으므로 기존 행은 비트 동일).
    반환 (x_path, meta_path, fail_dir, scene_plane[D,2]).
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(D, 3)))
    u1, u2, fail_dir = Q[:, 0], Q[:, 1], Q[:, 2]
    rows, meta = [], {k: [] for k in ("episode_idx", "record_idx", "token_idx", "token_seg",
                                      "phase_code", "success", "scenario_seed",
                                      "inference_seed", "layout_id", "style_id", "split")}
    seg = np.asarray([g3.seg_of_token(t) for t in range(T)], np.int8)
    e = 0
    for s in range(n_scene + n_pure_succ_scene):
        c = np.cos(2 * np.pi * s / n_scene) * u1 + np.sin(2 * np.pi * s / n_scene) * u2
        pure = s >= n_scene
        if pure:                                  # 전승 scene 은 평면 밖으로 크게 치우치게
            c = c * 3.0
        for j in range(n_inf):
            succ = 1 if pure else int(j % 2 == 0)  # 혼재 scene 은 succ/fail 공존
            for r in range(n_rec):
                base = scene_amp * c + (0.0 if succ else fail_amp) * fail_dir
                for t in range(T):
                    rows.append(base + rng.normal(scale=0.2, size=D))
                    meta["episode_idx"].append(e)
                    meta["record_idx"].append(r)
                    meta["token_idx"].append(t)
                    meta["token_seg"].append(int(seg[t]))
                    meta["phase_code"].append(0)
                    meta["success"].append(succ)
                    meta["scenario_seed"].append(100000 + s)
                    meta["inference_seed"].append(j * 1_000_000)
                    meta["layout_id"].append(s % 2)
                    meta["style_id"].append(s % 2)
                    meta["split"].append(0)
            e += 1
    X = np.asarray(rows, np.float16)
    meta = {k: np.asarray(v) for k, v in meta.items()}
    xp, mp = tmp / f"X{name}_L{layer}.npz", tmp / f"meta{name}.npz"
    np.savez(xp, X=X, row_fingerprint=np.asarray(fingerprint))
    np.savez(mp, row_fingerprint=np.asarray(fingerprint), **meta)
    return xp, mp, fail_dir, np.stack([u1, u2], axis=1)


def run_fit_raw(xp: Path, mp: Path, out: Path, *, rank=0, extra=()):
    cmd = [sys.executable, str(SCENE_SAE / "fit_g3_direction.py"),
           "--x", str(xp), "--meta", str(mp), "--layer", "12", "--cell", "synth",
           "--window", "8", "--residual-rank", str(rank), "--out-dir", str(out), "--quiet",
           *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


def run_fit(tmp: Path, xp: Path, mp: Path, out: Path, *, rank=0, extra=()):
    p = run_fit_raw(xp, mp, out, rank=rank, extra=extra)
    assert p.returncode == 0, p.stdout + p.stderr
    fm = json.loads((out / "fit_meta.json").read_text())
    z = np.load(out / "steer" / "dit_L12" / "conceptors.npz")
    return fm, z, p.stdout


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("g3")
    return (tmp, *make_inputs(tmp))


# ───────────────────────────────────────────────────────────────────── (i)(ii)
def test_recovers_planted_fail_direction_and_sign(synth, tmp_path):
    tmp, xp, mp, fail_dir, _plane = synth
    fm, z, _ = run_fit(tmp, xp, mp, tmp_path / "r0")
    r = z["alpha0_v_seg"][0].astype(np.float64)
    assert abs(float(np.linalg.norm(r)) - 1.0) < 1e-3
    # (i) 심은 실패 방향 회수
    assert float(r @ fail_dir) > 0.9, f"cos={r @ fail_dir:.3f}"
    # v_seg 3행 모두 같은 r̂ (세그먼트 확장 규약)
    assert np.allclose(z["alpha0_v_seg"][1], z["alpha0_v_seg"][0])
    # (ii) 부호: 실패 평균의 r̂ 좌표 > 성공 평균(setpoint) → −r̂ 가 성공 방향
    assert fm["diagnostics"]["gap_all_tokens_positive"] is True
    assert all(v > 0 for v in fm["diagnostics"]["gap_fail_minus_succ_per_segment"].values())
    assert "성공 쪽으로 미는 것은 −r̂" in fm["sign_convention"]
    # 진단 필수 항목
    c = fm["diagnostics"]["cos_scene_dir_vs_rhat"]
    assert c["min"] <= c["median"] <= c["max"] and c["mean"] > 0.5
    assert fm["diagnostics"]["n_scene_mixed"] == 6


def test_seg_mask_selects_apply_segments(synth, tmp_path):
    tmp, xp, mp, _fd, _pl = synth
    fm, z, _ = run_fit(tmp, xp, mp, tmp_path / "segs", extra=["--apply-segs", "future,action"])
    assert z["alpha0_seg_mask"].tolist() == [0.0, 1.0, 1.0]
    assert fm["apply_segs"] == ["future", "action"]
    # 기본(future 만)
    fm2, z2, _ = run_fit(tmp, xp, mp, tmp_path / "segs_def")
    assert z2["alpha0_seg_mask"].tolist() == [0.0, 1.0, 0.0]
    assert fm2["operator_modes"]["additive"]["serve_supported"] is False


# ───────────────────────────────────────────────────────────────────── (iii)
def test_fold_exclusion_drops_that_inference_seed(synth, tmp_path):
    tmp, xp, mp, _fd, _pl = synth
    full, _z, _ = run_fit(tmp, xp, mp, tmp_path / "fall")
    fold, _z2, _ = run_fit(tmp, xp, mp, tmp_path / "f2", extra=["--exclude-fold", "2"])
    assert full["fold"]["exclude_inference_seed"] == -1
    assert full["n_fit_episodes"] == 24                      # 6 scene × 4 inference_seed
    assert fold["n_fit_episodes"] == 18                      # inference_seed 2e6 (6판) 제외
    assert fold["fold"]["exclude_inference_seed"] == 2_000_000
    assert len(fold["fold"]["excluded_episodes"]) == 6
    # 제외된 episode 는 fit 목록에 없다
    assert not set(fold["fold"]["excluded_episodes"]) & set(fold["fit_episodes"])
    # 같은 격자에서 raw seed 지정도 동일 결과
    raw, _z3, _ = run_fit(tmp, xp, mp, tmp_path / "f2raw",
                          extra=["--exclude-inference-seed", "2000000"])
    assert raw["fit_episodes"] == fold["fit_episodes"]


# ───────────────────────────────────────────────────────────────────── (vi)
def test_fold_exclusion_changes_direction_and_subspace(synth, tmp_path):
    """fold 제외가 목록만 바꾸는 게 아니라 **fit 결과 벡터**를 실제로 바꾼다."""
    tmp, xp, mp, _fd, _pl = synth
    for rank in (0, 2):                          # rank>0 이면 r̂ 는 Q 에도 의존 → Q 변화 대리
        _f, zf, _ = run_fit(tmp, xp, mp, tmp_path / f"vall_r{rank}", rank=rank)
        _g, zg, _ = run_fit(tmp, xp, mp, tmp_path / f"vf2_r{rank}", rank=rank,
                            extra=["--exclude-fold", "2"])
        ra, rb = zf["g3_r_hat"].astype(np.float64), zg["g3_r_hat"].astype(np.float64)
        assert not np.allclose(ra, rb, atol=1e-7), f"rank={rank} r̂ 가 fold 제외에도 동일"
        assert float(ra @ rb) > 0.5              # 그래도 같은 축이어야 한다 (붕괴 아님)
        # setpoint 도 표본이 줄면 달라진다
        assert not np.allclose(zf["alpha0_s_tok"], zg["alpha0_s_tok"], atol=1e-6)


def test_fit_identities_manifest(synth, tmp_path):
    """eval 중복 검사용 (scenario_seed, inference_seed) 목록이 fit 표본과 정확히 일치."""
    tmp, xp, mp, _fd, _pl = synth
    fold, _z, _ = run_fit(tmp, xp, mp, tmp_path / "ident", extra=["--exclude-fold", "2"])
    ids = [tuple(p) for p in fold["fit_identities"]]
    assert len(ids) == fold["n_fit_episodes"] == 18
    assert len(set(ids)) == len(ids)                       # (scene, inf) 는 격자에서 유일
    assert {s for s, _ in ids} == set(fold["fit_scenes"])  # 6 scene
    assert 2_000_000 not in {i for _s, i in ids}           # 제외 fold 는 목록에 없다
    assert {i for _s, i in ids} == {0, 1_000_000, 3_000_000}
    # 전판 fit 은 4 fold 전부
    full, _z2, _ = run_fit(tmp, xp, mp, tmp_path / "ident_all")
    assert {i for _s, i in (tuple(p) for p in full["fit_identities"])} == {
        0, 1_000_000, 2_000_000, 3_000_000}
    assert "결정적" in full["determinism"]


# ───────────────────────────────────────────────────────────────────── (vii)
def test_setpoint_ignores_pure_success_scenes(tmp_path):
    """전승 scene 을 덧붙여도 s_tok·gap 이 흔들리지 않는다 (setpoint 표본 = 혼재 scene 한정)."""
    src = tmp_path / "base"
    src.mkdir()
    xp0, mp0, _fd, _pl = make_inputs(src)
    xp1, mp1, _fd1, _pl1 = make_inputs(src, n_pure_succ_scene=3, name="_pure")
    fm0, z0, _ = run_fit(tmp_path, xp0, mp0, tmp_path / "sp_base")
    fm1, z1, _ = run_fit(tmp_path, xp1, mp1, tmp_path / "sp_pure")
    # 전승 scene 은 혼재 scene 목록에 안 들어간다 → r̂ 는 그대로
    assert fm0["fit_scenes"] == fm1["fit_scenes"] == [100000 + s for s in range(6)]
    assert fm1["n_fit_episodes"] == 36 and fm1["setpoint"]["n_setpoint_episodes"] == 24
    assert len(fm1["setpoint"]["setpoint_episodes_excluded_pure_scene"]) == 12
    assert np.allclose(z0["g3_r_hat"], z1["g3_r_hat"], atol=1e-6)
    # ★ setpoint 불변 (전승 scene 이 μ_succ 을 끌어당기지 않는다)
    assert np.allclose(z0["alpha0_s_tok"], z1["alpha0_s_tok"], rtol=1e-4, atol=1e-3), (
        float(np.abs(z0["alpha0_s_tok"] - z1["alpha0_s_tok"]).max()))
    for nm in ("state", "future", "action"):
        g0 = fm0["diagnostics"]["gap_fail_minus_succ_per_segment"][nm]
        g1 = fm1["diagnostics"]["gap_fail_minus_succ_per_segment"][nm]
        assert abs(g0 - g1) < 1e-3 * max(1.0, abs(g0))
    # 창 충족 가드: cap=window 라 전 episode 가 창을 채운다
    assert fm1["window_fill"]["all_episodes_full"] is True
    assert fm1["window_fill"]["n_episodes_below_window"] == 0
    assert fm1["window_fill"]["per_class_record_counts"]["fail"]["min"] == 8


def test_window_fill_warns_on_short_episodes(tmp_path):
    """창보다 짧은 episode 가 있으면 경고 + 분포 기록 (assert 아님 — fit 은 계속)."""
    xp, mp, _fd, _pl = make_inputs(tmp_path, n_rec=8)
    p = run_fit_raw(xp, mp, tmp_path / "w12", extra=["--window", "12"])
    assert p.returncode == 0, p.stdout + p.stderr      # 경고일 뿐 실패는 아님
    assert "창 미달" in p.stderr
    fm = json.loads((tmp_path / "w12" / "fit_meta.json").read_text())
    wf = fm["window_fill"]
    assert wf["all_episodes_full"] is False and wf["n_episodes_below_window"] == 24
    assert wf["per_class_record_counts"]["succ"]["max"] == 8      # 창 12 인데 8 record 뿐
    assert wf["per_class_record_counts"]["fail"]["hist"] == {"8": 12}


# ───────────────────────────────────────────────────────────────────── (viii)
def _strip_fp(src: Path, dst: Path):
    z = np.load(src)
    np.savez(dst, **{k: z[k] for k in z.files if k != "row_fingerprint"})
    return dst


def test_fingerprint_required_on_both_sides(tmp_path):
    xp, mp, _fd, _pl = make_inputs(tmp_path)
    # 한쪽(meta)만 결측 → 에러
    mp_nofp = _strip_fp(mp, tmp_path / "meta_nofp.npz")
    p = run_fit_raw(xp, mp_nofp, tmp_path / "nofp")
    assert p.returncode != 0 and "row_fingerprint 결측" in (p.stdout + p.stderr)
    # 반대쪽(X)만 결측도 에러
    xp_nofp = _strip_fp(xp, tmp_path / "X_nofp_L12.npz")
    p2 = run_fit_raw(xp_nofp, mp, tmp_path / "nofp2")
    assert p2.returncode != 0 and "row_fingerprint 결측" in (p2.stdout + p2.stderr)
    # --allow-legacy-inputs 로만 우회 + meta 에 기록
    fm, _z, _ = run_fit(tmp_path, xp, mp_nofp, tmp_path / "legacy",
                        extra=["--allow-legacy-inputs"])
    assert fm["inputs_integrity"]["row_fingerprint_status"] == "only_x"
    assert fm["inputs_integrity"]["allow_legacy_inputs"] is True
    # 정상 입력은 both 로 기록
    ok, _z2, _ = run_fit(tmp_path, xp, mp, tmp_path / "bothfp")
    assert ok["inputs_integrity"]["row_fingerprint_status"] == "both"
    assert ok["inputs_integrity"]["allow_legacy_inputs"] is False
    # 양쪽 있으나 불일치 → 기존 에러 유지
    xp2, mp2, _f, _pp = make_inputs(tmp_path, fingerprint="OTHERFP", name="_other")
    p3 = run_fit_raw(xp2, mp, tmp_path / "mismatch")
    assert p3.returncode != 0 and "row_fingerprint 불일치" in (p3.stdout + p3.stderr)
    del mp2, _pp


def test_row_count_mismatch_errors(tmp_path):
    xp, mp, _fd, _pl = make_inputs(tmp_path)
    z = np.load(xp)
    short = tmp_path / "X_short_L12.npz"
    np.savez(short, X=z["X"][:-49], row_fingerprint=z["row_fingerprint"])
    p = run_fit_raw(short, mp, tmp_path / "shortrows")
    assert p.returncode != 0 and "행 수 불일치" in (p.stdout + p.stderr)


# ───────────────────────────────────────────────────────────────────── (iv)
def test_residual_mode_orthogonal_to_between_subspace(synth, tmp_path):
    tmp, xp, mp, fail_dir, plane = synth
    fm, z, _ = run_fit(tmp, xp, mp, tmp_path / "r2", rank=2)
    r = z["alpha0_v_seg"][0].astype(np.float64)
    # 제거한 scene 부분공간(공유 평면)과 직교
    assert float(np.abs(plane.T @ r).max()) < 0.05, f"cos={np.abs(plane.T @ r).max():.3f}"
    assert fm["diagnostics"]["cos_rhat_vs_removed_basis_max"] < 1e-6
    assert fm["diagnostics"]["residual_basis_rank_actual"] == 2
    # 잔차화 유/무 r̂ 비교치가 기록된다 ([4-1] 진단)
    assert 0.0 <= fm["diagnostics"]["cos_vs_no_residual_rhat"] <= 1.0
    # 실패 방향은 평면과 직교하게 심었으므로 잔차화해도 회수된다
    assert float(r @ fail_dir) > 0.9


# ───────────────────────────────────────────────────────────────────── (v)
def test_npz_loads_with_serve_loader(synth, tmp_path):
    tmp, xp, mp, _fd, _pl = synth
    pytest.importorskip("torch")
    sys.path.insert(0, str(REPO / "scripts" / "serve"))
    from steering_hooks import load_steering_segment            # noqa: PLC0415

    _fm, _z, _ = run_fit(tmp, xp, mp, tmp_path / "load")
    v_seg, s_tok, bounds, mask = load_steering_segment(
        tmp_path / "load" / "steer" / "dit_L12" / "conceptors.npz")
    assert v_seg.shape == (3, 24) and s_tok.shape == (T,)
    assert bounds.tolist() == [[0, 1], [1, 33], [33, 49]]
    assert mask.tolist() == [0.0, 1.0, 0.0]
