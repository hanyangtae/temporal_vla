"""N1.5 phase-event 대조 conceptor fit (Rung 3 steering target 생성).

방향 1: 관측 분리가 confound(scene/길이)로 약하다는 걸 확인했으니, **관측을 더 파지 않고
steering 연산자를 fit해 인과(ΔSR)로 판정**한다(사용자 논리 = 인과가 arbiter).

N1.6판 fit_conceptor_steering.py 는 [L,D]/[L,T,D] token layout·task_id 그룹핑이라 N1.5
phase-event([L=7,K=4,D=1536] denoise + feature_phases + seed별 cell)에 안 맞음. 이 어댑터는:
  - **단일 scene(cell dir, seed 구분)** 내에서만 fit → scene confound 제거([[dit-succfail-apparent-separation-confound]]).
  - denoise축(K=4) mean-pool → 주입점(transformer_blocks[ℓ] residual, D=1536) / VL(vlln, D=2048)과 동일 공간.
  - group = {global(전 record), 각 phase(reach/transport/insert-settle)} × layer(DiT capture 7 + VL).
  - C_steer = C_success ∧ ¬C_failure (src.conceptor.contrastive_conceptor), alpha grid sweep.

산출 NPZ 는 serve 소비 계약(steering_hooks.load_steering_matrix)과 동일: 키 `alpha{a}_C_steer`
/`_C_success`/`_C_failure`. 사용: serve `--steering-npz <.../conceptors.npz> --steering-pathway
{dit,vl} [--steering-layer <blk>] --steering-alpha <a> --steering-beta <b> --steering-key C_steer`.
DiT layer_tag `dit_L<blk>` 의 <blk> 를 그대로 --steering-layer 로 넘긴다(transformer_blocks 인덱스).

재설계 라운드 v2 확장 (docs/steering/17 배선 체크리스트, 2026-07-10):
  - NPZ 키 순서 = [선택 α, 안전 default] 명시 순서 (구판은 set 순회라 serve 첫-키 폴백이 hash
    우연에 좌우 — α 오배선 원인, [[alpha-wiring-audit]]).
  - `--min-per-class` 는 **episode 수** 기준 (구판은 record 수 — timeout 실패의 record 과대가중
    탓에 1 episode 로도 통과하던 구멍).
  - `--manifest` 로 fit 표본을 외부 명시 (pkl_path\tlabel[\tscene]): 30/30 split 층화 샘플링·
    corrected 라벨·위약(라벨 permutation) 은 전부 manifest 생성 단계(pq2)에서 결정되고, 이
    스크립트는 소비만 한다. 사용 표본·content 서명은 out_root/fit_inputs.json 에 기록.
  - 유효 fit 이 0 개면 exit 3 (빈 fit 이 [done] 으로 통과하던 게이트 구멍 봉합).

pq3 확장 (계획서 dynamic-riding-aurora v9, 2026-07-15):
  - **full-token pkl 지원**: capture_token_mode="all_token_full" pkl(record [L,K,T,D])을
    자체 로더로 소비 — `--token-pool {mean,state,future,action}` 으로 fit 시점 세그먼트
    mean (COAST 49토큰 정렬 = mean). 공용 load_rollout(pool_denoise ndim==3 강제)은 무수정.
  - `--require-capture-token-mode`: pkl meta 불일치/부재 시 exit 4 (구/신 캡처 혼입 게이트).
  - `--denoise per_step`: step k 별 conceptor 4세트 산출 — NPZ 키
    `step{k}_alpha{a}_C_steer/_C_success/_C_failure` + metadata `selected_alpha_per_step`
    (serve `--steering-denoise per_step` 소비 계약).
  - α grid default = COAST Table 14 GR00T RoboCasa 그대로.
  - **quota floor guard**: Stage2 α 선택 후보를 quota_steer≥floor 로 제한 (퇴화 α =
    C_steer≈0 → M≈(1-β)I 균일수축 배제). 전 α 미달 시 해당 group×layer skip(경고).
  - `--stage1-quota-sweep`: α₀(기본 10)에서 layer별 global(C_steer quota)·phase-bin
    (quota 평균) 표만 출력 — regime별 Stage1 layer 선택의 사용자 보고 게이트용.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/safe/groot_n15/robocasa/analyze"))

from phase_separation import load_rollout, phase_records  # noqa: E402
from src.conceptor import (  # noqa: E402
    and_conceptor,
    as_float32,
    compute_conceptor,
    conceptor_overlap,
    conceptor_quota,
    not_conceptor,
)

# legacy default 보존 (pq2 호출자 재실행 결과 불변 — Gate 2 높음#4).
DEFAULT_ALPHAS = [0.1, 0.3, 1.0, 3.0, 10.0]
# COAST Table 14 (p.29) GR00T RoboCasa aperture grid — pq3 는 --alphas table14 로 명시 사용.
TABLE14_ALPHAS = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
OVERLAP_BAND = (0.85, 0.95)  # COAST A.10.2 Stage2
FULLTOKEN_MODE = "all_token_full"  # serve --groot-dit-token-pool 신모드와 동일 enum
FULLTOKEN_KIND = "groot_n15_dit_block_residual_full_tokens_denoise"  # safe_metadata 와 일치


def carve_5phase(pkl_path, phases: list[str], w: int) -> list[str]:
    """5-phase carving (사용자 지시): event 직전 W record 를 pre-grasp/pre-place 로 재라벨.

    strict(drop-aware) 수집 pkl 의 grasp_steps(반복 가능)/event_steps['place:obj'] 사용.
    - pre-grasp: 각 grasp record-idx g 직전 [g-w, g) 중 reach-to-object 인 record.
    - pre-place: place record-idx p 직전 [p-w, p) 중 transport 인 record.
    → reach / pre-grasp / transport / pre-place / insert-settle 5-phase.
    """
    import pickle

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    out = list(phases)
    n = len(out)
    grasp_steps = list(d.get("grasp_steps") or [])
    if not grasp_steps and d.get("event_steps", {}).get("grasp:obj") is not None:
        grasp_steps = [d["event_steps"]["grasp:obj"]]
    for g in grasp_steps:
        for r in range(max(0, int(g) - w), min(int(g), n)):
            if out[r] == "reach-to-object":
                out[r] = "pre-grasp"
    p_step = d.get("event_steps", {}).get("place:obj")
    if p_step is not None:
        for r in range(max(0, int(p_step) - w), min(int(p_step), n)):
            if out[r] == "transport":
                out[r] = "pre-place"
    return out


def _roll_records(r, phase, layer_key):
    """rollout 1개의 (phase, layer) record 벡터 목록. phase='global'이면 전 record."""
    if phase == "global":
        if layer_key == "VL":
            return list(r["vl"]) if r["vl"] is not None else []
        return list(r["dit"][:, layer_key, :])
    return list(phase_records(r, phase, layer_key))


def gather_class_records(rolls, phase, layer_key, success):
    """success(0/1) rollout 들의 phase 소속 record 벡터 [N, D] 스택. phase='global'이면 전 record.

    wrong-grasp record 는 phase bin 에선 라벨 분리로 자동 제외(fit group 에 wrong-grasp 없음),
    global 은 사용자 결정대로 전 record 포함(phase 무구분 조건이므로).
    """
    return gather_class_records_eps(rolls, phase, layer_key, success)[0]


def gather_class_records_eps(rolls, phase, layer_key, success):
    """gather_class_records + 기여 episode 수 (해당 group/layer 에 record ≥1 인 rollout 수)."""
    out, n_eps = [], 0
    for r in rolls:
        if r["success"] != success:
            continue
        recs = _roll_records(r, phase, layer_key)
        if recs:
            n_eps += 1
            out.extend(recs)
    return (np.asarray(out, dtype=np.float64) if out else np.empty((0, 0))), n_eps


def select_alpha(sweep, band, quota_floor=0.0):
    """overlap 밴드 + quota floor guard 로 α 선택.

    후보 = quota_steer≥floor. 밴드 내 후보가 있으면 최소 α("band"), 없으면 후보 중
    밴드에 가장 가까운 α("closest"). 전 α 가 floor 미달이면 (None, "floor_exhausted")
    — 퇴화 α(C_steer≈0 → M≈(1-β)I 균일수축) 채택 방지 (pq2 함정 봉합).
    """
    lo, hi = band
    eligible = [r for r in sweep if r.get("quota_steer", 1.0) >= quota_floor]
    if not eligible:
        return None, "floor_exhausted"
    in_band = sorted(r["alpha"] for r in eligible if lo <= r["overlap"] <= hi)
    if in_band:
        return in_band[0], "band"
    return min(eligible, key=lambda r: min(abs(r["overlap"] - lo), abs(r["overlap"] - hi)))["alpha"], "closest"


def fit_one(Xs, Xf, alphas, band, quota_floor=0.0):
    """succ/fail 표본 → alpha별 C_steer + sweep. 반환 (fits{alpha:dict}|None, meta).

    fits=None 이면 전 α quota floor 미달 (호출자가 해당 group×layer skip).
    """
    sweep, cache = [], {}
    for a in alphas:
        Cs = compute_conceptor(Xs, a)
        Cf = compute_conceptor(Xf, a)
        Csteer = and_conceptor(Cs, not_conceptor(Cf))
        cache[a] = (Cs, Cf, Csteer)
        sweep.append({"alpha": float(a), "overlap": conceptor_overlap(Cs, Cf),
                      "quota_success": conceptor_quota(Cs), "quota_failure": conceptor_quota(Cf),
                      "quota_steer": conceptor_quota(Csteer)})
    sel_alpha, sel_mode = select_alpha(sweep, band, quota_floor)
    base_meta = {"alpha_sweep": sweep, "quota_floor": float(quota_floor),
                 "n_success": int(Xs.shape[0]), "n_failure": int(Xf.shape[0]),
                 "feature_dim": int(Xs.shape[1])}
    if sel_alpha is None:
        return None, {**base_meta, "selected_alpha": None, "selection_mode": sel_mode}
    fits = {}
    # 선택 alpha 를 반드시 첫 키로 저장 (serve 의 첫-키 폴백이 선택값을 집도록 — set 순회 금지)
    default_alpha = alphas[1] if len(alphas) > 1 else alphas[0]
    save_alphas = [sel_alpha] + ([default_alpha] if default_alpha != sel_alpha else [])
    for a in save_alphas:
        Cs, Cf, Csteer = cache[a]
        fits[a] = {"C_steer": Csteer, "C_success": Cs, "C_failure": Cf,
                   "quota_steer": conceptor_quota(Csteer)}
    return fits, {**base_meta, "selected_alpha": float(sel_alpha), "selection_mode": sel_mode,
                  "quota_steer": {f"{a:g}": g["quota_steer"] for a, g in fits.items()}}


def _content_sig(p: Path) -> str:
    """pkl content 서명: 전체 streaming sha256 (Gate 2 높음#9 — 부분 해시는 수십 MB
    full-token pkl 의 중간 변조를 못 잡음. Gate D hash 동결의 기반이므로 전체 해시)."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def save_npz(out_dir, fits, meta):
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for a, g in fits.items():
        arrays[f"alpha{a:g}_C_steer"] = as_float32(g["C_steer"])
        arrays[f"alpha{a:g}_C_success"] = as_float32(g["C_success"])
        arrays[f"alpha{a:g}_C_failure"] = as_float32(g["C_failure"])
    np.savez_compressed(out_dir / "conceptors.npz", **arrays)
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def save_npz_per_step(out_dir, step_fits, meta):
    """Per-Step NPZ: 키 `step{k}_alpha{a}_C_steer/_C_success/_C_failure` (serve
    load_steering_matrices_per_step 소비 계약). 저장 순서 = step 오름차순 × (선택 α 먼저)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for k in sorted(step_fits):
        for a, g in step_fits[k].items():
            arrays[f"step{k}_alpha{a:g}_C_steer"] = as_float32(g["C_steer"])
            arrays[f"step{k}_alpha{a:g}_C_success"] = as_float32(g["C_success"])
            arrays[f"step{k}_alpha{a:g}_C_failure"] = as_float32(g["C_failure"])
    np.savez_compressed(out_dir / "conceptors.npz", **arrays)
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def _token_segments(T: int, action_horizon: int, n_future: int = 32):
    """N1.5 DiT 시퀀스(state 1 + future n + action horizon) 세그먼트 경계.

    T != 1+n_future+action_horizon 이면 ValueError — S1 스모크 실측(T=49 가정) 불일치 시
    세그먼트 fit 을 막는다 (mean 풀링은 T 무관이라 영향 없음).
    """
    if 1 + n_future + action_horizon != T:
        raise ValueError(
            f"token segment 가정(1+{n_future}+{action_horizon}) != T={T} — "
            "S1 실측 후 --token-pool mean 외 모드는 경계 재정의 필요"
        )
    return {"state": (0, 1), "future": (1, 1 + n_future), "action": (T - action_horizon, T)}


def load_rollout_fulltoken(d: dict, pkl_path, token_pool: str):
    """full-token pkl(dict 로드본) → 기존 rolls 계약 + dit_k([n,L,K,D]) 확장.

    record [L,K,T,D] 를 record 단위로 token-pool 해 [L,K,D] 로 줄인 뒤 stack
    ([n,L,K,T,D] 를 통째로 올리지 않음 — 메모리). vl full-token([T_vl,Dvl])은
    load 시 mean 으로 기존 [n,Dvl] 계약 유지 (pq3 arm 은 DiT 전용).
    """
    hs = d.get("hidden_states") or []
    if not hs:
        raise ValueError(f"{pkl_path}: hidden_states 비어 있음")
    pooled = []
    T_seen = None
    for rec in hs:
        a = np.asarray(rec, dtype=np.float32)
        if a.ndim != 4:
            raise ValueError(f"{pkl_path}: expected full-token record [L,K,T,D], got {a.shape}")
        T_seen = a.shape[2]
        if token_pool == "mean":
            pooled.append(a.mean(axis=2))  # [L,K,D]
        else:
            horizon = int(d.get("model_action_horizon") or 16)
            lo, hi = _token_segments(T_seen, horizon)[token_pool]
            pooled.append(a[:, :, lo:hi, :].mean(axis=2))
    dit_k = np.stack(pooled, axis=0)  # [n, L, K, D]
    vl = d.get("vl_hidden_states")
    vl_arr = None
    if vl is not None:
        vl_rows = []
        for v in vl:
            va = np.asarray(v, dtype=np.float32)
            vl_rows.append(va.mean(axis=0) if va.ndim == 2 else va)  # [T_vl,Dvl]→[Dvl]
        vl_arr = np.stack(vl_rows, axis=0)
    phases = list(d.get("feature_phases") or [])
    if len(phases) != dit_k.shape[0]:
        raise ValueError(
            f"{Path(pkl_path).name}: feature_phases {len(phases)} != records {dit_k.shape[0]}"
        )
    return {
        "name": Path(pkl_path).name,
        "success": int(d.get("episode_success", 0)),
        "length": int(dit_k.shape[0]),
        "dit_k": dit_k,                      # [n, L, K, D] (denoise 처리 전)
        "dit": dit_k.mean(axis=2),           # [n, L, D] (pool 기본 뷰)
        "vl": vl_arr,
        "phases": phases,
        "capture_layers": d.get("capture_layers"),
        "token_count": T_seen,
        "capture_token_mode": d.get("capture_token_mode"),
    }


def main():
    ap = argparse.ArgumentParser(description="N1.5 phase-event 대조 conceptor fit")
    ap.add_argument("--run-dir", default="outputs/eval/robocasa/groot_n15/phase_event_aligned_4cell/raw_rollouts")
    ap.add_argument("--cell", required=True, help="예: PickPlaceCounterToCabinet/ppcc_bread")
    ap.add_argument("--groups", default="global,transport,reach-to-object")
    ap.add_argument("--alphas", default=None)
    ap.add_argument("--min-per-class", type=int, default=3,
                    help="클래스별 최소 기여 episode 수 (v2: record 수 아님)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--denoise", choices=["pool", "stack", "step0", "per_step"], default="pool",
                    help="K(denoise) 축 처리: pool=평균(현행) | stack=step별 개별 record"
                         "(COAST Global 충실 — inference step당 K개 표본) | step0=첫 step만 | "
                         "per_step=step k별 conceptor 산출(pq3 — NPZ 키 step{k}_alpha{a}_*, "
                         "full-token pkl 전용)")
    ap.add_argument("--require-capture-token-mode", default=None,
                    help="pkl meta capture_token_mode 강제 (예: all_token_full). 불일치/부재 "
                         "pkl 발견 시 exit 4 — 구/신 캡처 혼입 게이트")
    ap.add_argument("--token-pool", choices=["mean", "state", "future", "action"], default="mean",
                    help="full-token pkl 의 fit 시점 토큰 풀링: mean=전체 T mean(COAST 정렬) | "
                         "state/future/action=세그먼트별 mean (T=1+32+horizon 가정, S1 실측 대조)")
    ap.add_argument("--layers", default=None,
                    help="fit 할 layer 필터 (물리 번호 콤마목록 + 'VL', 예: '10,15,VL'). "
                         "미지정 시 전 capture layer + VL")
    ap.add_argument("--quota-floor", type=float, default=0.0,
                    help="Stage2 α 후보의 quota_steer 하한 (퇴화 α 배제 guard). "
                         "기본 0=미적용(legacy 보존) — pq3 는 0.01 명시 권장")
    ap.add_argument("--eval-reserved", action="append", default=None,
                    help="eval_reserved.json 경로 (반복 지정 — task-pooled manifest 는 소속 "
                         "cell 전부 필수). manifest 의 scene(seed)이 unseen 예약과 겹치면 "
                         "exit 5 (fit 실행 자체가 침범 검사를 수행, Gate 2 높음#7/R2)")
    ap.add_argument("--stage1-quota-sweep", action="store_true",
                    help="Stage1 layer 선택용 quota 표만 출력(NPZ 저장 없음): α₀에서 "
                         "layer별 global C_steer quota(perm regime) + phase-bin quota 평균"
                         "(gated regime). denoise 처리는 stack(전 step record 풀링) 고정")
    ap.add_argument("--stage1-alpha", type=float, default=10.0,
                    help="Stage1 quota sweep 의 α₀ (COAST A.10.2 = 10)")
    ap.add_argument("--manifest", default=None,
                    help="fit 표본 manifest tsv (pkl_path\\tlabel[\\tscene], # 주석 허용). 지정 시 "
                         "--cell glob 대신 이 목록만 사용하고 label 이 pkl 내장 episode_success 를 "
                         "override (corrected/위약 라벨 주입 경로). 경로는 절대 또는 repo-root 상대.")
    ap.add_argument("--carve-window", type=int, default=0,
                    help=">0 이면 5-phase carving: event 직전 W record 를 pre-grasp/pre-place 로 재라벨")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    manifest_rows = None
    if args.manifest:
        manifest_rows = []
        for line in Path(args.manifest).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            p = Path(parts[0])
            if not p.is_absolute():
                p = REPO / p
            manifest_rows.append({"pkl": p, "label": int(parts[1]),
                                  "scene": parts[2] if len(parts) > 2 else ""})
        pkls = [m["pkl"] for m in manifest_rows]
        missing = [str(p) for p in pkls if not p.exists()]
        if missing:
            raise SystemExit(f"manifest pkl 누락 {len(missing)}개: {missing[:3]}")
    else:
        cell_dir = run_dir / args.cell
        pkls = sorted(cell_dir.glob("*.pkl"))
    if not pkls:
        raise SystemExit(f"no pkl (cell={args.cell}, manifest={args.manifest})")
    import pickle as _pk

    # eval 예약 침범 검사 (fit 실행 자체가 필수 수행 — standalone check 생략 우회 봉쇄).
    # 반복 지정으로 task-pooled manifest 의 소속 cell 전부를 덮는다. scene 열이 비숫자인
    # 행은 검사 우회 경로이므로 fail-closed (Gate2 R2 높음#2).
    if args.eval_reserved and manifest_rows:
        unseen = set()
        for rp in args.eval_reserved:
            unseen |= {int(s) for s in json.loads(Path(rp).read_text())["unseen_seeds"]}
        malformed = [m for m in manifest_rows
                     if not str(m.get("scene", "")).strip().lstrip("-").isdigit()]
        if malformed:
            print(f"[EVAL-SEED 침범 검사] scene 열 비숫자 행 (fail-closed): "
                  f"{[str(m.get('scene')) for m in malformed][:5]}", file=sys.stderr)
            sys.exit(5)
        bad = [m for m in manifest_rows if int(m["scene"]) in unseen]
        if bad:
            print(f"[EVAL-SEED 침범] fit manifest 가 unseen 예약 seed 사용: "
                  f"{[m['scene'] for m in bad][:5]}", file=sys.stderr)
            sys.exit(5)

    # 로더 dispatch: pkl meta capture_token_mode 가 all_token_full 이면 자체 full-token
    # 로더(record [L,K,T,D] → token-pool → [L,K,D]), 아니면 기존 load_rollout(무수정 경로).
    require_mode = args.require_capture_token_mode
    rolls, fulltoken_flags = [], []
    contract = None  # (feature_kind, capture_layers, K, T, D) 전 pkl 일치 강제 (혼입 게이트 심화)
    for p in pkls:
        with open(p, "rb") as f:
            d = _pk.load(f)
        mode = d.get("capture_token_mode")
        if require_mode is not None and mode != require_mode:
            print(f"[require-capture-token-mode] {p}: capture_token_mode={mode!r} != "
                  f"{require_mode!r} — 구/신 캡처 혼입", file=sys.stderr)
            sys.exit(4)
        if mode == FULLTOKEN_MODE:
            kind = d.get("feature_kind")
            if require_mode is not None and kind != FULLTOKEN_KIND:
                print(f"[require-capture-token-mode] {p}: feature_kind={kind!r} != "
                      f"{FULLTOKEN_KIND!r}", file=sys.stderr)
                sys.exit(4)
            r = load_rollout_fulltoken(d, p, args.token_pool)
            sig = (kind, tuple(d.get("feature_axes") or []),
                   tuple(r["capture_layers"] or []),
                   int(r["dit_k"].shape[2]), int(r["token_count"] or -1),
                   int(r["dit_k"].shape[3]))
            if contract is None:
                contract = sig
            elif sig != contract:
                print(f"[contract 혼입] {p}: (kind,layers,K,T,D)={sig} != 첫 pkl {contract}",
                      file=sys.stderr)
                sys.exit(4)
            rolls.append(r)
            fulltoken_flags.append(True)
        else:
            rolls.append(load_rollout(p))
            fulltoken_flags.append(False)
    if manifest_rows:
        for r, m in zip(rolls, manifest_rows):
            r["success"] = m["label"]  # manifest 라벨이 pkl 내장값 override
            r["scene"] = m.get("scene", "")  # scene 보존 (gated 성립 게이트·LOO 용)
    if args.denoise == "per_step" and not all(fulltoken_flags):
        raise SystemExit("--denoise per_step 은 full-token pkl 전용 (구 캡처 pkl 혼입)")
    if args.stage1_quota_sweep and args.denoise != "per_step":
        # 배치되는 연산자(step별 M_k)와 동일한 객체의 quota 로 층을 골라야 함 —
        # pooled-stack conceptor 의 quota 는 다른 연산자 (Gate 2 높음#5)
        raise SystemExit("--stage1-quota-sweep 은 --denoise per_step 로 실행")
    if args.denoise in ("stack", "step0"):
        # K 축 재정의 (COAST 대조 — docs/collab/2026-07-10 §COAST 감사): denoise 평균을
        # 풀고 step별 개별 record(stack) 또는 step0 만 사용. phase/vl 은 record 수에 맞춰
        # 정렬 유지 (vl 중복은 R 에 무영향 — 동일 표본 복제).
        for r, p, is_full in zip(rolls, pkls, fulltoken_flags):
            if is_full:
                raw = r["dit_k"]  # [n,L,K,D] (token-pool 완료본)
            else:
                with open(p, "rb") as f:
                    d = _pk.load(f)
                raw = np.stack([np.asarray(x, dtype=np.float32) for x in d["hidden_states"]], axis=0)  # [n,L,K,D]
            n, L, K, D = raw.shape
            if args.denoise == "step0":
                r["dit"] = raw[:, :, 0, :]
            else:  # stack
                r["dit"] = raw.transpose(0, 2, 1, 3).reshape(n * K, L, D)
                r["phases"] = [ph for ph in r["phases"] for _ in range(K)]
                if r["vl"] is not None:
                    r["vl"] = np.repeat(r["vl"], K, axis=0)
                r["length"] = n * K
    if args.carve_window > 0:
        for r, p in zip(rolls, pkls):
            r["phases"] = carve_5phase(p, r["phases"], args.carve_window)
    cap = rolls[0]["capture_layers"]
    has_vl = rolls[0]["vl"] is not None
    layer_keys = list(range(rolls[0]["dit"].shape[1])) + (["VL"] if has_vl else [])
    if args.layers:
        wanted = {tok.strip() for tok in args.layers.split(",") if tok.strip()}
        layer_keys = [
            lk for lk in layer_keys
            if ("VL" if lk == "VL" else str(int(cap[lk]))) in wanted
        ]
        if not layer_keys:
            raise SystemExit(f"--layers {args.layers} 에 해당하는 capture layer 없음 (cap={cap})")
    if args.alphas == "table14":
        alphas = list(TABLE14_ALPHAS)
    else:
        alphas = [float(x) for x in args.alphas.split(",")] if args.alphas else DEFAULT_ALPHAS
    groups = args.groups.split(",")
    cell_id = args.cell.split("/")[-1]
    out_root = Path(args.out_dir) if args.out_dir else run_dir.parent / "analysis" / "conceptor_steering_n15" / cell_id
    n_s = sum(r["success"] for r in rolls); n_f = len(rolls) - n_s
    print(f"[cell {cell_id}] {len(rolls)} rollouts succ={n_s} fail={n_f}  layers={cap}+VL={has_vl}")

    if args.stage1_quota_sweep:
        # Stage1 (COAST A.10.2 변형, regime별·**배치 연산자 동일 객체 기준** — Gate 2 높음#5):
        # α₀ 고정, 전 CAP layer 에 대해 step k 별 C_steer 를 각각 fit 하고
        #   perm regime  = mean_k quota(C_steer^{(k)} | global records)
        #   gated regime = mean_{phase×k} quota(C_steer^{(phase,k)}) — phase×step 고정 그리드
        # phase 의 min-class 판정은 layer-무관(episode 구성 동일)이라 분모는 전 층 동일.
        a0 = float(args.stage1_alpha)
        phase_groups = [g for g in groups if g != "global"]
        K = rolls[0]["dit_k"].shape[2]
        rolls_by_k = [
            [{**r, "dit": r["dit_k"][:, :, k, :]} for r in rolls] for k in range(K)
        ]

        def _gather_capped(rolls_k, group, lk, success, cap_n):
            """episode 당 record 를 cap_n 개로 절단 (episode-equal 민감도용)."""
            out, n_eps = [], 0
            for r in rolls_k:
                if r["success"] != success:
                    continue
                recs = _roll_records(r, group, lk)
                if recs:
                    n_eps += 1
                    out.extend(recs[:cap_n])
            return (np.asarray(out, dtype=np.float64) if out else np.empty((0, 0))), n_eps

        def _min_ep_records(group, success):
            counts = [len(_roll_records(r, group, 0)) for r in rolls_by_k[0]
                      if r["success"] == success and _roll_records(r, group, 0)]
            return min(counts) if counts else 0

        def _steer_quota(rolls_k, group, lk, cap_n=None):
            if cap_n:
                Xs, ns_ep = _gather_capped(rolls_k, group, lk, 1, cap_n)
                Xf, nf_ep = _gather_capped(rolls_k, group, lk, 0, cap_n)
            else:
                Xs, ns_ep = gather_class_records_eps(rolls_k, group, lk, 1)
                Xf, nf_ep = gather_class_records_eps(rolls_k, group, lk, 0)
            if Xs.size == 0 or Xf.size == 0 or ns_ep < args.min_per_class or nf_ep < args.min_per_class:
                return None, ns_ep, nf_ep
            if Xs.shape[0] < 2 or Xf.shape[0] < 2:
                return None, ns_ep, nf_ep
            Cs = compute_conceptor(Xs, a0)
            Cf = compute_conceptor(Xf, a0)
            return conceptor_quota(and_conceptor(Cs, not_conceptor(Cf))), ns_ep, nf_ep

        table = {}
        for lk in layer_keys:
            if lk == "VL":
                continue  # Stage1 은 DiT 층 선택 전용
            tag = f"dit_L{cap[lk]}"
            row = {"layer": int(cap[lk]), "perm_quota": None, "perm_quota_per_step": [],
                   "perm_quota_epeq": None,
                   "gated_quota_mean": None, "gated_quota_mean_epeq": None,
                   "gated_per_phase": {}, "skipped_phases": []}
            perm_qs, perm_qs_eq = [], []
            cap_g = min(_min_ep_records("global", 1), _min_ep_records("global", 0)) or None
            for k in range(K):
                q, ns_ep, nf_ep = _steer_quota(rolls_by_k[k], "global", lk)
                if q is not None:
                    perm_qs.append(q)
                q_eq, _, _ = _steer_quota(rolls_by_k[k], "global", lk, cap_n=cap_g)
                if q_eq is not None:
                    perm_qs_eq.append(q_eq)
                row.setdefault("global_eps", {"s": ns_ep, "f": nf_ep})
            if len(perm_qs) == K:
                row["perm_quota_per_step"] = perm_qs
                row["perm_quota"] = float(np.mean(perm_qs))
            if len(perm_qs_eq) == K:
                row["perm_quota_epeq"] = float(np.mean(perm_qs_eq))
            phase_means, phase_means_eq = [], []
            for g in phase_groups:
                qs, qs_eq = [], []
                cap_p = min(_min_ep_records(g, 1), _min_ep_records(g, 0)) or None
                for k in range(K):
                    q, ns_g, nf_g = _steer_quota(rolls_by_k[k], g, lk)
                    if q is None:
                        row["skipped_phases"].append({"phase": g, "eps_s": ns_g, "eps_f": nf_g})
                        qs = []
                        break
                    qs.append(q)
                    q_eq, _, _ = _steer_quota(rolls_by_k[k], g, lk, cap_n=cap_p)
                    if q_eq is not None:
                        qs_eq.append(q_eq)
                if qs:
                    row["gated_per_phase"][g] = {"per_step": qs, "mean": float(np.mean(qs))}
                    phase_means.append(float(np.mean(qs)))
                    if len(qs_eq) == K:
                        phase_means_eq.append(float(np.mean(qs_eq)))
            if phase_means:
                row["gated_quota_mean"] = float(np.mean(phase_means))
            if phase_means_eq and len(phase_means_eq) == len(phase_means):
                row["gated_quota_mean_epeq"] = float(np.mean(phase_means_eq))
            # skipped_phases 중복 제거 (phase 당 1회)
            seen_ph = set()
            row["skipped_phases"] = [
                s for s in row["skipped_phases"]
                if not (s["phase"] in seen_ph or seen_ph.add(s["phase"]))
            ]
            table[tag] = row
            print(f"  [stage1 {tag}] perm_quota={row['perm_quota']} "
                  f"gated_mean={row['gated_quota_mean']} "
                  f"skipped={[s['phase'] for s in row['skipped_phases']]}")
        def _argmax(key):
            rows = [(t, r[key]) for t, r in table.items() if r[key] is not None]
            return max(rows, key=lambda x: x[1])[0] if rows else None
        report = {"cell": cell_id, "alpha0": a0, "denoise": "per_step",
                  "num_denoise_steps": int(K),
                  "token_pool": args.token_pool, "min_per_class_eps": args.min_per_class,
                  "groups": groups, "table": table,
                  "perm_argmax": _argmax("perm_quota"),
                  "gated_argmax": _argmax("gated_quota_mean"),
                  # episode-equal 민감도 (episode 당 record 를 최소 episode 수준으로 절단
                  # — 긴 timeout/dwell 의 covariance 지배 진단, Gate2 R2 중간#2)
                  "perm_argmax_epeq": _argmax("perm_quota_epeq"),
                  "gated_argmax_epeq": _argmax("gated_quota_mean_epeq"),
                  "layer_choice_stable_epeq": {
                      "perm": _argmax("perm_quota") == _argmax("perm_quota_epeq"),
                      "gated": _argmax("gated_quota_mean") == _argmax("gated_quota_mean_epeq"),
                  },
                  "note": "quota = 배치 연산자(step별 C_steer)와 동일 객체 기준; "
                          "epeq 불일치 시 layer 선택 불안정 — 사용자 게이트에서 병기 보고"}
        out_root.mkdir(parents=True, exist_ok=True)
        out_path = out_root / "stage1_quota_sweep.json"
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"[stage1] perm_argmax={report['perm_argmax']} "
              f"gated_argmax={report['gated_argmax']} -> {out_path}")
        return

    summary = {}
    common_meta = {"token_pool": args.token_pool if any(fulltoken_flags) else None,
                   "capture_token_mode": rolls[0].get("capture_token_mode"),
                   "token_count": rolls[0].get("token_count"),
                   "quota_floor": float(args.quota_floor)}
    for group in groups:
        for lk in layer_keys:
            tag = "vl" if lk == "VL" else f"dit_L{cap[lk]}"

            if args.denoise == "per_step":
                # -- Per-Step: step k 별 conceptor (pq3 신규 NPZ 계약) -------------
                if lk == "VL":
                    print(f"  [skip {group}/vl] per_step 은 DiT 전용 (VL 은 K 축 없음)")
                    continue
                K = rolls[0]["dit_k"].shape[2]
                # 클래스 게이트는 step 공통 (membership 동일) — step0 뷰로 검사
                rolls_0 = [{**r, "dit": r["dit_k"][:, :, 0, :]} for r in rolls]
                Xs, ns_ep = gather_class_records_eps(rolls_0, group, lk, 1)
                Xf, nf_ep = gather_class_records_eps(rolls_0, group, lk, 0)
                if Xs.size == 0 or Xf.size == 0 or ns_ep < args.min_per_class or nf_ep < args.min_per_class:
                    print(f"  [skip {group}/{tag}] eps s={ns_ep} f={nf_ep} < min {args.min_per_class}")
                    continue
                step_fits, step_meta, degenerate = {}, {}, None
                for k in range(K):
                    rolls_k = [{**r, "dit": r["dit_k"][:, :, k, :]} for r in rolls]
                    Xs_k, _ = gather_class_records_eps(rolls_k, group, lk, 1)
                    Xf_k, _ = gather_class_records_eps(rolls_k, group, lk, 0)
                    fits_k, meta_k = fit_one(Xs_k, Xf_k, alphas, OVERLAP_BAND, args.quota_floor)
                    if fits_k is None:
                        degenerate = k
                        break
                    step_fits[k], step_meta[k] = fits_k, meta_k
                if degenerate is not None:
                    # 부분 per-step NPZ 는 serve M_seq 계약 위반 — 전체 skip (배선 사고 방지)
                    print(f"  [skip {group}/{tag}] step{degenerate} 전 α quota<floor "
                          f"{args.quota_floor:g} (퇴화 — per-step NPZ 미생성)")
                    continue
                meta = {**common_meta, "cell": cell_id, "group": group, "layer_tag": tag,
                        "steering_layer": int(cap[lk]), "pathway": "dit",
                        "denoise_mode": "per_step", "num_denoise_steps": int(K),
                        "selected_alpha_per_step": {str(k): step_meta[k]["selected_alpha"] for k in range(K)},
                        "selection_mode_per_step": {str(k): step_meta[k]["selection_mode"] for k in range(K)},
                        "alpha_sweep_per_step": {str(k): step_meta[k]["alpha_sweep"] for k in range(K)},
                        "n_success_eps": ns_ep, "n_failure_eps": nf_ep,
                        "n_success": int(Xs.shape[0]), "n_failure": int(Xf.shape[0])}
                save_npz_per_step(out_root / group / tag, step_fits, meta)
                summary.setdefault(group, {})[tag] = {
                    "sel_alpha_per_step": meta["selected_alpha_per_step"],
                    "n_s": int(Xs.shape[0]), "n_f": int(Xf.shape[0]),
                    "n_s_eps": ns_ep, "n_f_eps": nf_ep}
                print(f"  [{group}/{tag}] per_step K={K} Ns={Xs.shape[0]}({ns_ep}ep) "
                      f"Nf={Xf.shape[0]}({nf_ep}ep) sel_alphas={meta['selected_alpha_per_step']}")
                continue

            Xs, ns_ep = gather_class_records_eps(rolls, group, lk, 1)
            Xf, nf_ep = gather_class_records_eps(rolls, group, lk, 0)
            # v2 게이트: 클래스별 기여 episode 수 기준 (record 수 아님 — 길이 과대가중 구멍 봉합)
            if Xs.size == 0 or Xf.size == 0 or ns_ep < args.min_per_class or nf_ep < args.min_per_class:
                print(f"  [skip {group}/{tag}] eps s={ns_ep} f={nf_ep} < min {args.min_per_class}")
                continue
            fits, meta = fit_one(Xs, Xf, alphas, OVERLAP_BAND, args.quota_floor)
            if fits is None:
                print(f"  [skip {group}/{tag}] 전 α quota<floor {args.quota_floor:g} (퇴화)")
                continue
            meta.update({**common_meta, "cell": cell_id, "group": group, "layer_tag": tag,
                         "steering_layer": None if lk == "VL" else int(cap[lk]),
                         "pathway": "vl" if lk == "VL" else "dit",
                         "denoise_mode": args.denoise,
                         "n_success_eps": ns_ep, "n_failure_eps": nf_ep})
            save_npz(out_root / group / tag, fits, meta)
            summary.setdefault(group, {})[tag] = {"sel_alpha": meta["selected_alpha"],
                                                  "n_s": meta["n_success"], "n_f": meta["n_failure"],
                                                  "n_s_eps": ns_ep, "n_f_eps": nf_ep}
            print(f"  [{group}/{tag}] Ns={Xs.shape[0]}({ns_ep}ep) Nf={Xf.shape[0]}({nf_ep}ep) "
                  f"sel_alpha={meta['selected_alpha']:g} "
                  f"overlap@sel={[s['overlap'] for s in meta['alpha_sweep'] if s['alpha']==meta['selected_alpha']][0]:.3f}")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "fit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    # 사용 표본 기록 (검증 가능성: split 교집합 게이트가 content 서명으로 대조)
    inputs = {"cell": cell_id, "manifest": args.manifest, "min_per_class_eps": args.min_per_class,
              "episodes": [{"pkl": str(p), "label": int(r["success"]),
                            "scene": r.get("scene", ""), "sig": _content_sig(p)}
                           for p, r in zip(pkls, rolls)]}
    (out_root / "fit_inputs.json").write_text(json.dumps(inputs, indent=2, ensure_ascii=False))
    if not summary:
        print(f"[empty] 유효 group×layer 0개 (episode min-class 미달) -> {out_root}")
        sys.exit(3)
    print(f"[done] -> {out_root}")


if __name__ == "__main__":
    main()
