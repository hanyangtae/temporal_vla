"""exp5-2 clean-vs-perturbed setM fit — 섭동 회복 방향 (07-27).

exp4-1 의 setM(setpoint mean-diff) 을 **succ/fail 대조가 아니라 clean/perturbed 대조**로
fit 한다. 회복 방향은 perturbed → clean 이므로

    r̂ = normalize(μ_clean − μ_perturbed),   s = μ_clean · r̂
    h' = h − β[(h·r̂) − s]·r̂

perturbed 레코드는 (h·r̂) < s 이므로 delta 가 음수 → h 가 r̂ 방향(=clean 쪽)으로 이동한다.
clean 레코드는 (h·r̂) ≈ s 라 개입량이 자기소멸한다.

★ 토큰 공간 주의 (이 데이터 고유의 함정)
  exp42_induced 캡처는 ``capture_token_mode = "action_token_mean"`` —
  feature_kind = groot_n15_dit_block_residual_action_tokens_denoise 로 **action token 16개
  평균**만 저장돼 있다 (exp4-1 의 all_token_full [T=49] 이 아니다). 따라서 토큰 위치별
  setpoint s_t 를 데이터에서 산출할 수 없다. 두 가지 저장 모드를 둔다:

    --seg-mode action_only (기본) : v_seg 3행 모두 r̂, s_tok 은 action 구간(33:49)에만 s,
        seg_mask = [0, 0, 1] → **fit 공간(action token 평균)과 적용 공간(action token)이
        일치**. state/future 토큰은 게인 0 이라 무개입.
    --seg-mode all : 단일 세그먼트 [0,T) · mask 1 · s_tok 전부 s. 요청 계약("전 토큰 한
        세그먼트")과 정확히 일치하지만 action-token 공간에서 fit 한 방향을 state/future
        토큰에도 씌우는 공간 불일치가 있다 (exp4-1 v1 pooled 회귀와 동종). 탐색용.

  serve 계약: SetpointSteering.set_segment / load_steering_segment
  (alpha0_v_seg [S,D] · alpha0_s_tok [T] · alpha0_seg_bounds [S,2] · alpha0_seg_mask [S]).
  seg_mask 는 0/1 플래그가 아니라 **세그먼트별 게인 승수**다 (위약 dose-match 에 사용).
  디렉토리 계약은 exp4-1 permanent 와 동일: <out>/<cfg>/L<lyr>/setM/steer/dit_L<lyr>/.
  serve 는 ``--steering-op setpoint_seg --steering-token-select all`` 필수.

★ ``--pathway vl`` (exp5-2 VL 확장, 07-27)
  feature = 캡처 ``vl_hidden_states`` (= action_head.vlln 출력의 **토큰 평균**,
  vl_feature_kind=groot_n15_vlln_seq_meanpool, D=2048). layer 개념이 없어 출력은
  ``<out>/<cfg>/VLsetM/{setM,setM_pl}/conceptors.npz`` (pooled 계약:
  ``alpha0_v_steer`` [D] 단위벡터 + ``alpha0_s`` 스칼라) + sibling metadata.json.
  serve 는 ``--steering-npz <...>/conceptors.npz --steering-op setpoint_vl
  --steering-pathway vl`` (SetpointSteering pathway="vl" — 토큰 평균을 setpoint 로
  이동, 토큰 내 분산 보존). 세그먼트/토큰위치 setpoint 는 쓰지 않는다 (fit 공간이
  이미 토큰 평균이라 pooled 가 정확히 일치하는 공간).

위약(setM_pl): clean/perturbed **episode 라벨 순열**로 같은 파이프라인 fit. 실제 클래스 갭이
없어 raw 이동량이 작으므로, held-out dose 중앙값 비로 seg_mask 게인을 맞춘다 (exp4-1
select_placebo_seg 와 같은 규약). VL(pooled)은 게인 필드가 없으므로 **setpoint 를 평행이동**해
dose 를 맞춘다: s' = median(proj_pert) + dose_real → median|proj−s'| = dose_real 이고
개입 부호(+r̂ 방향 밀기)도 real 과 같다.

  docker exec -i lerobot python fit_setm_induced.py \
      --clean-dir <baseline_cap/raw_rollouts/<task>/<cell>> \
      --capture-root <p0/capture> --record-start <p0/manifests/rs_all.tsv> \
      --configs c1_s200,p1_d003,p2_f040d2 --layers 8,10,12 \
      --placebo-seed 1 --out-root <exp52/fits_ppcc>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from induced_common import load_roll_any, read_record_start  # noqa: E402

# GR00T N1.5 action head 토큰 레이아웃 (exp4-1 fit_mean_diff.SEGMENTS 와 동일 규약)
SEGMENTS = (("state", 0, 1), ("future", 1, 33), ("action", 33, 49))
T_TOKENS = SEGMENTS[-1][2]
N_PL_CAND = 64          # 위약 순열 후보 수 (|cos| 최소 후보 선택)
PL_COS_MAX = 0.30       # 위약 방향 준직교 상한


def _ep_of(p: Path) -> int:
    return int(re.search(r"--ep(\d+)--", p.name).group(1))


def _feat(r: dict, lk) -> np.ndarray:
    """[n_record, D] — DiT block residual (denoise pooled, action-token mean)."""
    if lk == "VL":
        return np.asarray(r["vl"], dtype=np.float32)
    return r["dit"][:, r["capture_layers"].index(lk), :].astype(np.float32)


def _sub(X: np.ndarray, k: int) -> np.ndarray:
    """episode 당 균등 k 서브샘플 (dwell/길이 가중 통제 — layer_geometry_probe 와 동일)."""
    n = X.shape[0]
    if n <= k:
        return X
    return X[np.linspace(0, n - 1, k).round().astype(int)]


def _split_keep(ep: int, split: str) -> bool:
    if split == "all":
        return True
    return (ep % 2 == 0) if split == "even" else (ep % 2 == 1)


def _mean_diff(Xc: np.ndarray, Xp: np.ndarray) -> tuple[np.ndarray, float, float]:
    """r̂ = normalize(μ_clean − μ_pert), s = μ_clean·r̂, 그리고 fit d'."""
    mc, mp = Xc.mean(axis=0), Xp.mean(axis=0)
    d = mc - mp
    nrm = float(np.linalg.norm(d))
    r = d / max(nrm, 1e-12)
    s = float(mc @ r)
    pc, pp = Xc @ r, Xp @ r
    pooled = float(np.sqrt(0.5 * (pc.var() + pp.var())))
    dprime = float((pc.mean() - pp.mean()) / max(pooled, 1e-12))
    return r.astype(np.float64), s, dprime


def _auroc(s0: np.ndarray, s1: np.ndarray) -> float:
    s = np.concatenate([s0, s1])
    n0, n1 = len(s0), len(s1)
    if n0 == 0 or n1 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def _dose(X: np.ndarray, r: np.ndarray, s: float) -> dict:
    """|（h·r̂) − s| 요약 — β=1 에서의 실제 이동 크기 (serve delta 노름과 동일)."""
    d = np.abs(X @ r - s)
    q1, q3 = np.percentile(d, [25, 75])
    return {"median": float(np.median(d)), "iqr": [float(q1), float(q3)],
            "mean": float(d.mean()), "n": int(d.shape[0])}


def build_seg(r: np.ndarray, s: float, seg_mode: str, gain: float = 1.0):
    """(v_seg [S,D], s_tok [T], bounds [S,2], mask [S]) — serve 계약 텐서."""
    D = r.shape[0]
    s_tok = np.zeros(T_TOKENS, dtype=np.float32)
    if seg_mode == "all":
        v_seg = r.reshape(1, D).astype(np.float32)
        bounds = np.asarray([[0, T_TOKENS]], dtype=np.int32)
        mask = np.asarray([gain], dtype=np.float32)
        s_tok[:] = s
    else:  # action_only — fit 공간(action token 평균)과 적용 공간 일치
        v_seg = np.repeat(r.reshape(1, D), len(SEGMENTS), axis=0).astype(np.float32)
        bounds = np.asarray([[lo, hi] for _n, lo, hi in SEGMENTS], dtype=np.int32)
        mask = np.zeros(len(SEGMENTS), dtype=np.float32)
        mask[-1] = gain
        lo, hi = SEGMENTS[-1][1], SEGMENTS[-1][2]
        s_tok[lo:hi] = s
    return v_seg, s_tok, bounds, mask


def save_arm_vl(base: Path, r: np.ndarray, s: float, meta: dict) -> Path:
    """VL pooled setpoint NPZ — <base>/conceptors.npz + metadata.json.

    serve 계약: load_steering_setpoint (``alpha{a}_v_steer`` 단위벡터 + ``alpha{a}_s``),
    metadata.json 의 ``selected_alpha=0`` 으로 α 를 명시(첫-키 폴백 회피).
    """
    base.mkdir(parents=True, exist_ok=True)
    npz = base / "conceptors.npz"
    np.savez_compressed(
        npz,
        alpha0_v_steer=r.astype(np.float32),
        alpha0_s=np.asarray(s, dtype=np.float32),
    )
    (base / "metadata.json").write_text(json.dumps(
        {**meta, "selected_alpha": 0, "op": "setpoint_vl", "pathway": "vl",
         "token_pool": "vlln_seq_meanpool"}, indent=2, ensure_ascii=False))
    return npz


def save_arm(base: Path, layer: int, seg, meta: dict) -> Path:
    """<base>/steer/dit_L<layer>/{conceptors.npz, metadata.json} (exp4-1 permanent 계약)."""
    v_seg, s_tok, bounds, mask = seg
    d = base / "steer" / f"dit_L{layer}"
    d.mkdir(parents=True, exist_ok=True)
    npz = d / "conceptors.npz"
    np.savez_compressed(
        npz,
        alpha0_v_seg=v_seg.astype(np.float32),
        alpha0_s_tok=s_tok.astype(np.float32),
        alpha0_seg_bounds=bounds.astype(np.int32),
        alpha0_seg_mask=mask.astype(np.float32),
    )
    (d / "metadata.json").write_text(json.dumps(
        {**meta, "selected_alpha": 0, "op": "setpoint_seg",
         "token_pool": "action_token_mean",
         "segments": ([f"{n}" for n, _l, _h in SEGMENTS]
                      if meta.get("seg_mode") != "all" else ["all"]),
         "seg_mask": [float(x) for x in mask]}, indent=2, ensure_ascii=False))
    return npz


def collect(paths, rs, lk, k_sub, split, use_rs: bool):
    """(pooled X [N,D], episode 리스트, raw record 수, per-episode 서브샘플 블록 리스트)."""
    blocks, eps, n_raw = [], [], 0
    for p in paths:
        ep = _ep_of(p)
        if not _split_keep(ep, split):
            continue
        r = load_roll_any(p)
        X = _feat(r, lk)
        start = rs.get(str(p.resolve()), 0) if use_rs else 0
        X = X[start:]
        if X.shape[0] == 0:
            continue
        n_raw += X.shape[0]
        blocks.append(_sub(X, k_sub))
        eps.append(ep)
    if not blocks:
        return None, [], 0, []
    return np.concatenate(blocks, axis=0), eps, n_raw, blocks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean-dir", required=True, help="clean baseline 캡처 pkl 디렉토리")
    ap.add_argument("--clean-glob", default="task*--ep*--succ1.pkl",
                    help="clean 측 pkl glob. 기본 = 성공만(ppcc 선례). clean-succ 표본이 빈약한 "
                         "cell 은 'task*--ep*--succ*.pkl' 로 전 라벨 참고 fit 을 따로 산출해 "
                         "clean 실패 rollout 이 clean 분포를 오염하는지 비교한다 (exp5-2 mixer).")
    ap.add_argument("--capture-root", required=True, help="섭동 캡처 루트 (<cfg>/raw_rollouts/...)")
    ap.add_argument("--record-start", required=True, help="perturbed 시간분리 절단 tsv")
    ap.add_argument("--configs", default="", help="콤마 리스트 (기본 = capture-root 전부)")
    ap.add_argument("--layers", default="8,10,12", help="DiT block layer 콤마 리스트")
    ap.add_argument("--pathway", choices=("dit", "vl"), default="dit",
                    help="dit=DiT block residual(layer별) | vl=vlln 토큰평균(layer 없음)")
    ap.add_argument("--split", choices=("even", "odd", "all"), default="even",
                    help="fit 에 쓸 episode 패리티 (held-out = 반대)")
    ap.add_argument("--k-sub", type=int, default=20, help="episode 당 균등 서브샘플 수")
    ap.add_argument("--placebo-seed", type=int, default=0, help=">0 이면 라벨순열 위약 생성")
    ap.add_argument("--seg-mode", choices=("action_only", "all"), default="action_only")
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    rs = read_record_start(args.record_start)
    is_vl = args.pathway == "vl"
    # VL 은 layer 축이 없다 (vlln 단일 지점) — 루프를 "VL" 하나로 축약.
    layers = ["VL"] if is_vl else [int(x) for x in args.layers.split(",") if x.strip()]
    cfgs = [c.strip() for c in args.configs.split(",") if c.strip()]
    cap_root = Path(args.capture_root)
    if not cfgs:
        cfgs = sorted(d.name for d in cap_root.iterdir() if d.is_dir())
    held = {"even": "odd", "odd": "even", "all": "all"}[args.split]

    clean_paths = sorted(Path(args.clean_dir).glob(args.clean_glob))
    if not clean_paths:
        raise SystemExit(f"clean pkl 0개: {args.clean_dir}/{args.clean_glob}")
    out_root = Path(args.out_root)
    print(f"clean {len(clean_paths)} pkl (glob={args.clean_glob}) · cfgs={cfgs} · pathway={args.pathway} "
          f"· layers={layers} · fit split={args.split} held={held} "
          f"· seg_mode={'n/a(pooled)' if is_vl else args.seg_mode}")

    rng_global = np.random.default_rng(args.placebo_seed)
    summary = []
    for cfg in cfgs:
        pert_paths = sorted((cap_root / cfg).glob("raw_rollouts/*/*/task*--ep*--succ*.pkl"))
        if not pert_paths:
            print(f"[{cfg}] pkl 0개 — skip")
            continue
        for lyr in layers:
            Xc, epc, nrc, _ = collect(clean_paths, rs, lyr, args.k_sub, args.split, False)
            Xp, epp, nrp, _ = collect(pert_paths, rs, lyr, args.k_sub, args.split, True)
            Hc, _epch, _, Bc = collect(clean_paths, rs, lyr, args.k_sub, held, False)
            Hp, _epph, _, Bp = collect(pert_paths, rs, lyr, args.k_sub, held, True)
            if Xc is None or Xp is None:
                print(f"[{cfg}/L{lyr}] fit 표본 없음 — skip")
                continue

            r, s, dprime = _mean_diff(Xc, Xp)
            cell = out_root / cfg / ("VLsetM" if is_vl else f"L{lyr}")
            cell.mkdir(parents=True, exist_ok=True)
            meta = {
                "exp": "exp5-2", "contrast": "clean_vs_perturbed",
                "cfg": cfg, "pathway": args.pathway, "layer": lyr,
                "split_fit": args.split, "split_held": held,
                "k_sub": args.k_sub,
                "seg_mode": (None if is_vl else args.seg_mode),
                "n_ep_clean": len(epc), "n_ep_perturbed": len(epp),
                "n_rec_clean_fit": int(Xc.shape[0]), "n_rec_pert_fit": int(Xp.shape[0]),
                "n_rec_clean_raw": nrc, "n_rec_pert_raw": nrp,
                "dim": int(r.shape[0]), "r_norm_raw": float(np.linalg.norm(Xc.mean(0) - Xp.mean(0))),
                "setpoint_s": s, "dprime_fit": dprime,
                "token_layout": (None if is_vl
                                 else {"T": T_TOKENS, "segments": [list(x) for x in SEGMENTS]}),
                "clean_dir": str(args.clean_dir), "clean_glob": args.clean_glob,
                "n_pkl_clean_total": len(clean_paths),
                "capture_root": str(args.capture_root),
                "record_start": str(args.record_start),
            }
            if is_vl:
                npz_real = save_arm_vl(cell / "setM", r, s, meta)
            else:
                npz_real = save_arm(cell / "setM", lyr, build_seg(r, s, args.seg_mode), meta)
            shutil.copyfile(npz_real, cell / "setM.npz")

            # ---- held-out 검증: 부호 방향 + dose -----------------------------------
            dose = {}
            if Hc is not None and Hp is not None:
                pc, pp = Hc @ r, Hp @ r
                dose["real"] = {
                    "clean": _dose(Hc, r, s), "perturbed": _dose(Hp, r, s),
                    "proj_mean_clean": float(pc.mean()), "proj_mean_perturbed": float(pp.mean()),
                    "sign_ok": bool(pc.mean() > pp.mean()),
                    "dprime_heldout": float(
                        (pc.mean() - pp.mean())
                        / max(float(np.sqrt(0.5 * (pc.var() + pp.var()))), 1e-12)),
                    # 양성 클래스 = clean (r̂ 는 clean 쪽이 큼) → 1.0 이 완전 분리
                    "auroc_heldout_record": _auroc(pp, pc),
                    "auroc_heldout_episode": _auroc(
                        np.asarray([float(b.mean(axis=0) @ r) for b in Bp]),
                        np.asarray([float(b.mean(axis=0) @ r) for b in Bc])),
                }
            meta["heldout"] = dose.get("real")

            # ---- 위약: episode 라벨 순열 ------------------------------------------
            pl_info = None
            if args.placebo_seed > 0:
                pl_info = fit_placebo(clean_paths, pert_paths, rs, lyr, args, held,
                                      r, s, cell, meta, rng_global, Hc, Hp)
                dose["placebo"] = pl_info

            (cell / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
            (cell / "dose_check.json").write_text(json.dumps(dose, indent=2, ensure_ascii=False))
            hd = dose.get("real")
            tag = f"{cfg}/{'VL' if is_vl else f'L{lyr}'}"
            print(f"[{tag}] ‖Δμ‖={meta['r_norm_raw']:.3f} s={s:.3f} d'fit={dprime:.2f} "
                  f"held AUROC rec={hd['auroc_heldout_record']:.3f} "
                  f"ep={hd['auroc_heldout_episode']:.3f} sign_ok={hd['sign_ok']} "
                  f"dose_med clean={hd['clean']['median']:.3f} pert={hd['perturbed']['median']:.3f}"
                  if hd else f"[{cfg}/L{lyr}] held-out 없음")
            summary.append({"cfg": cfg, "layer": lyr, **({} if not hd else {
                "auroc_rec": hd["auroc_heldout_record"],
                "auroc_ep": hd["auroc_heldout_episode"], "sign_ok": hd["sign_ok"],
                "dose_pert": hd["perturbed"]["median"], "dose_clean": hd["clean"]["median"]}),
                "placebo_dose": (pl_info or {}).get("dose_perturbed_median"),
                "placebo_gain": (pl_info or {}).get("gain")})

    out_root.mkdir(parents=True, exist_ok=True)
    # pathway 별로 분리 — 같은 out-root 에 dit/vl fit 을 얹어도 서로 덮어쓰지 않는다.
    sm_path = out_root / ("fit_summary_vl.json" if is_vl else "fit_summary.json")
    sm_path.write_text(json.dumps(
        {"argv": vars(args), "rows": summary}, indent=2, ensure_ascii=False))
    print(f"wrote {sm_path} ({len(summary)} rows)")
    return 0


def fit_placebo(clean_paths, pert_paths, rs, lyr, args, held, r_real, s_real,
                cell: Path, meta: dict, rng, Hc, Hp) -> dict:
    """episode 라벨 순열 위약 — 방향 준직교 후보 선택 + held-out dose-match 게인."""
    # fit split 의 episode 별 서브샘플 블록을 (라벨과 함께) 모아 순열
    blocks = []
    for p in clean_paths:
        ep = _ep_of(p)
        if not _split_keep(ep, args.split):
            continue
        X = _sub(_feat(load_roll_any(p), lyr), args.k_sub)
        if X.shape[0]:
            blocks.append(X)
    n_clean = len(blocks)
    for p in pert_paths:
        ep = _ep_of(p)
        if not _split_keep(ep, args.split):
            continue
        X = _feat(load_roll_any(p), lyr)[rs.get(str(p.resolve()), 0):]
        if X.shape[0]:
            blocks.append(_sub(X, args.k_sub))
    n_all = len(blocks)
    best = None
    for _ in range(N_PL_CAND):
        perm = rng.permutation(n_all)
        A = np.concatenate([blocks[i] for i in perm[:n_clean]], axis=0)
        B = np.concatenate([blocks[i] for i in perm[n_clean:]], axis=0)
        rp, sp, dp = _mean_diff(A, B)
        c = abs(float(rp @ r_real))
        if best is None or c < best[0]:
            best = (c, rp, sp, dp, perm.tolist())
        if c <= PL_COS_MAX:
            break
    cos_pl, rp, sp, dp, perm = best
    is_vl = getattr(args, "pathway", "dit") == "vl"
    dose_real_p = float(np.median(np.abs(Hp @ r_real - s_real))) if Hp is not None else float("nan")
    dose_pl_p = float(np.median(np.abs(Hp @ rp - sp))) if Hp is not None else float("nan")
    gain = float(dose_real_p / max(dose_pl_p, 1e-12)) if Hp is not None else 1.0
    sp_raw = sp
    if is_vl and Hp is not None:
        # VL pooled 는 게인 필드가 없다 → setpoint 평행이동으로 dose 매칭.
        # s' = median(proj_pert) + dose_real ⇒ median|proj−s'| = dose_real, 부호도 real 과 동일
        # (perturbed 가 s' 아래 → h 가 +r̂_pl 로 밀림).
        sp = float(np.median(Hp @ rp) + dose_real_p)
        gain = 1.0
    meta_pl = {**meta, "arm": "setM_pl", "placebo_seed": args.placebo_seed,
               "placebo_cos_with_real": cos_pl, "placebo_dprime_fit": dp,
               "placebo_setpoint_s": sp, "placebo_setpoint_s_raw": sp_raw,
               "placebo_gain": gain, "placebo_dose_match": ("setpoint_shift" if is_vl
                                                            else "seg_mask_gain"),
               "placebo_perm_head": perm[:12]}
    meta_pl.pop("heldout", None)
    if is_vl:
        npz_pl = save_arm_vl(cell / "setM_pl", rp, sp, meta_pl)
    else:
        npz_pl = save_arm(cell / "setM_pl", lyr, build_seg(rp, sp, args.seg_mode, gain), meta_pl)
    shutil.copyfile(npz_pl, cell / "setM_pl.npz")
    return {
        "cos_with_real": cos_pl, "dprime_fit": dp, "setpoint_s": sp,
        "setpoint_s_raw": sp_raw, "gain": gain,
        "dose_match": ("setpoint_shift" if is_vl else "seg_mask_gain"),
        "dose_perturbed_median": (float(np.median(np.abs(Hp @ rp - sp)))
                                  if Hp is not None else float("nan")),
        "dose_perturbed_median_raw": dose_pl_p,
        "dose_real_perturbed_median": dose_real_p,
        "dose_perturbed": (_dose(Hp, rp, sp) if Hp is not None else None),
        "dose_clean": (_dose(Hc, rp, sp) if Hc is not None else None),
        "auroc_heldout_record": (_auroc(Hp @ rp, Hc @ rp)
                                 if (Hc is not None and Hp is not None) else None),
    }


if __name__ == "__main__":
    sys.exit(main())
