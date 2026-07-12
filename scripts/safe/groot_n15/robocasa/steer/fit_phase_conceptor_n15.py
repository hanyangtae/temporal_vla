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

DEFAULT_ALPHAS = [0.1, 0.3, 1.0, 3.0, 10.0]
OVERLAP_BAND = (0.85, 0.95)  # COAST A.10.2 Stage2


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


def select_alpha(sweep, band):
    lo, hi = band
    in_band = sorted((r["alpha"] for r in sweep if lo <= r["overlap"] <= hi))
    if in_band:
        return in_band[0], "band"
    return min(sweep, key=lambda r: min(abs(r["overlap"] - lo), abs(r["overlap"] - hi)))["alpha"], "closest"


def fit_one(Xs, Xf, alphas, band):
    """succ/fail 표본 → alpha별 C_steer + sweep. 반환 (fits{alpha:dict}, meta)."""
    sweep, cache = [], {}
    for a in alphas:
        Cs = compute_conceptor(Xs, a)
        Cf = compute_conceptor(Xf, a)
        cache[a] = (Cs, Cf)
        sweep.append({"alpha": float(a), "overlap": conceptor_overlap(Cs, Cf),
                      "quota_success": conceptor_quota(Cs), "quota_failure": conceptor_quota(Cf)})
    sel_alpha, sel_mode = select_alpha(sweep, band)
    fits = {}
    # 선택 alpha 를 반드시 첫 키로 저장 (serve 의 첫-키 폴백이 선택값을 집도록 — set 순회 금지)
    default_alpha = alphas[1] if len(alphas) > 1 else alphas[0]
    save_alphas = [sel_alpha] + ([default_alpha] if default_alpha != sel_alpha else [])
    for a in save_alphas:
        Cs, Cf = cache[a]
        Csteer = and_conceptor(Cs, not_conceptor(Cf))
        fits[a] = {"C_steer": Csteer, "C_success": Cs, "C_failure": Cf,
                   "quota_steer": conceptor_quota(Csteer)}
    return fits, {"alpha_sweep": sweep, "selected_alpha": float(sel_alpha), "selection_mode": sel_mode,
                  "quota_steer": {f"{a:g}": g["quota_steer"] for a, g in fits.items()},
                  "n_success": int(Xs.shape[0]), "n_failure": int(Xf.shape[0]), "feature_dim": int(Xs.shape[1])}


def _content_sig(p: Path) -> str:
    """pkl content 서명: size + 앞/뒤 1MB sha256 (동일 rollout 재유입 탐지용, 전체 해시는 IO 과다)."""
    h = hashlib.sha256()
    size = p.stat().st_size
    h.update(str(size).encode())
    with open(p, "rb") as f:
        h.update(f.read(1 << 20))
        if size > (2 << 20):
            f.seek(-(1 << 20), 2)
            h.update(f.read())
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


def main():
    ap = argparse.ArgumentParser(description="N1.5 phase-event 대조 conceptor fit")
    ap.add_argument("--run-dir", default="outputs/eval/robocasa/groot_n15/phase_event_aligned_4cell/raw_rollouts")
    ap.add_argument("--cell", required=True, help="예: PickPlaceCounterToCabinet/ppcc_bread")
    ap.add_argument("--groups", default="global,transport,reach-to-object")
    ap.add_argument("--alphas", default=None)
    ap.add_argument("--min-per-class", type=int, default=3,
                    help="클래스별 최소 기여 episode 수 (v2: record 수 아님)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--denoise", choices=["pool", "stack", "step0"], default="pool",
                    help="K(denoise) 축 처리: pool=평균(현행) | stack=step별 개별 record"
                         "(COAST Global 충실 — inference step당 K개 표본) | step0=첫 step만")
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
    rolls = [load_rollout(p) for p in pkls]
    if manifest_rows:
        for r, m in zip(rolls, manifest_rows):
            r["success"] = m["label"]  # manifest 라벨이 pkl 내장값 override
    if args.denoise != "pool":
        # K 축 재정의 (COAST 대조 — docs/collab/2026-07-10 §COAST 감사): load_rollout 의
        # denoise 평균을 풀고 step별 개별 record(stack) 또는 step0 만 사용. phase/vl 은
        # record 수에 맞춰 정렬 유지 (vl 중복은 R 에 무영향 — 동일 표본 복제).
        import pickle as _pk
        for r, p in zip(rolls, pkls):
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
    alphas = [float(x) for x in args.alphas.split(",")] if args.alphas else DEFAULT_ALPHAS
    groups = args.groups.split(",")
    cell_id = args.cell.split("/")[-1]
    out_root = Path(args.out_dir) if args.out_dir else run_dir.parent / "analysis" / "conceptor_steering_n15" / cell_id
    n_s = sum(r["success"] for r in rolls); n_f = len(rolls) - n_s
    print(f"[cell {cell_id}] {len(rolls)} rollouts succ={n_s} fail={n_f}  layers={cap}+VL={has_vl}")

    summary = {}
    for group in groups:
        for lk in layer_keys:
            tag = "vl" if lk == "VL" else f"dit_L{cap[lk]}"
            Xs, ns_ep = gather_class_records_eps(rolls, group, lk, 1)
            Xf, nf_ep = gather_class_records_eps(rolls, group, lk, 0)
            # v2 게이트: 클래스별 기여 episode 수 기준 (record 수 아님 — 길이 과대가중 구멍 봉합)
            if Xs.size == 0 or Xf.size == 0 or ns_ep < args.min_per_class or nf_ep < args.min_per_class:
                print(f"  [skip {group}/{tag}] eps s={ns_ep} f={nf_ep} < min {args.min_per_class}")
                continue
            fits, meta = fit_one(Xs, Xf, alphas, OVERLAP_BAND)
            meta.update({"cell": cell_id, "group": group, "layer_tag": tag,
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
              "episodes": [{"pkl": str(p), "label": int(r["success"]), "sig": _content_sig(p)}
                           for p, r in zip(pkls, rolls)]}
    (out_root / "fit_inputs.json").write_text(json.dumps(inputs, indent=2, ensure_ascii=False))
    if not summary:
        print(f"[empty] 유효 group×layer 0개 (episode min-class 미달) -> {out_root}")
        sys.exit(3)
    print(f"[done] -> {out_root}")


if __name__ == "__main__":
    main()
